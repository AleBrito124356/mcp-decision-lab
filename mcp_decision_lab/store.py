"""Process-safe JSON persistence for decision sessions.

Several MCP clients (Claude Desktop, Claude Code, an IDE...) can each launch
their own ``mcp-decision-lab`` process pointed at the same data directory. The
v0.1 store loaded ``decisions.json`` once and rewrote the whole file from
memory on every save, so the last writer silently erased the other processes'
sessions. This module fixes that with three rules:

1. Every mutation runs under an exclusive cross-process lock
   (``decisions.json.lock``, created with ``O_CREAT | O_EXCL``, stdlib only).
2. Inside the lock the store is re-read from disk, the mutation is applied to
   that fresh copy, and the result is written atomically (temp file +
   ``os.replace``). Disk is the source of truth, so sessions created by other
   processes are kept.
3. Reads refresh the in-memory copy whenever the file's identity/mtime/size
   changes, so a session created in one client is visible in another.

A lock left behind by a crashed process is recovered once it is older than
``stale_after`` seconds (critical sections take milliseconds).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

STORE_VERSION = 2
"""On-disk format version. v1 (0.1.x) files are migrated transparently."""

DECISION_SCHEMA_VERSION = 1
"""Per-decision record version, stamped on every record for later migrations."""

_REQUIRED_DECISION_KEYS = ("id", "question", "options", "criteria", "scores")
_DIRTY = (-1, -1, -1)  # a stat signature no real file can have

T = TypeVar("T")


class StoreError(ValueError):
    """The store file is unreadable, malformed, or locked for too long."""


class FileLock:
    """Exclusive inter-process lock built on ``O_CREAT | O_EXCL`` (stdlib only).

    Re-entrant within one thread is *not* supported; :class:`JsonStore` only
    takes it around short critical sections.
    """

    def __init__(
        self,
        path: Path,
        timeout: float = 10.0,
        stale_after: float = 30.0,
        poll: float = 0.005,
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.stale_after = stale_after
        self.poll = poll
        self._token: str | None = None

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        token = f"{os.getpid()}:{uuid.uuid4().hex}"
        delay = self.poll
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                self._break_if_stale()
            except PermissionError:
                # Windows reports a lock file that is being deleted as
                # "access denied" rather than "exists"; just retry.
                pass
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(f"{token}\n{time.time():.6f}\n")
                self._token = token
                return
            if time.monotonic() >= deadline:
                raise StoreError(
                    f"Timed out after {self.timeout:g}s waiting for the store lock "
                    f"{self.path}. Another mcp-decision-lab process is holding it; if "
                    "none is running, delete that lock file."
                )
            time.sleep(delay)
            delay = min(delay * 2, 0.05)

    def release(self) -> None:
        if self._token is None:
            return
        self._token = None
        for _ in range(50):
            try:
                os.unlink(self.path)
                return
            except FileNotFoundError:
                return
            except PermissionError:  # pragma: no cover - Windows sharing race
                time.sleep(0.01)

    def _read(self, path: Path) -> tuple[str, float] | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            return None
        parts = text.split()
        if len(parts) >= 2:
            try:
                return parts[0], float(parts[1])
            except ValueError:
                pass
        # Half-written or foreign content: fall back to the file's mtime.
        try:
            return text, path.stat().st_mtime
        except OSError:
            return None

    def _break_if_stale(self) -> None:
        info = self._read(self.path)
        if info is None:
            return
        token, stamp = info
        if time.time() - stamp < self.stale_after:
            return
        # Move the stale lock aside under a unique name, then make sure we
        # moved the lock we judged stale and not a fresh one that replaced it.
        aside = self.path.with_name(f"{self.path.name}.stale-{uuid.uuid4().hex}")
        try:
            os.rename(self.path, aside)
        except OSError:
            return
        moved = self._read(aside)
        if moved is not None and moved[0] != token:  # pragma: no cover - tiny race
            try:
                os.rename(aside, self.path)  # give the fresh lock back
                return
            except OSError:
                pass
        try:
            os.unlink(aside)
        except OSError:  # pragma: no cover
            pass

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _replace_with_retry(src: Path, dst: Path, attempts: int = 100) -> None:
    """``os.replace`` that tolerates Windows' transient sharing violations.

    On Windows, replacing a file fails with PermissionError while any other
    process (a reader, an antivirus, the search indexer) has it open.
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.01)


def _validate(raw: Any, path: Path) -> dict[str, dict]:
    """Check the store's shape and migrate older formats. Returns ``decisions``."""

    def bad(why: str) -> StoreError:
        return StoreError(
            f"Decision store at {path} is not a valid mcp-decision-lab store: {why}. "
            "Fix or delete the file (sessions in it would otherwise be lost)."
        )

    if not isinstance(raw, dict):
        raise bad(f"top level is a JSON {type(raw).__name__}, expected an object")
    version = raw.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise bad(f"'version' is {version!r}, expected an integer")
    if version > STORE_VERSION:
        raise StoreError(
            f"Decision store at {path} has format version {version}, but this "
            f"mcp-decision-lab only understands up to {STORE_VERSION}. Upgrade "
            "mcp-decision-lab (a newer version wrote this file)."
        )
    decisions = raw.get("decisions", {})
    if not isinstance(decisions, dict):
        raise bad(f"'decisions' is a JSON {type(decisions).__name__}, expected an object")
    for did, dec in decisions.items():
        if not isinstance(dec, dict):
            raise bad(f"decision {did!r} is a JSON {type(dec).__name__}, expected an object")
        missing = [k for k in _REQUIRED_DECISION_KEYS if k not in dec]
        if missing:
            raise bad(f"decision {did!r} is missing {', '.join(missing)}")
        if not isinstance(dec["options"], list) or not isinstance(dec["criteria"], list):
            raise bad(f"decision {did!r} has malformed options/criteria")
        if not isinstance(dec["scores"], dict):
            raise bad(f"decision {did!r} has malformed scores")
        for crit in dec["criteria"]:
            if not isinstance(crit, dict) or not {"name", "weight"} <= set(crit):
                raise bad(f"decision {did!r} has a malformed criterion {crit!r}")
    # v1 -> v2: stamp a per-record schema version and an updated_at.
    for dec in decisions.values():
        dec.setdefault("schema_version", DECISION_SCHEMA_VERSION)
        dec.setdefault("updated_at", dec.get("created_at", ""))
        dec.setdefault("analyzed", False)
        for crit in dec["criteria"]:
            crit.setdefault("higher_is_better", True)
    return decisions


class JsonStore:
    """``decisions.json`` with cross-process locking and change detection."""

    def __init__(self, path: Path, lock_timeout: float = 10.0, stale_after: float = 30.0):
        self.path = Path(path)
        self.lock = FileLock(
            self.path.with_name(self.path.name + ".lock"),
            timeout=lock_timeout,
            stale_after=stale_after,
        )
        self._decisions: dict[str, dict] = {}
        self._signature: tuple[int, int, int] | None = None
        self._mutex = threading.RLock()  # threads of one process
        self._locked = False

    # -- low level ---------------------------------------------------------

    def _stat_signature(self) -> tuple[int, int, int] | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _read_disk(self) -> None:
        sig = self._stat_signature()
        if sig is None:
            self._decisions, self._signature = {}, None
            return
        text = None
        for i in range(50):
            try:
                text = self.path.read_text(encoding="utf-8")
                break
            except PermissionError:  # pragma: no cover - Windows sharing race
                if i == 49:
                    raise
                time.sleep(0.01)
            except FileNotFoundError:
                self._decisions, self._signature = {}, None
                return
        try:
            raw = json.loads(text or "")
        except json.JSONDecodeError as exc:
            raise StoreError(
                f"Could not read decision store at {self.path} ({exc}). "
                "Fix or delete the file."
            ) from exc
        self._decisions = _validate(raw, self.path)
        self._signature = self._stat_signature()

    def _write_disk(self) -> None:
        payload = {"version": STORE_VERSION, "decisions": self._decisions}
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            _replace_with_retry(tmp, self.path)
        finally:
            if tmp.exists():  # pragma: no cover - only after a failed replace
                tmp.unlink()
        self._signature = self._stat_signature()

    # -- public ------------------------------------------------------------

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._mutex:
            if self._locked:  # nested use from the same thread
                yield
                return
            self.lock.acquire()
            self._locked = True
            try:
                yield
            finally:
                self._locked = False
                self.lock.release()

    def snapshot(self) -> dict[str, dict]:
        """Current decisions, refreshed from disk if another process changed it."""
        with self._mutex:
            if self._locked or self._stat_signature() == self._signature:
                return self._decisions
            with self._exclusive():
                self._read_disk()
            return self._decisions

    def mutate(self, fn: Callable[[dict[str, dict]], T]) -> T:
        """Run ``fn(decisions)`` on a fresh copy under the lock, then persist it.

        If ``fn`` raises, nothing is written and the in-memory copy is reloaded
        from disk on the next access.
        """
        with self._exclusive():
            self._read_disk()
            try:
                result = fn(self._decisions)
            except BaseException:
                # fn may have half-mutated the cache: force a reload next time.
                self._signature = _DIRTY
                raise
            self._write_disk()
            return result
