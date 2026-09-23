"""Core logic for mcp-decision-lab.

Weighted decision matrices with criteria scoring, exact (closed-form) weight
and score sensitivity analysis, seeded Monte Carlo robustness and a
defensible recommendation.

Pure Python stdlib — no external dependencies — so it can be imported and
tested without the MCP SDK installed. ``server.py`` is a thin MCP wrapper
around the :class:`DecisionLab` class defined here; the math lives in
:mod:`mcp_decision_lab.analysis` and persistence in
:mod:`mcp_decision_lab.store`.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import analysis as A
from .store import DECISION_SCHEMA_VERSION, JsonStore, StoreError

ENV_DATA_DIR = "DECISION_LAB_DIR"
DEFAULT_DIR_NAME = ".mcp-decision-lab"
STORE_FILENAME = "decisions.json"

SCORE_MIN = A.SCORE_MIN
SCORE_MAX = A.SCORE_MAX
MIN_RATIONALE_CHARS = 10
ROBUSTNESS_BAND = A.WEIGHT_BAND  # kept for backwards compatibility

_ALLOWED_CRITERION_KEYS = {"name", "weight", "higher_is_better"}

__all__ = ["DecisionLab", "StoreError", "default_data_dir"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _r4(x: float) -> float:
    return A.r4(x)


def _effective(score: float, higher_is_better: bool) -> float:
    return A.effective(score, higher_is_better)


def default_data_dir() -> Path:
    """``$DECISION_LAB_DIR`` if set, else ``~/.mcp-decision-lab``."""
    env = os.environ.get(ENV_DATA_DIR)
    return Path(env) if env else Path.home() / DEFAULT_DIR_NAME


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fingerprint(dec: dict) -> str:
    """Identity of everything an analysis depends on."""
    return json.dumps(
        [dec["options"], dec["criteria"], dec["scores"]], sort_keys=True, ensure_ascii=False
    )


class DecisionLab:
    """Stateful decision-matrix engine with process-safe JSON persistence.

    Parameters
    ----------
    data_dir:
        Directory where ``decisions.json`` lives. Defaults to the
        ``DECISION_LAB_DIR`` environment variable, or ``~/.mcp-decision-lab``.
        Several processes (e.g. two MCP clients) can share it safely.
    lock_timeout:
        Seconds to wait for another process holding the store lock.
    """

    def __init__(self, data_dir: str | Path | None = None, *, lock_timeout: float = 10.0) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.data_dir / STORE_FILENAME
        self._store = JsonStore(self.store_path, lock_timeout=lock_timeout)
        self._store.snapshot()  # fail fast on a corrupt or foreign store file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _decisions(self) -> dict[str, dict]:
        return self._store.snapshot()

    @staticmethod
    def _find(decisions: dict[str, dict], decision_id: str) -> dict:
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError("decision_id must be a non-empty string (e.g. 'dec-a3f2').")
        dec = decisions.get(decision_id.strip())
        if dec is None:
            known = ", ".join(sorted(decisions)) or "none yet"
            raise ValueError(
                f"Decision '{decision_id}' not found (known ids: {known}). "
                "Call list_decisions to see sessions, or start_decision to create one."
            )
        return dec

    def _get(self, decision_id: str) -> dict:
        return self._find(self._decisions, decision_id)

    @staticmethod
    def _new_id(decisions: dict[str, dict]) -> str:
        while True:
            did = "dec-" + uuid.uuid4().hex[:4]
            if did not in decisions:
                return did

    @staticmethod
    def _resolve(name: Any, valid: list[str], kind: str) -> str:
        """Resolve ``name`` against ``valid`` (exact, then case-insensitive)."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{kind} must be a non-empty string.")
        name = name.strip()
        if name in valid:
            return name
        folded = {v.casefold(): v for v in valid}
        hit = folded.get(name.casefold())
        if hit is not None:
            return hit
        raise ValueError(f"Unknown {kind} '{name}'. Valid {kind}s: {', '.join(valid)}.")

    @staticmethod
    def _missing_cells(dec: dict, scores: dict | None = None) -> list[dict]:
        scores = dec["scores"] if scores is None else scores
        return [
            {"option": o, "criterion": c["name"]}
            for o in dec["options"]
            for c in dec["criteria"]
            if c["name"] not in scores.get(o, {})
        ]

    @staticmethod
    def _status(dec: dict) -> str:
        if DecisionLab._missing_cells(dec):
            return "pending"
        return "analyzed" if dec.get("analyzed") else "complete"

    @staticmethod
    def _totals(dec: dict) -> dict[str, float]:
        """Weighted totals over *scored* cells (full totals once complete)."""
        totals: dict[str, float] = {}
        for o in dec["options"]:
            t = 0.0
            for c in dec["criteria"]:
                cell = dec["scores"].get(o, {}).get(c["name"])
                if cell is not None:
                    t += c["weight"] * _effective(cell["score"], c["higher_is_better"])
            totals[o] = t
        return totals

    @staticmethod
    def _criteria_view(dec: dict, weights: list[float] | None = None) -> list[dict]:
        return [
            {
                "name": c["name"],
                "weight": _r4(c["weight"] if weights is None else weights[j]),
                "higher_is_better": c["higher_is_better"],
            }
            for j, c in enumerate(dec["criteria"])
        ]

    @staticmethod
    def _problem(
        dec: dict, scores: dict | None = None, weights: list[float] | None = None
    ) -> A.Problem:
        scores = dec["scores"] if scores is None else scores
        crits = dec["criteria"]
        return A.Problem(
            options=list(dec["options"]),
            criteria=[c["name"] for c in crits],
            weights=[c["weight"] for c in crits] if weights is None else list(weights),
            higher_is_better=[c["higher_is_better"] for c in crits],
            raw=[[scores[o][c["name"]]["score"] for c in crits] for o in dec["options"]],
            rationales=[
                [scores[o][c["name"]].get("rationale", "") for c in crits] for o in dec["options"]
            ],
        )

    def _require_complete(self, dec: dict, action: str) -> None:
        missing = self._missing_cells(dec)
        if missing:
            preview = ", ".join(f"{m['option']}/{m['criterion']}" for m in missing[:6])
            raise ValueError(
                f"Matrix incomplete — {len(missing)} cell(s) missing ({preview}"
                f"{', …' if len(missing) > 6 else ''}). Score them with "
                f"score_option before calling {action}."
            )

    @staticmethod
    def _validate_score(score: Any) -> float:
        if not _is_number(score):
            raise ValueError("score must be a number between 0 and 10.")
        score = float(score)
        if not math.isfinite(score) or not (SCORE_MIN <= score <= SCORE_MAX):
            raise ValueError(
                f"score must be between {SCORE_MIN:g} and {SCORE_MAX:g} (got {score:g})."
            )
        return score

    # ------------------------------------------------------------------
    # Public API (mirrored 1:1 by server.py tools)
    # ------------------------------------------------------------------

    def start_decision(self, question: str, options: list[str], criteria: list[dict]) -> dict:
        """Create a decision session with normalized criterion weights."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string describing the decision.")
        question = question.strip()

        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(
                "At least 2 options are required — a decision with fewer than "
                "2 options is not a decision."
            )
        clean_options: list[str] = []
        seen_opts: set[str] = set()
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                raise ValueError("Every option must be a non-empty string.")
            opt = opt.strip()
            key = opt.casefold()
            if key in seen_opts:
                raise ValueError(f"Duplicate option '{opt}' — options must be unique.")
            seen_opts.add(key)
            clean_options.append(opt)

        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError(
                "At least 2 criteria are required — with a single criterion, "
                "just pick the option with the best score."
            )
        clean_criteria: list[dict] = []
        seen_crit: set[str] = set()
        for crit in criteria:
            if not isinstance(crit, dict):
                raise ValueError(
                    'Each criterion must be a dict like {"name": "cost", '
                    '"weight": 0.5, "higher_is_better": false}.'
                )
            unknown = set(crit) - _ALLOWED_CRITERION_KEYS
            if unknown:
                raise ValueError(
                    f"Unknown criterion key(s) {sorted(unknown)} — allowed keys: "
                    f"{sorted(_ALLOWED_CRITERION_KEYS)}."
                )
            name = crit.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each criterion needs a non-empty 'name' string.")
            name = name.strip()
            key = name.casefold()
            if key in seen_crit:
                raise ValueError(f"Duplicate criterion '{name}' — criteria must be unique.")
            seen_crit.add(key)
            weight = crit.get("weight")
            if not _is_number(weight):
                raise ValueError(f"Criterion '{name}': 'weight' must be a number.")
            weight = float(weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"Criterion '{name}': weight must be a finite number > 0 "
                    f"(got {weight}). Weights are normalized to sum to 1.0."
                )
            hib = crit.get("higher_is_better", True)
            if not isinstance(hib, bool):
                raise ValueError(f"Criterion '{name}': 'higher_is_better' must be true or false.")
            clean_criteria.append({"name": name, "weight": weight, "higher_is_better": hib})

        # Normalize via the largest weight first so huge inputs cannot overflow.
        top = max(c["weight"] for c in clean_criteria)
        scaled = [c["weight"] / top for c in clean_criteria]
        total_w = math.fsum(scaled)
        for c, s in zip(clean_criteria, scaled):
            c["weight"] = s / total_w
            if not c["weight"] > 0.0:
                raise ValueError(
                    f"Criterion '{c['name']}': its weight is so small relative to the "
                    "others that it normalizes to 0. Use weights within a sane range "
                    "(e.g. 0.01-100)."
                )

        def create(decisions: dict[str, dict]) -> dict:
            decision_id = self._new_id(decisions)
            now = _now_iso()
            dec = {
                "schema_version": DECISION_SCHEMA_VERSION,
                "id": decision_id,
                "question": question,
                "options": clean_options,
                "criteria": clean_criteria,
                "scores": {},
                "analyzed": False,
                "created_at": now,
                "updated_at": now,
            }
            decisions[decision_id] = dec
            return dec

        dec = self._store.mutate(create)
        missing = self._missing_cells(dec)
        return {
            "decision_id": dec["id"],
            "question": question,
            "options": clean_options,
            "criteria": self._criteria_view(dec),
            "weights_normalized": True,
            "cells_scored": 0,
            "cells_total": len(missing),
            "missing_cells": missing,
            "next_step": (
                f"Score each option against each criterion using score_option "
                f"(score 0-10 with a rationale of at least {MIN_RATIONALE_CHARS} "
                f"characters). {len(missing)} cells to fill, then call analyze."
            ),
        }

    def score_option(
        self,
        decision_id: str,
        option: str,
        criterion: str,
        score: float,
        rationale: str,
    ) -> dict:
        """Record one cell of the matrix: how ``option`` does on ``criterion``."""
        score = self._validate_score(score)
        if not isinstance(rationale, str) or len(rationale.strip()) < MIN_RATIONALE_CHARS:
            raise ValueError(
                f"rationale is required (>= {MIN_RATIONALE_CHARS} characters): "
                "briefly justify the score so the final recommendation is defensible."
            )
        rationale = rationale.strip()

        def record(decisions: dict[str, dict]) -> tuple[dict, str, str, bool]:
            dec = self._find(decisions, decision_id)
            opt = self._resolve(option, dec["options"], "option")
            crit = self._resolve(criterion, [c["name"] for c in dec["criteria"]], "criterion")
            cells = dec["scores"].setdefault(opt, {})
            overwrote = crit in cells
            cells[crit] = {"score": score, "rationale": rationale}
            dec["analyzed"] = False  # scores changed -> any prior analysis is stale
            dec.pop("analysis_summary", None)
            dec["updated_at"] = _now_iso()
            return dec, opt, crit, overwrote

        dec, opt, crit, overwrote = self._store.mutate(record)
        did = dec["id"]
        missing = self._missing_cells(dec)
        total_cells = len(dec["options"]) * len(dec["criteria"])
        if missing:
            nxt = (
                f"{len(missing)} cell(s) remaining — keep calling score_option. "
                f"Next missing: {missing[0]['option']} / {missing[0]['criterion']}."
            )
        else:
            nxt = (
                f"Matrix complete — call analyze('{did}') for the ranking, "
                "sensitivity analysis and recommendation."
            )
        return {
            "decision_id": did,
            "recorded": {"option": opt, "criterion": crit, "score": score, "rationale": rationale},
            "overwrote_previous": overwrote,
            "cells_scored": total_cells - len(missing),
            "cells_total": total_cells,
            "missing_cells": missing,
            "next_step": nxt,
        }

    def get_matrix(self, decision_id: str) -> dict:
        """Full matrix view: raw scores, weighted scores, per-option totals."""
        dec = self._get(decision_id)
        totals = self._totals(dec)
        matrix: dict[str, dict] = {}
        for o in dec["options"]:
            row: dict[str, dict | None] = {}
            for c in dec["criteria"]:
                cell = dec["scores"].get(o, {}).get(c["name"])
                if cell is None:
                    row[c["name"]] = None
                else:
                    eff = _effective(cell["score"], c["higher_is_better"])
                    row[c["name"]] = {
                        "score": cell["score"],
                        "effective_score": _r4(eff),
                        "weighted_score": _r4(c["weight"] * eff),
                        "rationale": cell["rationale"],
                    }
            matrix[o] = row
        missing = self._missing_cells(dec)
        return {
            "decision_id": dec["id"],
            "question": dec["question"],
            "status": self._status(dec),
            "criteria": self._criteria_view(dec),
            "matrix": matrix,
            "totals": {o: _r4(t) for o, t in totals.items()},
            "totals_note": (
                "Totals cover all cells."
                if not missing
                else f"Partial totals — {len(missing)} cell(s) still unscored."
            ),
            "missing_cells": missing,
        }

    def _analysis(
        self,
        dec: dict,
        simulations: int = A.DEFAULT_SIMULATIONS,
        seed: int = A.DEFAULT_SEED,
        score_noise: float = A.DEFAULT_SCORE_NOISE,
        weight_concentration: float = A.DEFAULT_CONCENTRATION,
        top_cells: int | None = 5,
    ) -> dict:
        """Pure analysis of a complete decision record (no state change)."""
        if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 0:
            raise ValueError("simulations must be a non-negative integer (0 disables Monte Carlo).")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer.")
        if not _is_number(score_noise) or not _is_number(weight_concentration):
            raise ValueError("score_noise and weight_concentration must be numbers.")
        result = A.analyze_problem(
            self._problem(dec),
            simulations=simulations,
            seed=seed,
            score_noise=float(score_noise),
            concentration=float(weight_concentration),
            top_cells=top_cells,
        )
        return {"decision_id": dec["id"], "question": dec["question"], **result}

    def analyze(
        self,
        decision_id: str,
        simulations: int = A.DEFAULT_SIMULATIONS,
        seed: int = A.DEFAULT_SEED,
        score_noise: float = A.DEFAULT_SCORE_NOISE,
        weight_concentration: float = A.DEFAULT_CONCENTRATION,
    ) -> dict:
        """Ranking, sensitivity analysis, Monte Carlo robustness, verdict, recommendation.

        ``simulations=0`` skips the Monte Carlo step. The same ``seed`` always
        gives the same Monte Carlo numbers.
        """
        dec = self._get(decision_id)
        self._require_complete(dec, "analyze")
        result = self._analysis(dec, simulations, seed, score_noise, weight_concentration)
        fp = _fingerprint(dec)

        def mark(decisions: dict[str, dict]) -> None:
            fresh = decisions.get(dec["id"])
            # Only mark analyzed if nobody re-scored the matrix meanwhile.
            if fresh is not None and _fingerprint(fresh) == fp:
                fresh["analyzed"] = True
                fresh["analysis_summary"] = {
                    "winner": result["winner"],
                    "tied": result["tied"],
                    "margin": result["margin"],
                    "robustness": result["robustness"],
                    "analyzed_at": _now_iso(),
                }

        self._store.mutate(mark)
        return result

    def _what_if_weights(self, dec: dict, weights: Any) -> list[float]:
        crit_names = [c["name"] for c in dec["criteria"]]
        base_w = [c["weight"] for c in dec["criteria"]]
        if not isinstance(weights, dict) or not weights:
            raise ValueError("weights must be a non-empty object {criterion: weight}.")
        given: dict[int, float] = {}
        for name, val in weights.items():
            j = crit_names.index(self._resolve(name, crit_names, "criterion"))
            if j in given:
                raise ValueError(f"Criterion '{crit_names[j]}' is given twice in weights.")
            if not _is_number(val) or not math.isfinite(float(val)) or float(val) < 0:
                raise ValueError(f"weights['{name}'] must be a finite number >= 0 (got {val!r}).")
            given[j] = float(val)
        total = math.fsum(given.values())
        if len(given) == len(crit_names):
            if not total > 0:
                raise ValueError("At least one criterion needs a weight > 0.")
            return [given[j] / total for j in range(len(crit_names))]
        if any(v > 1 for v in given.values()) or total > 1 + 1e-12:
            raise ValueError(
                "When naming only some criteria, give their new *normalized* weights "
                "(fractions 0-1 whose sum is at most 1); the other criteria share the "
                "rest proportionally. To set relative importances instead, name every "
                "criterion: " + ", ".join(crit_names) + "."
            )
        rest_old = math.fsum(base_w[j] for j in range(len(crit_names)) if j not in given)
        remaining = max(0.0, 1.0 - total)
        new_w = [
            given[j] if j in given else base_w[j] / rest_old * remaining
            for j in range(len(crit_names))
        ]
        if not math.fsum(new_w) > 0:
            raise ValueError("At least one criterion needs a weight > 0.")
        return new_w

    def what_if(
        self,
        decision_id: str,
        weights: dict[str, float] | None = None,
        score_overrides: list[dict] | None = None,
    ) -> dict:
        """Ranking under hypothetical weights and/or scores. Never changes the session.

        ``weights`` maps criterion names to new weights. Naming *every* criterion
        gives new relative importances (normalized like start_decision). Naming
        only *some* sets those criteria to exact normalized weights (fractions
        0-1 summing to at most 1); the unnamed criteria share the remainder in
        proportion to their current weights — the same convention analyze's
        sensitivity uses, so a reported flip weight can be tested directly.
        ``score_overrides`` is a list of ``{"option", "criterion", "score"}``.
        """
        dec = self._get(decision_id)
        crit_names = [c["name"] for c in dec["criteria"]]
        if weights is None and not score_overrides:
            raise ValueError(
                'Pass weights (e.g. {"cost": 0.2}) and/or score_overrides '
                '(e.g. [{"option": "A", "criterion": "cost", "score": 4}]).'
            )
        new_w = (
            [c["weight"] for c in dec["criteria"]]
            if weights is None
            else self._what_if_weights(dec, weights)
        )

        scores = {o: dict(cells) for o, cells in dec["scores"].items()}
        applied = []
        for ov in score_overrides or []:
            if not isinstance(ov, dict):
                raise ValueError(
                    'Each score override must be {"option": ..., "criterion": ..., "score": ...}.'
                )
            extra = set(ov) - {"option", "criterion", "score"}
            if extra:
                raise ValueError(f"Unknown score override key(s) {sorted(extra)}.")
            opt = self._resolve(ov.get("option"), dec["options"], "option")
            crit = self._resolve(ov.get("criterion"), crit_names, "criterion")
            new_score = self._validate_score(ov.get("score"))
            old = scores.get(opt, {}).get(crit)
            scores.setdefault(opt, {})[crit] = {
                "score": new_score,
                "rationale": "(hypothetical override)",
            }
            applied.append(
                {
                    "option": opt,
                    "criterion": crit,
                    "from": None if old is None else old["score"],
                    "to": new_score,
                }
            )

        missing = self._missing_cells(dec, scores)
        if missing:
            preview = ", ".join(f"{m['option']}/{m['criterion']}" for m in missing[:6])
            raise ValueError(
                f"what_if needs a complete matrix — {len(missing)} cell(s) are unscored "
                f"({preview}). Score them, or cover them with score_overrides."
            )

        base = None
        if not self._missing_cells(dec):
            bp = self._problem(dec)
            bt = bp.totals()
            bl = A.leaders(bt)
            base = {
                "winner": dec["options"][bl[0]] if len(bl) == 1 else None,
                "tied": [dec["options"][i] for i in bl] if len(bl) > 1 else [],
                "ranking": A.rank(bp.options, bt),
                "totals": bt,
            }

        hp = self._problem(dec, scores=scores, weights=new_w)
        ht = hp.totals()
        hl = A.leaders(ht)
        h_winner = dec["options"][hl[0]] if len(hl) == 1 else None
        h_tied = [dec["options"][i] for i in hl] if len(hl) > 1 else []
        margin = 0.0 if h_tied else ht[hl[0]] - max(t for i, t in enumerate(ht) if i != hl[0])

        if base is None:
            changed = None
            summary = "The stored matrix is incomplete, so there is no baseline winner to compare."
            total_changes = None
        else:
            changed = (base["winner"], base["tied"]) != (h_winner, h_tied)
            before = base["winner"] or "a tie (" + ", ".join(base["tied"]) + ")"
            after = h_winner or "a tie (" + ", ".join(h_tied) + ")"
            summary = (
                f"Winner changes from {before} to {after}."
                if changed
                else f"Winner unchanged: {after}."
            )
            total_changes = {o: _r4(ht[i] - base["totals"][i]) for i, o in enumerate(dec["options"])}
            base.pop("totals")
        if h_winner:
            summary += f" Hypothetical margin {_r4(margin)}."

        return {
            "decision_id": dec["id"],
            "hypothetical": True,
            "session_modified": False,
            "weights_used": self._criteria_view(dec, new_w),
            "score_overrides_applied": applied,
            "baseline": base,
            "ranking": A.rank(hp.options, ht),
            "winner": h_winner,
            "tied": h_tied,
            "margin": _r4(margin),
            "winner_changed": changed,
            "total_changes": total_changes,
            "summary": summary,
        }

    def export_decision(self, decision_id: str, format: str = "markdown") -> dict:
        """A reviewable decision record (Markdown ADR or JSON). Never changes the session."""
        from .report import to_json, to_markdown

        fmt = format.strip().lower() if isinstance(format, str) else ""
        if fmt in ("md", "markdown"):
            fmt = "markdown"
        elif fmt != "json":
            raise ValueError(f"format must be 'markdown' or 'json' (got {format!r}).")
        dec = self._get(decision_id)
        matrix = self.get_matrix(dec["id"])
        analysis = None if self._missing_cells(dec) else self._analysis(dec, top_cells=None)
        render = to_markdown if fmt == "markdown" else to_json
        return {
            "decision_id": dec["id"],
            "format": fmt,
            "status": self._status(dec),
            "content": render(dec, matrix, analysis),
        }

    def delete_decision(self, decision_id: str) -> dict:
        """Permanently remove a decision session from the store."""

        def drop(decisions: dict[str, dict]) -> tuple[dict, int]:
            dec = self._find(decisions, decision_id)
            del decisions[dec["id"]]
            return dec, len(decisions)

        dec, remaining = self._store.mutate(drop)
        return {
            "deleted": dec["id"],
            "question": dec["question"],
            "remaining_decisions": remaining,
        }

    def list_decisions(self) -> dict:
        """All decision sessions with their status."""
        items = []
        decisions = self._decisions
        # Stable sort: same-second sessions keep the store's insertion (= creation) order.
        for dec in sorted(decisions.values(), key=lambda d: d.get("created_at", "")):
            total_cells = len(dec["options"]) * len(dec["criteria"])
            missing = len(self._missing_cells(dec))
            item = {
                "decision_id": dec["id"],
                "question": dec["question"],
                "status": self._status(dec),
                "options": dec["options"],
                "criteria": [c["name"] for c in dec["criteria"]],
                "cells_scored": total_cells - missing,
                "cells_total": total_cells,
                "created_at": dec.get("created_at", ""),
                "updated_at": dec.get("updated_at", ""),
            }
            summary = dec.get("analysis_summary")
            if dec.get("analyzed") and summary:
                item["result"] = summary
            items.append(item)
        return {
            "count": len(items),
            "decisions": items,
            "statuses": {
                "pending": "matrix has unscored cells",
                "complete": "fully scored, not yet analyzed",
                "analyzed": "analyze() has been run on the current scores",
            },
        }
