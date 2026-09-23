"""Persistence: several processes sharing one store, corrupt files, migrations, locks."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcp_decision_lab.core import DecisionLab
from mcp_decision_lab.store import STORE_VERSION, FileLock, StoreError

CRITERIA = [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]


def test_two_instances_do_not_erase_each_other(tmp_path):
    """v0.1: each instance rewrote the file from its own memory; the last writer won."""
    p1, p2 = DecisionLab(tmp_path), DecisionLab(tmp_path)
    d1 = p1.start_decision("from client 1", ["A", "B"], CRITERIA)["decision_id"]
    d2 = p2.start_decision("from client 2", ["A", "B"], CRITERIA)["decision_id"]
    p1.score_option(d1, "A", "x", 7, "client 1 scores its own")
    p2.score_option(d1, "A", "y", 3, "client 2 scores client 1's")
    p1.score_option(d2, "B", "x", 4, "client 1 scores client 2's")

    fresh = DecisionLab(tmp_path)
    questions = [d["question"] for d in fresh.list_decisions()["decisions"]]
    assert questions == ["from client 1", "from client 2"]  # creation order
    m1 = fresh.get_matrix(d1)["matrix"]["A"]
    assert m1["x"]["score"] == 7 and m1["y"]["score"] == 3
    assert fresh.get_matrix(d2)["matrix"]["B"]["x"]["score"] == 4


def test_reads_see_other_processes_changes(tmp_path):
    p1, p2 = DecisionLab(tmp_path), DecisionLab(tmp_path)
    assert p2.list_decisions()["count"] == 0
    did = p1.start_decision("created elsewhere", ["A", "B"], CRITERIA)["decision_id"]
    assert p2.list_decisions()["decisions"][0]["decision_id"] == did
    p1.delete_decision(did)
    with pytest.raises(ValueError, match="not found"):
        p2.score_option(did, "A", "x", 5, "scoring a deleted one")


def test_multiprocess_stress_keeps_every_write(tmp_path):
    """4 processes x 10 decisions x (1 create + 4 scores) = 200 concurrent writes."""
    worker = Path(__file__).with_name("_stress_worker.py")
    procs = [
        subprocess.Popen([sys.executable, str(worker), str(tmp_path), f"p{i}", "10"])
        for i in range(4)
    ]
    for p in procs:
        assert p.wait(timeout=180) == 0

    lab = DecisionLab(tmp_path)
    listing = lab.list_decisions()
    assert listing["count"] == 40
    assert len({d["decision_id"] for d in listing["decisions"]}) == 40
    assert all(d["cells_scored"] == 4 for d in listing["decisions"])
    for i in range(4):
        assert sum(d["question"].startswith(f"p{i} ") for d in listing["decisions"]) == 10
    # No lock or temp files left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["decisions.json"]


@pytest.mark.parametrize(
    "content, match",
    [
        ("[]", "top level is a JSON list"),
        ('{"decisions": []}', "'decisions' is a JSON list"),
        ('{"decisions": {"dec-1": 5}}', "decision 'dec-1' is a JSON int"),
        ('{"decisions": {"dec-1": {"id": "dec-1"}}}', "missing question"),
        ('{"version": "two", "decisions": {}}', "'version'"),
        ("{not json", "Could not read decision store"),
    ],
)
def test_malformed_store_gives_a_clear_error_and_is_not_overwritten(tmp_path, content, match):
    store = tmp_path / "decisions.json"
    store.write_text(content, encoding="utf-8")
    with pytest.raises(StoreError, match=match) as info:
        DecisionLab(tmp_path)
    assert "Fix or delete the file" in str(info.value)
    assert isinstance(info.value, ValueError)
    assert store.read_text(encoding="utf-8") == content  # user data untouched


def test_newer_store_version_is_refused(tmp_path):
    (tmp_path / "decisions.json").write_text(
        json.dumps({"version": STORE_VERSION + 1, "decisions": {}}), encoding="utf-8"
    )
    with pytest.raises(StoreError, match="Upgrade mcp-decision-lab"):
        DecisionLab(tmp_path)


def test_v1_store_is_migrated(tmp_path):
    """A file exactly as mcp-decision-lab 0.1.0 wrote it."""
    v1 = {
        "version": 1,
        "decisions": {
            "dec-0001": {
                "id": "dec-0001",
                "question": "Old decision?",
                "options": ["A", "B"],
                "criteria": [
                    {"name": "x", "weight": 0.5, "higher_is_better": True},
                    {"name": "y", "weight": 0.5, "higher_is_better": False},
                ],
                "scores": {"A": {"x": {"score": 8.0, "rationale": "A is strong on x"}}},
                "analyzed": False,
                "created_at": "2026-07-26T10:00:00+00:00",
            }
        },
    }
    (tmp_path / "decisions.json").write_text(json.dumps(v1), encoding="utf-8")
    lab = DecisionLab(tmp_path)
    assert lab.get_matrix("dec-0001")["matrix"]["A"]["x"]["score"] == 8.0
    lab.score_option("dec-0001", "B", "x", 2, "B is weak on x here")
    raw = json.loads((tmp_path / "decisions.json").read_text(encoding="utf-8"))
    assert raw["version"] == STORE_VERSION
    dec = raw["decisions"]["dec-0001"]
    assert dec["schema_version"] == 1
    assert dec["created_at"] == "2026-07-26T10:00:00+00:00"
    assert dec["updated_at"] >= dec["created_at"]


def test_stale_lock_from_a_crashed_process_is_recovered(tmp_path):
    lab = DecisionLab(tmp_path, lock_timeout=5)
    lock = tmp_path / "decisions.json.lock"
    lock.write_text(f"999999:deadbeef\n{time.time() - 3600:.6f}\n", encoding="utf-8")
    did = lab.start_decision("after a crash", ["A", "B"], CRITERIA)["decision_id"]
    assert did in {d["decision_id"] for d in lab.list_decisions()["decisions"]}
    assert not lock.exists()


def test_a_live_lock_times_out_with_a_clear_error(tmp_path):
    lab = DecisionLab(tmp_path, lock_timeout=0.3)
    holder = FileLock(tmp_path / "decisions.json.lock")
    holder.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(StoreError, match="Timed out"):
            lab.start_decision("blocked", ["A", "B"], CRITERIA)
        assert time.monotonic() - started < 5
    finally:
        holder.release()
    assert lab.start_decision("unblocked", ["A", "B"], CRITERIA)["decision_id"]


def test_failed_mutation_writes_nothing(tmp_path):
    lab = DecisionLab(tmp_path)
    did = lab.start_decision("q?", ["A", "B"], CRITERIA)["decision_id"]
    before = (tmp_path / "decisions.json").read_bytes()
    with pytest.raises(ValueError, match="Unknown option"):
        lab.score_option(did, "C", "x", 5, "no such option exists")
    assert (tmp_path / "decisions.json").read_bytes() == before
    assert lab.get_matrix(did)["matrix"]["A"]["x"] is None


def test_delete_decision_removes_it_from_disk(tmp_path):
    lab = DecisionLab(tmp_path)
    keep = lab.start_decision("keep me", ["A", "B"], CRITERIA)["decision_id"]
    drop = lab.start_decision("drop me", ["A", "B"], CRITERIA)["decision_id"]
    out = lab.delete_decision(drop)
    assert out == {"deleted": drop, "question": "drop me", "remaining_decisions": 1}
    raw = json.loads((tmp_path / "decisions.json").read_text(encoding="utf-8"))
    assert list(raw["decisions"]) == [keep]
    with pytest.raises(ValueError, match="not found"):
        lab.delete_decision(drop)


def test_default_data_dir_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_LAB_DIR", str(tmp_path / "from-env"))
    lab = DecisionLab()
    assert lab.store_path == tmp_path / "from-env" / "decisions.json"
    assert os.path.isdir(tmp_path / "from-env")
