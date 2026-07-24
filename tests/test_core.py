"""Tests for core.py — run without the mcp package installed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import DecisionLab  # noqa: E402


@pytest.fixture()
def lab(tmp_path):
    return DecisionLab(data_dir=tmp_path)


def make_ab(lab, w1=0.6, w2=0.4):
    """Hand-computed 2x2 case: A wins 6.0 vs 4.0 with weights 0.6/0.4."""
    res = lab.start_decision(
        "A or B?",
        ["A", "B"],
        [{"name": "c1", "weight": w1}, {"name": "c2", "weight": w2}],
    )
    did = res["decision_id"]
    lab.score_option(did, "A", "c1", 10, "A is perfect on c1")
    lab.score_option(did, "A", "c2", 0, "A is terrible on c2")
    lab.score_option(did, "B", "c1", 0, "B is terrible on c1")
    lab.score_option(did, "B", "c2", 10, "B is perfect on c2")
    return did


# ----------------------------------------------------------------------
# Weight normalization
# ----------------------------------------------------------------------


def test_weights_normalize_to_one(lab):
    res = lab.start_decision(
        "q?",
        ["A", "B"],
        [
            {"name": "x", "weight": 2},
            {"name": "y", "weight": 1},
            {"name": "z", "weight": 1},
        ],
    )
    weights = {c["name"]: c["weight"] for c in res["criteria"]}
    assert weights == {"x": 0.5, "y": 0.25, "z": 0.25}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_already_normalized_weights_kept(lab):
    res = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 0.7}, {"name": "y", "weight": 0.3}]
    )
    weights = {c["name"]: c["weight"] for c in res["criteria"]}
    assert weights["x"] == pytest.approx(0.7)
    assert weights["y"] == pytest.approx(0.3)


# ----------------------------------------------------------------------
# Validations
# ----------------------------------------------------------------------


def test_duplicate_option_raises(lab):
    with pytest.raises(ValueError, match="Duplicate option"):
        lab.start_decision(
            "q?",
            ["A", "a "],
            [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}],
        )


def test_too_few_options_raises(lab):
    with pytest.raises(ValueError, match="At least 2 options"):
        lab.start_decision(
            "q?", ["A"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
        )


def test_too_few_criteria_raises(lab):
    with pytest.raises(ValueError, match="At least 2 criteria"):
        lab.start_decision("q?", ["A", "B"], [{"name": "x", "weight": 1}])


def test_nonpositive_weight_raises(lab):
    with pytest.raises(ValueError, match="weight"):
        lab.start_decision(
            "q?", ["A", "B"], [{"name": "x", "weight": 0}, {"name": "y", "weight": 1}]
        )


def test_duplicate_criterion_raises(lab):
    with pytest.raises(ValueError, match="Duplicate criterion"):
        lab.start_decision(
            "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "X", "weight": 1}]
        )


def test_score_out_of_range_raises(lab):
    did = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )["decision_id"]
    with pytest.raises(ValueError, match="between 0 and 10"):
        lab.score_option(did, "A", "x", 10.5, "way too high a score")
    with pytest.raises(ValueError, match="between 0 and 10"):
        lab.score_option(did, "A", "x", -1, "way too low a score")


def test_short_rationale_raises(lab):
    did = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )["decision_id"]
    with pytest.raises(ValueError, match="rationale"):
        lab.score_option(did, "A", "x", 5, "short")


def test_unknown_option_and_decision_raise(lab):
    did = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )["decision_id"]
    with pytest.raises(ValueError, match="Unknown option"):
        lab.score_option(did, "C", "x", 5, "no such option here")
    with pytest.raises(ValueError, match="not found"):
        lab.get_matrix("dec-zzzz")


def test_analyze_incomplete_matrix_raises(lab):
    did = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )["decision_id"]
    lab.score_option(did, "A", "x", 5, "only one cell scored")
    with pytest.raises(ValueError, match="incomplete"):
        lab.analyze(did)


# ----------------------------------------------------------------------
# Sensitivity math (hand-computed)
# ----------------------------------------------------------------------


def test_sensitivity_flip_hand_computed(lab):
    """A=(10,0), B=(0,10), weights (0.6, 0.4).

    Totals: A=6.0, B=4.0 → A wins by 2.0.
    gap(w') for c1: 20*w' - 10 → flip at exactly w'=0.5 (decrease).
    gap(w') for c2: 10 - 20*w' → flip at exactly w'=0.5 (increase).
    Both flips are within ±50% of the current weights → fragile.
    """
    did = make_ab(lab)
    res = lab.analyze(did)

    assert res["winner"] == "A"
    assert res["runner_up"] == "B"
    assert res["margin"] == pytest.approx(2.0)
    assert [r["total"] for r in res["ranking"]] == pytest.approx([6.0, 4.0])

    sens = {s["criterion"]: s for s in res["sensitivity"]}
    c1, c2 = sens["c1"], sens["c2"]

    assert c1["decisive"] is True
    assert c1["flip"]["flip_weight"] == pytest.approx(0.5)
    assert c1["flip"]["direction"] == "decrease"
    assert c1["flip"]["new_winner"] == "B"

    assert c2["decisive"] is True
    assert c2["flip"]["flip_weight"] == pytest.approx(0.5)
    assert c2["flip"]["direction"] == "increase"
    assert c2["flip"]["new_winner"] == "B"

    assert res["robustness"] == "fragile"
    assert set(res["decisive_criteria"]) == {"c1", "c2"}


def test_flip_weight_actually_flips(lab):
    """Re-running the decision with weights just past the flip point flips the winner."""
    did = make_ab(lab, w1=0.45, w2=0.55)  # past c1's flip point of 0.5
    res = lab.analyze(did)
    assert res["winner"] == "B"


def test_dominant_winner_is_robust(lab):
    """A beats B on every criterion → no weight can ever flip the winner."""
    res = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )
    did = res["decision_id"]
    lab.score_option(did, "A", "x", 9, "clearly ahead on x")
    lab.score_option(did, "A", "y", 9, "clearly ahead on y")
    lab.score_option(did, "B", "x", 1, "clearly behind on x")
    lab.score_option(did, "B", "y", 1, "clearly behind on y")
    out = lab.analyze(did)
    assert out["winner"] == "A"
    assert out["robustness"] == "robust"
    assert out["decisive_criteria"] == []
    assert all(s["flip"] is None for s in out["sensitivity"])


# ----------------------------------------------------------------------
# higher_is_better = False
# ----------------------------------------------------------------------


def test_higher_is_better_false_inverts_effective_score(lab):
    res = lab.start_decision(
        "q?",
        ["A", "B"],
        [
            {"name": "quality", "weight": 1},
            {"name": "cost", "weight": 1, "higher_is_better": False},
        ],
    )
    did = res["decision_id"]
    lab.score_option(did, "A", "quality", 8, "great quality overall")
    lab.score_option(did, "A", "cost", 2, "cheap to run monthly")
    lab.score_option(did, "B", "quality", 2, "poor quality overall")
    lab.score_option(did, "B", "cost", 8, "expensive to run monthly")

    m = lab.get_matrix(did)
    assert m["matrix"]["A"]["cost"]["score"] == 2
    assert m["matrix"]["A"]["cost"]["effective_score"] == 8  # 10 - 2
    assert m["matrix"]["B"]["cost"]["effective_score"] == 2  # 10 - 8
    assert m["totals"]["A"] == pytest.approx(8.0)  # 0.5*8 + 0.5*8
    assert m["totals"]["B"] == pytest.approx(2.0)  # 0.5*2 + 0.5*2

    out = lab.analyze(did)
    assert out["winner"] == "A"
    # 'cost' scored low-raw/high-effective is A's co-best criterion
    assert out["strengths_weaknesses"]["A"]["best_criterion"]["effective_score"] == 8


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_persistence_write_and_reload(tmp_path):
    lab1 = DecisionLab(data_dir=tmp_path)
    did = make_ab(lab1)
    lab1.analyze(did)

    lab2 = DecisionLab(data_dir=tmp_path)  # fresh instance, same dir
    listing = lab2.list_decisions()
    assert listing["count"] == 1
    assert listing["decisions"][0]["decision_id"] == did
    assert listing["decisions"][0]["status"] == "analyzed"

    m = lab2.get_matrix(did)
    assert m["question"] == "A or B?"
    assert m["matrix"]["A"]["c1"]["score"] == 10
    assert m["matrix"]["A"]["c1"]["rationale"] == "A is perfect on c1"
    weights = {c["name"]: c["weight"] for c in m["criteria"]}
    assert weights["c1"] == pytest.approx(0.6)

    # analysis is reproducible from the reloaded state
    assert lab2.analyze(did)["winner"] == "A"


def test_rescore_overwrites_and_marks_analysis_stale(lab):
    did = make_ab(lab)
    lab.analyze(did)
    assert lab.list_decisions()["decisions"][0]["status"] == "analyzed"

    res = lab.score_option(did, "A", "c1", 1, "revised: A is actually weak on c1")
    assert res["overwrote_previous"] is True
    assert res["cells_scored"] == 4  # still complete
    assert lab.list_decisions()["decisions"][0]["status"] == "complete"  # stale reset


# ----------------------------------------------------------------------
# Status lifecycle
# ----------------------------------------------------------------------


def test_status_lifecycle(lab):
    did = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )["decision_id"]
    assert lab.list_decisions()["decisions"][0]["status"] == "pending"

    lab.score_option(did, "A", "x", 5, "middle of the road")
    lab.score_option(did, "A", "y", 5, "middle of the road")
    lab.score_option(did, "B", "x", 4, "slightly behind A")
    assert lab.list_decisions()["decisions"][0]["status"] == "pending"

    last = lab.score_option(did, "B", "y", 4, "slightly behind A")
    assert last["missing_cells"] == []
    assert lab.list_decisions()["decisions"][0]["status"] == "complete"

    lab.analyze(did)
    assert lab.list_decisions()["decisions"][0]["status"] == "analyzed"
