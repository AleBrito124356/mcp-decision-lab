"""Analysis engine: ties, extreme weights, score sensitivity, Monte Carlo, what_if.

Every expected number below is computed by hand in the docstring or comment.
"""

import pytest

from conftest import make_matrix
from mcp_decision_lab import demo

# ----------------------------------------------------------------------
# Ties
# ----------------------------------------------------------------------


def test_exact_tie_is_not_reported_as_a_robust_win(lab):
    """A=(5,5), B=(5,5): identical options. v0.1 said 'Choose A ... robust'."""
    did = make_matrix(lab, {"A": (5, 5), "B": (5, 5)})
    out = lab.analyze(did, score_noise=0)
    assert out["robustness"] == "tie"
    assert out["winner"] is None and out["runner_up"] is None
    assert out["tied"] == ["A", "B"]
    assert out["margin"] == 0.0
    assert "Choose" not in out["recommendation"]
    assert "identical effective scores" in out["recommendation"]
    assert out["decisive_criteria"] == []
    assert [r["rank"] for r in out["ranking"]] == [1, 1]
    # identical options and no score noise: every sample is an exact tie,
    # so each gets exactly half of the Monte Carlo credit
    mc = out["monte_carlo"]["options"]
    assert mc["A"]["win_probability"] == pytest.approx(0.5)
    assert mc["B"]["win_probability"] == pytest.approx(0.5)


def test_tradeoff_tie_names_the_tie_breakers(lab):
    """A=(8,2), B=(2,8), equal weights: both total 5.0.

    Raising c0's weight favors A (8 > 2); lowering it favors B. The reverse for c1.
    v0.1 said 'Choose A' and 'if c0 weight drops below 0.5 (now 0.5), B wins'.
    """
    did = make_matrix(lab, {"A": (8, 2), "B": (2, 8)})
    out = lab.analyze(did)
    assert out["robustness"] == "tie" and out["winner"] is None
    assert "Choose" not in out["recommendation"]
    sens = {e["criterion"]: e for e in out["sensitivity"]}
    assert sens["c0"]["tie_break"]["raise_weight_favors"] == ["A"]
    assert sens["c0"]["tie_break"]["lower_weight_favors"] == ["B"]
    assert sens["c1"]["tie_break"]["raise_weight_favors"] == ["B"]
    assert sens["c1"]["tie_break"]["lower_weight_favors"] == ["A"]
    assert set(out["decisive_criteria"]) == {"c0", "c1"}
    assert "giving 'c0' more weight favors A" in out["recommendation"]

    # And the tie-breakers are real: c0 at 0.6 -> A = 5.6 vs B = 4.4.
    assert lab.what_if(did, weights={"c0": 0.6})["winner"] == "A"
    assert lab.what_if(did, weights={"c0": 0.4})["winner"] == "B"


def test_tie_for_second_place_keeps_a_clear_winner(lab):
    did = make_matrix(lab, {"A": (9, 9), "B": (5, 5), "C": (5, 5)})
    out = lab.analyze(did, simulations=0)
    assert out["winner"] == "A" and out["tied"] == []
    assert [r["rank"] for r in out["ranking"]] == [1, 2, 2]


# ----------------------------------------------------------------------
# Extreme weights
# ----------------------------------------------------------------------


def test_weight_that_normalizes_to_one_does_not_crash(lab):
    """Weights [1e20, 1] -> w0 = 1.0 exactly in floating point (1 - w0 == 0).

    v0.1 divided by (1 - w) and raised ZeroDivisionError. A=(8,4), B=(2,8):
    c0: d = 6, r = 4 - 8 = -4 -> gap(w') = 10 w' - 4 -> flip at w' = 0.4 (decrease).
    c1: d = -4, r = 6       -> gap(w') = 6 - 10 w'  -> flip at w' = 0.6 (increase).
    Both are far outside ±50% of the current weights, so the result is robust.
    """
    did = make_matrix(lab, {"A": (8, 4), "B": (2, 8)}, weights=(1e20, 1))
    out = lab.analyze(did)
    assert out["winner"] == "A"
    sens = {e["criterion"]: e for e in out["sensitivity"]}
    assert sens["c0"]["weight"] == 1.0
    assert sens["c0"]["flip"]["flip_weight"] == pytest.approx(0.4)
    assert sens["c0"]["flip"]["direction"] == "decrease"
    assert sens["c1"]["flip"]["flip_weight"] == pytest.approx(0.6)
    assert sens["c1"]["flip"]["direction"] == "increase"
    assert out["robustness"] == "robust"


def test_extreme_weights_dominant_option_has_no_flip(lab):
    did = make_matrix(lab, {"A": (9, 9), "B": (1, 1)}, weights=(1e20, 1))
    out = lab.analyze(did)
    assert out["winner"] == "A"
    assert all(e["flip"] is None for e in out["sensitivity"])
    assert out["robustness"] == "robust"


def test_huge_weights_do_not_overflow_normalization(lab):
    """v0.1 summed first: 1e308 + 1e308 = inf, so every weight became 0."""
    res = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1e308}, {"name": "y", "weight": 1e308}]
    )
    assert [c["weight"] for c in res["criteria"]] == [0.5, 0.5]


def test_weight_ratio_beyond_float_range_is_rejected(lab):
    with pytest.raises(ValueError, match="normalizes to 0"):
        lab.start_decision(
            "q?", ["A", "B"], [{"name": "x", "weight": 1e308}, {"name": "y", "weight": 1e-308}]
        )


# ----------------------------------------------------------------------
# Strengths / weaknesses
# ----------------------------------------------------------------------


def test_uniform_option_is_not_named_best_and_worst(lab):
    did = make_matrix(lab, {"A": (7, 7), "B": (2, 9)})
    sw = lab.analyze(did, simulations=0)["strengths_weaknesses"]
    assert sw["A"] == {
        "uniform": True,
        "effective_score": 7.0,
        "best_criterion": None,
        "worst_criterion": None,
    }
    assert sw["B"]["best_criterion"]["name"] == "c1"
    assert sw["B"]["worst_criterion"]["name"] == "c0"


# ----------------------------------------------------------------------
# Score sensitivity
# ----------------------------------------------------------------------


def test_score_flip_thresholds_hand_computed(lab):
    """A=(10,0), B=(0,10), weights 0.6/0.4 -> A 6.0, B 4.0, margin 2.0.

    A's c0 must fall below 10 - 2/0.6 = 6.6667; B's c0 must rise above
    0 + 2/0.6 = 3.3333. The c1 cells would need 10 - 5 < 0 or 10 + 5 > 10,
    which the 0-10 scale cannot reach.
    """
    did = make_matrix(lab, {"A": (10, 0), "B": (0, 10)}, weights=(0.6, 0.4))
    ss = lab.analyze(did, simulations=0)["score_sensitivity"]
    assert ss["flippable_cells"] == 2 and ss["cells_total"] == 4
    cells = {(c["option"], c["criterion"]): c for c in ss["most_fragile"]}
    a = cells[("A", "c0")]
    assert a["flip_score"] == pytest.approx(6.6667) and a["direction"] == "below"
    assert a["change_needed"] == pytest.approx(3.3333) and a["new_winner"] == "B"
    b = cells[("B", "c0")]
    assert b["flip_score"] == pytest.approx(3.3333) and b["direction"] == "above"
    assert b["new_winner"] == "B"
    assert a["rationale"] == "A scores 10 on c0"


def _rescore_and_winner(lab, did, cell, score):
    lab.score_option(did, cell["option"], cell["criterion"], score, "threshold probe rescoring")
    return lab.analyze(did, simulations=0)


@pytest.mark.parametrize("which", range(5))
def test_reported_score_thresholds_really_flip(lab, which):
    """Crossing each reported threshold by 0.01 flips the winner; stopping 0.01 short does not.

    README example: 5 of 9 cells can flip it (Postgres x3, DynamoDB cost, MongoDB cost);
    the others would need a score beyond the 0-10 scale.
    """
    did = demo.build(lab)
    base = lab.analyze(did, simulations=0)
    assert base["score_sensitivity"]["flippable_cells"] == 5
    cell = base["score_sensitivity"]["most_fragile"][which]
    step = 0.01 if cell["direction"] == "above" else -0.01
    short = _rescore_and_winner(lab, did, cell, cell["flip_score"] - step)
    assert short["winner"] == base["winner"]
    past = _rescore_and_winner(lab, did, cell, cell["flip_score"] + step)
    assert past["winner"] == cell["new_winner"] != base["winner"]


def test_inverted_criterion_threshold_maps_back_to_raw_score(lab):
    """README example: Postgres cost 3 (inverted, effective 7), weight 0.5, margin 1.3.

    Effective must fall below 7 - 1.3/0.5 = 4.4, i.e. the raw cost must rise above 5.6.
    """
    did = demo.build(lab)
    cells = lab.analyze(did, simulations=0)["score_sensitivity"]["most_fragile"]
    pg = next(c for c in cells if (c["option"], c["criterion"]) == ("Postgres", "cost"))
    assert pg["current_score"] == 3
    assert pg["flip_score"] == pytest.approx(5.6)
    assert pg["direction"] == "above"
    assert pg["new_winner"] == "DynamoDB"


def test_single_score_within_one_point_makes_it_fragile(lab):
    """A=(6,6), B=(5.5,6): A never loses on any weight (weakly dominant), but
    A's c0 only has to drop 0.5 points (margin 0.25 / weight 0.5) for a tie."""
    did = make_matrix(lab, {"A": (6, 6), "B": (5.5, 6)})
    out = lab.analyze(did)
    assert out["robustness_checks"]["weights"]["passed"] is True
    assert out["robustness_checks"]["scores"]["passed"] is False
    assert "A/c0" in out["robustness_checks"]["scores"]["failing_cells"]
    assert out["robustness"] == "fragile"
    assert "single judgment can flip it" in out["recommendation"]


def test_recommendation_names_the_hinge_cell_and_rationale(lab):
    did = demo.build(lab)
    rec = lab.analyze(did)["recommendation"]
    assert rec.startswith("Choose Postgres (7.1 weighted) over DynamoDB (5.8); margin 1.3.")
    assert (
        "The decision hinges most on the judgment that Postgres scores 3 on 'cost' "
        '("Managed Postgres (RDS/Neon) is cheap and predictable at our scale"): '
        "above 5.6, DynamoDB wins." in rec
    )


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------


def test_dominant_option_wins_every_simulation(lab):
    """A=(9,9), B=(1,1): with ±1 noise A's scores stay in [8,10], B's in [0,2]."""
    did = make_matrix(lab, {"A": (9, 9), "B": (1, 1)})
    mc = lab.analyze(did)["monte_carlo"]
    assert mc["options"]["A"]["win_probability"] == 1.0
    assert mc["options"]["A"]["rank_acceptability"] == [1.0, 0.0]
    assert mc["options"]["B"]["expected_rank"] == 2.0


def test_symmetric_tie_splits_win_probability(lab):
    did = make_matrix(lab, {"A": (8, 2), "B": (2, 8)})
    mc = lab.analyze(did, simulations=4000, seed=123)["monte_carlo"]
    pa = mc["options"]["A"]["win_probability"]
    assert pa == pytest.approx(0.5, abs=0.05)
    assert pa + mc["options"]["B"]["win_probability"] == pytest.approx(1.0)


def test_monte_carlo_is_reproducible_and_seed_sensitive(lab):
    did = demo.build(lab)
    a = lab.analyze(did, seed=11)["monte_carlo"]
    b = lab.analyze(did, seed=11)["monte_carlo"]
    c = lab.analyze(did, seed=12)["monte_carlo"]
    assert a == b
    assert a["options"] != c["options"]
    for s in a["options"].values():
        assert sum(s["rank_acceptability"]) == pytest.approx(1.0)
        assert s["total_p05"] <= s["total_median"] <= s["total_p95"]


def test_low_win_probability_makes_it_fragile(lab):
    """A=(6,4), B=(4,6), weights .55/.45: A 5.1 vs B 4.9. Heavy noise makes it a coin flip."""
    did = make_matrix(lab, {"A": (6, 4), "B": (4, 6)}, weights=(0.55, 0.45))
    out = lab.analyze(did, score_noise=3, weight_concentration=5)
    assert out["robustness_checks"]["monte_carlo"]["passed"] is False
    assert out["robustness"] == "fragile"
    assert "ranks first in only" in out["recommendation"]


def test_simulations_zero_skips_monte_carlo(lab):
    did = demo.build(lab)
    out = lab.analyze(did, simulations=0)
    assert out["monte_carlo"] is None
    assert out["robustness_checks"]["monte_carlo"]["passed"] is None
    assert "Monte Carlo" not in out["recommendation"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"simulations": -1}, "non-negative"),
        ({"simulations": 10**6}, "between 1 and"),
        ({"score_noise": 11}, "score_noise"),
        ({"weight_concentration": 0}, "weight_concentration"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_monte_carlo_parameter_validation(lab, kwargs, match):
    did = demo.build(lab)
    with pytest.raises(ValueError, match=match):
        lab.analyze(did, **kwargs)


# ----------------------------------------------------------------------
# what_if
# ----------------------------------------------------------------------


def test_what_if_applies_reported_flip_weights(lab):
    """Flip weights from analyze (cost 0.1176 decrease, scalability 0.4717 increase)
    change the winner once crossed, and not before."""
    did = demo.build(lab)
    sens = {e["criterion"]: e["flip"] for e in lab.analyze(did)["sensitivity"]}
    cost, scal = sens["cost"]["flip_weight"], sens["scalability"]["flip_weight"]
    assert lab.what_if(did, weights={"cost": cost - 0.005})["winner"] == "DynamoDB"
    assert lab.what_if(did, weights={"cost": cost + 0.005})["winner"] == "Postgres"
    assert lab.what_if(did, weights={"scalability": scal + 0.005})["winner"] == "DynamoDB"
    over = lab.what_if(did, weights={"scalability": scal + 0.005})
    assert over["winner_changed"] is True
    assert over["summary"].startswith("Winner changes from Postgres to DynamoDB.")
    under = lab.what_if(did, weights={"scalability": scal - 0.005})
    assert under["winner"] == "Postgres" and under["winner_changed"] is False


def test_what_if_never_modifies_the_session(lab):
    did = demo.build(lab)
    lab.analyze(did)
    before_matrix = lab.get_matrix(did)
    before_list = lab.list_decisions()
    raw_before = lab.store_path.read_bytes()
    lab.what_if(
        did,
        weights={"cost": 0.1},
        score_overrides=[{"option": "Postgres", "criterion": "cost", "score": 9}],
    )
    assert lab.get_matrix(did) == before_matrix
    assert lab.list_decisions() == before_list
    assert lab.store_path.read_bytes() == raw_before


def test_what_if_partial_weights_keep_the_others_proportional(lab):
    """cost -> 0: scalability and familiarity share 1.0 as 0.3:0.2 = 0.6/0.4.
    Postgres 0.6*6 + 0.4*9 = 7.2; DynamoDB 0.6*10 + 0.4*4 = 7.6 -> DynamoDB."""
    did = demo.build(lab)
    out = lab.what_if(did, weights={"cost": 0})
    w = {c["name"]: c["weight"] for c in out["weights_used"]}
    assert w == {"cost": 0.0, "scalability": 0.6, "team-familiarity": 0.4}
    totals = {r["option"]: r["total"] for r in out["ranking"]}
    assert totals["Postgres"] == pytest.approx(7.2)
    assert totals["DynamoDB"] == pytest.approx(7.6)
    assert out["winner"] == "DynamoDB"
    assert out["total_changes"]["Postgres"] == pytest.approx(0.1)


def test_what_if_all_weights_are_relative_importances(lab):
    """Equal importance: Postgres (7+6+9)/3 = 7.3333, DynamoDB 6.0, MongoDB 5.6667."""
    did = demo.build(lab)
    out = lab.what_if(did, weights={"cost": 1, "scalability": 1, "team-familiarity": 1})
    totals = {r["option"]: r["total"] for r in out["ranking"]}
    assert totals == pytest.approx({"Postgres": 7.3333, "DynamoDB": 6.0, "MongoDB": 5.6667})


def test_what_if_score_override(lab):
    """Postgres cost 3 -> 6 (effective 4): 0.5*4 + 1.8 + 1.8 = 5.6 < DynamoDB 5.8."""
    did = demo.build(lab)
    out = lab.what_if(
        did, score_overrides=[{"option": "postgres", "criterion": "COST", "score": 6}]
    )
    assert out["winner"] == "DynamoDB" and out["winner_changed"] is True
    assert out["score_overrides_applied"] == [
        {"option": "Postgres", "criterion": "cost", "from": 3.0, "to": 6.0}
    ]


def test_what_if_can_fill_missing_cells(lab):
    did = lab.start_decision(
        "q?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )["decision_id"]
    lab.score_option(did, "A", "x", 9, "A is great on x indeed")
    with pytest.raises(ValueError, match="needs a complete matrix"):
        lab.what_if(did, weights={"x": 0.5})
    out = lab.what_if(
        did,
        score_overrides=[
            {"option": "A", "criterion": "y", "score": 1},
            {"option": "B", "criterion": "x", "score": 2},
            {"option": "B", "criterion": "y", "score": 3},
        ],
    )
    assert out["winner"] == "A" and out["baseline"] is None and out["winner_changed"] is None


@pytest.mark.parametrize(
    "weights, match",
    [
        ({"cost": 1.5}, "normalized"),
        ({"cost": 0.7, "scalability": 0.6}, "normalized"),
        ({"nope": 0.2}, "Unknown criterion"),
        ({"cost": -1}, ">= 0"),
        ({"cost": 0, "scalability": 0, "team-familiarity": 0}, "weight > 0"),
        ({}, "non-empty"),
    ],
)
def test_what_if_weight_validation(lab, weights, match):
    did = demo.build(lab)
    with pytest.raises(ValueError, match=match):
        lab.what_if(did, weights=weights)


def test_what_if_requires_a_hypothesis(lab):
    did = demo.build(lab)
    with pytest.raises(ValueError, match="Pass weights"):
        lab.what_if(did)


def test_monte_carlo_draws_only_from_random_random(lab, monkeypatch):
    """random.random() is the only generator whose output Python promises not to
    change across versions; the seeded numbers must not depend on anything else."""
    import random

    def forbidden(*args, **kwargs):
        raise AssertionError("Monte Carlo must only use Random.random()")

    for name in ("gammavariate", "uniform", "gauss", "normalvariate", "expovariate", "betavariate"):
        monkeypatch.setattr(random.Random, name, forbidden)
    did = demo.build(lab)
    assert lab.analyze(did)["monte_carlo"]["options"]["Postgres"]["win_probability"] == 0.929


@pytest.mark.parametrize("alpha", [0.3, 1.0, 4.0, 10.0])
def test_gamma_sampler_matches_theory(alpha):
    """Gamma(alpha, 1) has mean alpha and variance alpha."""
    import random
    import statistics

    from mcp_decision_lab.analysis import _gamma

    rng = random.Random(99)
    xs = [_gamma(rng, alpha) for _ in range(40_000)]
    assert statistics.fmean(xs) == pytest.approx(alpha, rel=0.03)
    assert statistics.pvariance(xs) == pytest.approx(alpha, rel=0.06)


def test_dirichlet_weights_centre_on_the_chosen_weights():
    """Dirichlet(20 * w): mean w, sd sqrt(w (1 - w) / 21)."""
    import math
    import random
    import statistics

    from mcp_decision_lab.analysis import _dirichlet

    rng = random.Random(5)
    w = [0.5, 0.3, 0.2]
    draws = [_dirichlet(rng, [20 * x for x in w], w) for _ in range(20_000)]
    for j, wj in enumerate(w):
        col = [d[j] for d in draws]
        assert statistics.fmean(col) == pytest.approx(wj, abs=0.005)
        assert statistics.pstdev(col) == pytest.approx(math.sqrt(wj * (1 - wj) / 21), rel=0.05)
    assert all(abs(sum(d) - 1.0) < 1e-12 for d in draws[:100])
