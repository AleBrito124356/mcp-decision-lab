"""Pure decision-matrix math (stdlib only, no I/O).

Everything here works on a :class:`Problem`: options x criteria, normalized
weights and effective scores. :mod:`mcp_decision_lab.core` builds a Problem
from a stored session and turns the results into tool output.

Weight sensitivity (closed form)
--------------------------------
Weights are normalized to sum to 1. Moving criterion ``c`` from weight ``w``
to ``w'`` while the other criteria keep their *relative* weights makes each
option's total linear in ``w'``::

    total'(o) = w' * e(o, c) + (1 - w') * rest(o)
    rest(o)   = sum_{k != c} (w_k / W) * e(o, k),   W = sum_{k != c} w_k

For the winner A and a challenger B the gap is linear too::

    gap(w') = w' * d + (1 - w') * r,   d = e(A,c) - e(B,c),   r = rest(A) - rest(B)

so ``gap(0) = r`` and ``gap(1) = d``. B can only overtake beyond the exact
root ``w* = r / (r - d)``: below it when ``r < 0`` (lower the weight), above it
when ``d < 0`` (raise it). ``r`` is computed from the relative weights of the
*other* criteria, never by dividing by ``1 - w``, so a criterion that carries
essentially all of the weight (``1 - w`` rounds to 0) is handled exactly.

Score sensitivity (closed form)
-------------------------------
Totals are linear in every cell, so lowering the winner's effective score on
``c`` by ``delta`` lowers its total by ``w_c * delta``: the runner-up takes over
once ``delta > margin / w_c``. Symmetrically a challenger overtakes once its
effective score rises by more than ``(total_A - total_B) / w_c``. A threshold is
reported only if it is reachable inside the 0-10 scale; inverted criteria
(``higher_is_better = false``) are mapped back to raw scores.

Monte Carlo
-----------
Joint perturbation of every weight and every score at once: weights are drawn
from a Dirichlet distribution centred on the chosen weights
(``alpha_i = concentration * w_i``) and each score gets independent uniform
noise ``±score_noise``, clipped to 0-10. The result is each option's
probability of ranking 1st (its "rank-1 acceptability"), plus the full rank
distribution.

Every variate is built from :meth:`random.Random.random` alone (Marsaglia-Tsang
gamma sampling on Box-Muller normals). That method is the one part of the
``random`` module whose output Python guarantees not to change between
versions for a given seed; ``gammavariate`` and friends carry no such promise.
So a seed gives the same numbers on every supported Python.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

SCORE_MIN = 0.0
SCORE_MAX = 10.0

WEIGHT_BAND = 0.5
"""A 'reasonable' single-weight change: ±50% of the weight's current value."""

SCORE_BAND = 1.0
"""A 'reasonable' single-score change: ±1 point on the 0-10 scale."""

MC_ROBUST_MIN = 0.8
"""Minimum Monte Carlo win probability for a 'robust' verdict."""

DEFAULT_SIMULATIONS = 2000
DEFAULT_SEED = 7
DEFAULT_SCORE_NOISE = 1.0
DEFAULT_CONCENTRATION = 20.0
MAX_SIMULATIONS = 50_000

_REL_TOL = 1e-9


def r4(x: float) -> float:
    """Round for display; ``+ 0.0`` turns -0.0 into 0.0."""
    return round(x + 0.0, 4)


def effective(score: float, higher_is_better: bool) -> float:
    """Effective score: the raw score, or ``10 - score`` for inverted criteria."""
    return score if higher_is_better else SCORE_MAX - score


def tie_tolerance(values: list[float]) -> float:
    return _REL_TOL * max(1.0, max((abs(v) for v in values), default=1.0))


@dataclass
class Problem:
    options: list[str]
    criteria: list[str]
    weights: list[float]  # normalized, sum to 1
    higher_is_better: list[bool]
    raw: list[list[float]]  # raw[i][j]: option i on criterion j
    rationales: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.eff = [
            [effective(self.raw[i][j], self.higher_is_better[j]) for j in range(len(self.criteria))]
            for i in range(len(self.options))
        ]

    def totals(self, weights: list[float] | None = None) -> list[float]:
        ws = self.weights if weights is None else weights
        return [math.fsum(w * e for w, e in zip(ws, row)) for row in self.eff]


# ----------------------------------------------------------------------
# Ranking and ties
# ----------------------------------------------------------------------


def rank(options: list[str], totals: list[float]) -> list[dict]:
    """Options by total, best first, with competition ranks (tied totals share a rank)."""
    tol = tie_tolerance(totals)
    order = sorted(range(len(options)), key=lambda i: (-totals[i], i))
    out: list[dict] = []
    for pos, i in enumerate(order):
        if out and abs(totals[order[pos - 1]] - totals[i]) <= tol:
            r = out[-1]["rank"]
        else:
            r = pos + 1
        out.append({"rank": r, "option": options[i], "total": r4(totals[i])})
    return out


def leaders(totals: list[float]) -> list[int]:
    """Indices of every option tied (within tolerance) for the top total, in input order."""
    tol = tie_tolerance(totals)
    top = max(totals)
    return [i for i, t in enumerate(totals) if top - t <= tol]


# ----------------------------------------------------------------------
# Weight sensitivity
# ----------------------------------------------------------------------


def _rest_diff(p: Problem, a: int, b: int, j: int) -> float | None:
    """Relative-weight average of (e(a,k) - e(b,k)) over the criteria k != j."""
    others = [k for k in range(len(p.criteria)) if k != j]
    mass = math.fsum(p.weights[k] for k in others)
    if mass <= 0.0:
        return None  # the other criteria carry no weight at all
    return math.fsum((p.weights[k] / mass) * (p.eff[a][k] - p.eff[b][k]) for k in others)


def weight_sensitivity(p: Problem, winner: int) -> list[dict]:
    """Per criterion: the exact weight at which some other option would take over."""
    out = []
    for j, name in enumerate(p.criteria):
        w = p.weights[j]
        best: tuple[float, float, str, int] | None = None
        for b in range(len(p.options)):
            if b == winner:
                continue
            d = p.eff[winner][j] - p.eff[b][j]
            r = _rest_diff(p, winner, b, j)
            if r is None:
                continue
            tol = _REL_TOL * SCORE_MAX
            if r < -tol:  # challenger wins at w' = 0 -> flip when lowering
                w_star, direction = r / (r - d), "decrease"
            elif d < -tol:  # challenger wins at w' = 1 -> flip when raising
                w_star, direction = r / (r - d), "increase"
            else:
                continue
            w_star = min(max(w_star, 0.0), 1.0)
            cand = (abs(w_star - w), w_star, direction, b)
            if best is None or cand[0] < best[0]:
                best = cand
        entry: dict = {"criterion": name, "weight": r4(w)}
        if best is None:
            entry.update({"decisive": False, "flip": None})
        else:
            dist, w_star, direction, b = best
            entry.update(
                {
                    "decisive": True,
                    "flip": {
                        "flip_weight": r4(w_star),
                        "direction": direction,
                        "new_winner": p.options[b],
                        "weight_change": r4(w_star - w),
                        "within_50pct_band": dist <= WEIGHT_BAND * w + 1e-12,
                    },
                }
            )
        out.append(entry)
    return out


def tie_breakers(p: Problem, tied: list[int]) -> list[dict]:
    """For a tie: which tied option an infinitesimal weight change on each criterion favors.

    Among options with equal totals, d total / d w_c is proportional to
    ``e(o, c) - total``, so raising ``c``'s weight favors the tied option with the
    highest effective score on ``c`` and lowering it favors the lowest.
    """
    out = []
    for j, name in enumerate(p.criteria):
        vals = [p.eff[i][j] for i in tied]
        hi, lo = max(vals), min(vals)
        entry: dict = {"criterion": name, "weight": r4(p.weights[j]), "flip": None}
        if hi - lo <= _REL_TOL * SCORE_MAX:
            entry.update({"decisive": False, "tie_break": None})
        else:
            entry.update(
                {
                    "decisive": True,
                    "tie_break": {
                        "raise_weight_favors": [p.options[i] for i in tied if p.eff[i][j] == hi],
                        "lower_weight_favors": [p.options[i] for i in tied if p.eff[i][j] == lo],
                        "effective_scores": {p.options[i]: r4(p.eff[i][j]) for i in tied},
                    },
                }
            )
        out.append(entry)
    return out


# ----------------------------------------------------------------------
# Score sensitivity
# ----------------------------------------------------------------------


def score_sensitivity(p: Problem, winner: int, runner_up: int) -> list[dict]:
    """Every cell whose score alone can change the winner, most fragile first."""
    totals = p.totals()
    margin = totals[winner] - totals[runner_up]
    cells = []
    for i, opt in enumerate(p.options):
        for j, crit in enumerate(p.criteria):
            w = p.weights[j]
            if w <= 0.0:
                continue
            e = p.eff[i][j]
            if i == winner:
                eff_star = e - margin / w  # winner must fall below this
                if not eff_star > SCORE_MIN:
                    continue
                new_winner = p.options[runner_up]
                eff_side = "below"
            else:
                eff_star = e + (totals[winner] - totals[i]) / w  # challenger must exceed
                if not eff_star < SCORE_MAX:
                    continue
                new_winner = opt
                eff_side = "above"
            if p.higher_is_better[j]:
                raw_star, side = eff_star, eff_side
            else:  # effective = 10 - raw: flip the scale and the side
                raw_star = SCORE_MAX - eff_star
                side = "above" if eff_side == "below" else "below"
            change = abs(raw_star - p.raw[i][j])
            cells.append(
                {
                    "option": opt,
                    "criterion": crit,
                    "current_score": r4(p.raw[i][j]),
                    "flip_score": r4(raw_star),
                    "direction": side,
                    "change_needed": r4(change),
                    "new_winner": new_winner,
                    "within_1pt": change <= SCORE_BAND + 1e-12,
                    "rationale": p.rationales[i][j] if p.rationales else "",
                    "_key": (round(change, 9), i != winner, i, j),
                }
            )
    # Smallest change first; on equal changes, the winner's own cells first,
    # then input order (deterministic, independent of names).
    cells.sort(key=lambda c: c["_key"])
    for c in cells:
        del c["_key"]
    return cells


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------


def _normal(rng: random.Random) -> float:
    """Standard normal via Box-Muller, from ``rng.random()`` only."""
    u1 = 1.0 - rng.random()  # (0, 1]: log is always finite
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _gamma(rng: random.Random, alpha: float) -> float:
    """Gamma(alpha, 1) by Marsaglia & Tsang (2000), from ``rng.random()`` only.

    For alpha < 1 it uses the standard boost Gamma(a) = Gamma(a + 1) * U**(1/a).
    """
    if alpha < 1.0:
        u = 1.0 - rng.random()
        return _gamma(rng, alpha + 1.0) * u ** (1.0 / alpha)
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = _normal(rng)
        v = 1.0 + c * x
        if v <= 0.0:
            continue
        v = v * v * v
        u = 1.0 - rng.random()
        if u < 1.0 - 0.0331 * x**4 or math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v


def _dirichlet(rng: random.Random, alphas: list[float], fallback: list[float]) -> list[float]:
    g = [_gamma(rng, a) if a > 0 else 0.0 for a in alphas]
    s = math.fsum(g)
    if not s > 0.0 or not math.isfinite(s):
        return list(fallback)
    return [x / s for x in g]


def monte_carlo(
    p: Problem,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    score_noise: float = DEFAULT_SCORE_NOISE,
    concentration: float = DEFAULT_CONCENTRATION,
) -> dict:
    """Joint weight + score perturbation; win probability and rank distribution per option."""
    if not isinstance(simulations, int) or isinstance(simulations, bool):
        raise ValueError("simulations must be an integer.")
    if not 1 <= simulations <= MAX_SIMULATIONS:
        raise ValueError(f"simulations must be between 1 and {MAX_SIMULATIONS} (got {simulations}).")
    if not (math.isfinite(score_noise) and 0.0 <= score_noise <= SCORE_MAX):
        raise ValueError(f"score_noise must be between 0 and 10 (got {score_noise}).")
    if not (math.isfinite(concentration) and concentration > 0):
        raise ValueError(f"weight_concentration must be a finite number > 0 (got {concentration}).")

    rng = random.Random(seed)
    m, n = len(p.options), len(p.criteria)
    alphas = [concentration * w for w in p.weights]
    rank_counts = [[0.0] * m for _ in range(m)]
    samples: list[list[float]] = [[] for _ in range(m)]

    for _ in range(simulations):
        ws = _dirichlet(rng, alphas, p.weights)
        tots = []
        for i in range(m):
            t = 0.0
            row = p.eff[i]
            for j in range(n):
                e = row[j]
                if score_noise > 0.0:
                    noise = score_noise * (2.0 * rng.random() - 1.0)
                    e = min(SCORE_MAX, max(SCORE_MIN, e + noise))
                t += ws[j] * e
            tots.append(t)
            samples[i].append(t)
        # rank with fractional credit for ties
        tol = tie_tolerance(tots)
        order = sorted(range(m), key=lambda i: -tots[i])
        pos = 0
        while pos < m:
            grp = [order[pos]]
            while pos + len(grp) < m and abs(tots[order[pos]] - tots[order[pos + len(grp)]]) <= tol:
                grp.append(order[pos + len(grp)])
            share = 1.0 / len(grp)
            for i in grp:
                for rpos in range(pos, pos + len(grp)):
                    rank_counts[i][rpos] += share
            pos += len(grp)

    def pct(xs: list[float], q: float) -> float:
        xs = sorted(xs)
        k = (len(xs) - 1) * q
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    per_option = {}
    for i, opt in enumerate(p.options):
        dist = [c / simulations for c in rank_counts[i]]
        per_option[opt] = {
            "win_probability": r4(dist[0]),
            "rank_acceptability": [r4(x) for x in dist],
            "expected_rank": r4(sum((k + 1) * x for k, x in enumerate(dist))),
            "total_p05": r4(pct(samples[i], 0.05)),
            "total_median": r4(pct(samples[i], 0.5)),
            "total_p95": r4(pct(samples[i], 0.95)),
        }
    most_likely = max(p.options, key=lambda o: per_option[o]["win_probability"])
    return {
        "simulations": simulations,
        "seed": seed,
        "score_noise": score_noise,
        "weight_concentration": concentration,
        "weight_spread_sd": {
            p.criteria[j]: r4(math.sqrt(w * (1 - w) / (concentration + 1)))
            for j, w in enumerate(p.weights)
        },
        "options": per_option,
        "most_likely_winner": most_likely,
    }


# ----------------------------------------------------------------------
# Strengths and weaknesses
# ----------------------------------------------------------------------


def strengths(p: Problem) -> dict:
    out = {}
    for i, opt in enumerate(p.options):
        row = p.eff[i]
        hi, lo = max(row), min(row)
        if hi - lo <= _REL_TOL * SCORE_MAX:
            out[opt] = {
                "uniform": True,
                "effective_score": r4(hi),
                "best_criterion": None,
                "worst_criterion": None,
            }
            continue
        b = row.index(hi)
        w = row.index(lo)
        out[opt] = {
            "uniform": False,
            "best_criterion": {"name": p.criteria[b], "effective_score": r4(hi)},
            "worst_criterion": {"name": p.criteria[w], "effective_score": r4(lo)},
        }
    return out


# ----------------------------------------------------------------------
# Full analysis
# ----------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{r4(x):g}"


def _join(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _strength_text(opt: str, s: dict) -> str:
    if s["uniform"]:
        return f"{opt} scores {s['effective_score']:g}/10 on every criterion. "
    return (
        f"{opt} is strongest on '{s['best_criterion']['name']}' "
        f"({s['best_criterion']['effective_score']:g}/10) and weakest on "
        f"'{s['worst_criterion']['name']}' ({s['worst_criterion']['effective_score']:g}/10). "
    )


def _mc_text(mc: dict, subject: list[str]) -> str:
    parts = [
        f"{o} {mc['options'][o]['win_probability'] * 100:.1f}%" for o in subject
    ]
    return (
        f"Monte Carlo ({mc['simulations']} joint perturbations of all weights and "
        f"scores ±{mc['score_noise']:g}, seed {mc['seed']}): win probability "
        + ", ".join(parts)
        + "."
    )


def analyze_problem(
    p: Problem,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    score_noise: float = DEFAULT_SCORE_NOISE,
    concentration: float = DEFAULT_CONCENTRATION,
    top_cells: int | None = 5,
) -> dict:
    """Ranking, sensitivity (weights and scores), Monte Carlo, verdict, recommendation."""
    totals = p.totals()
    ranking = rank(p.options, totals)
    tied = leaders(totals)
    strengths_out = strengths(p)
    mc = monte_carlo(p, simulations, seed, score_noise, concentration) if simulations else None

    if len(tied) > 1:
        return _analyze_tie(p, totals, ranking, tied, strengths_out, mc)

    winner = tied[0]
    others = [i for i in range(len(p.options)) if i != winner]
    runner = max(others, key=lambda i: (totals[i], -i))
    margin = totals[winner] - totals[runner]

    sens = weight_sensitivity(p, winner)
    decisive = [e["criterion"] for e in sens if e["decisive"]]
    cells = score_sensitivity(p, winner, runner)

    weight_fragile = [e for e in sens if e["flip"] and e["flip"]["within_50pct_band"]]
    score_fragile = [c for c in cells if c["within_1pt"]]
    win_p = mc["options"][p.options[winner]]["win_probability"] if mc else None
    mc_fragile = win_p is not None and win_p < MC_ROBUST_MIN

    checks = {
        "weights": {
            "passed": not weight_fragile,
            "rule": "no ±50% relative change to a single criterion weight flips the winner",
            "failing_criteria": [e["criterion"] for e in weight_fragile],
        },
        "scores": {
            "passed": not score_fragile,
            "rule": "no change of 1 point or less to a single score flips the winner",
            "failing_cells": [f"{c['option']}/{c['criterion']}" for c in score_fragile],
        },
        "monte_carlo": {
            "passed": None if mc is None else not mc_fragile,
            "rule": f"winner ranks 1st in at least {MC_ROBUST_MIN:.0%} of joint perturbations",
            "win_probability": win_p,
        },
    }
    robust = not (weight_fragile or score_fragile or mc_fragile)
    wname, rname = p.options[winner], p.options[runner]

    rec = (
        f"Choose {wname} ({_fmt(totals[winner])} weighted) over {rname} "
        f"({_fmt(totals[runner])}); margin {_fmt(margin)}. "
        + _strength_text(wname, strengths_out[wname])
    )
    if robust:
        rec += (
            "The result is robust: no ±50% relative change to any single "
            "criterion weight changes the winner"
        )
        if cells:
            rec += (
                f", no single score is within 1 point of flipping it (the closest "
                f"needs a {cells[0]['change_needed']:g}-point change)"
            )
        else:
            rec += ", and no single score change within 0-10 can flip it"
        rec += "."
        if decisive:
            examples = [
                f"'{e['criterion']}' would have to {e['flip']['direction']} from "
                f"{e['weight']:g} to {e['flip']['flip_weight']:g} for "
                f"{e['flip']['new_winner']} to win"
                for e in sens
                if e["flip"]
            ]
            rec += " (Extreme shifts could still flip it: " + "; ".join(examples) + ".)"
    else:
        reasons = []
        if weight_fragile:
            parts = [
                f"if '{e['criterion']}' weight "
                f"{'drops below' if e['flip']['direction'] == 'decrease' else 'rises above'} "
                f"{e['flip']['flip_weight']:g} (now {e['weight']:g}), "
                f"{e['flip']['new_winner']} wins"
                for e in weight_fragile
            ]
            reasons.append(
                "it hinges on "
                + ", ".join(f"'{e['criterion']}'" for e in weight_fragile)
                + ": "
                + "; ".join(parts)
            )
        if score_fragile:
            parts = [
                f"{c['option']}'s '{c['criterion']}' score {c['current_score']:g} only has to go "
                f"{c['direction']} {c['flip_score']:g} for {c['new_winner']} to win"
                for c in score_fragile[:3]
            ]
            reasons.append("a single judgment can flip it: " + "; ".join(parts))
        if mc_fragile:
            reasons.append(
                f"{wname} ranks first in only {win_p * 100:.1f}% of joint perturbations"
            )
        rec += (
            "The result is FRAGILE — "
            + "; and ".join(reasons)
            + ". Double-check those weights and scores before committing."
        )
    if cells:
        c = cells[0]
        rec += (
            f" The decision hinges most on the judgment that {c['option']} scores "
            f"{c['current_score']:g} on '{c['criterion']}'"
            + (f" (\"{c['rationale']}\")" if c["rationale"] else "")
            + f": {c['direction']} {c['flip_score']:g}, {c['new_winner']} wins."
        )
    if mc:
        rec += " " + _mc_text(mc, [wname, rname])

    return {
        "ranking": ranking,
        "winner": wname,
        "tied": [],
        "runner_up": rname,
        "margin": r4(margin),
        "sensitivity": sens,
        "decisive_criteria": decisive,
        "score_sensitivity": {
            "flippable_cells": len(cells),
            "cells_total": len(p.options) * len(p.criteria),
            "most_fragile": cells if top_cells is None else cells[:top_cells],
        },
        "monte_carlo": mc,
        "strengths_weaknesses": strengths_out,
        "robustness": "robust" if robust else "fragile",
        "robustness_checks": checks,
        "recommendation": rec,
    }


def _analyze_tie(
    p: Problem,
    totals: list[float],
    ranking: list[dict],
    tied: list[int],
    strengths_out: dict,
    mc: dict | None,
) -> dict:
    names = [p.options[i] for i in tied]
    breakers = tie_breakers(p, tied)
    decisive = [e["criterion"] for e in breakers if e["decisive"]]
    rec = (
        f"No single winner: {_join(names)} are tied at {_fmt(totals[tied[0]])} weighted "
        "(margin 0). Do not pick one by list order — the matrix cannot separate them. "
    )
    if decisive:
        parts = []
        for e in breakers:
            tb = e["tie_break"]
            if not tb:
                continue
            parts.append(
                f"giving '{e['criterion']}' more weight favors {_join(tb['raise_weight_favors'])}, "
                f"less weight favors {_join(tb['lower_weight_favors'])}"
            )
        rec += (
            "What would break the tie: "
            + "; ".join(parts)
            + ". Decide which of those criteria matters more (then re-weight with "
            "what_if or a new session), or add a criterion that separates the options."
        )
    else:
        rec += (
            f"{_join(names)} have identical effective scores on every criterion, so no "
            "weight change can separate them: add a criterion that distinguishes them, "
            "or re-check the scores."
        )
    if mc:
        rec += " " + _mc_text(mc, names)
    return {
        "ranking": ranking,
        "winner": None,
        "tied": names,
        "runner_up": None,
        "margin": 0.0,
        "sensitivity": breakers,
        "decisive_criteria": decisive,
        "score_sensitivity": {
            "flippable_cells": None,
            "cells_total": len(p.options) * len(p.criteria),
            "most_fragile": [],
            "note": "With a tie, raising any tied option's effective score on any "
            "criterion by any amount makes it the winner.",
        },
        "monte_carlo": mc,
        "strengths_weaknesses": strengths_out,
        "robustness": "tie",
        "robustness_checks": None,
        "recommendation": rec,
    }
