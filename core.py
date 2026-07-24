"""Core logic for mcp-decision-lab.

Weighted decision matrices with criteria scoring, exact (closed-form) linear
sensitivity analysis and a defensible recommendation.

Pure Python stdlib — no external dependencies — so it can be imported and
tested without the MCP SDK installed. ``server.py`` is a thin FastMCP wrapper
around the :class:`DecisionLab` class defined here.

Math notes (sensitivity analysis)
---------------------------------
Weights are normalized to sum to 1.0. For a criterion ``c`` with weight ``w``,
varying it to ``w'`` while renormalizing the other weights proportionally
(each ``w_k`` becomes ``w_k * (1 - w') / (1 - w)``) makes every option's
weighted total *linear* in ``w'``::

    total'(o) = w' * e(o, c) + rest(o) * (1 - w') / (1 - w)

where ``e(o, c)`` is the option's effective score on ``c`` and
``rest(o) = total(o) - w * e(o, c)``. The gap between the current winner A and
a challenger B is therefore also linear in ``w'``::

    gap(w') = w' * d + (1 - w') * r
    d = e(A, c) - e(B, c)
    r = (rest(A) - rest(B)) / (1 - w)

Solving ``gap(w') = 0`` gives the exact flip weight ``w* = r / (r - d)``
(no brute force). The flip is only reported when a region of ``[0, 1]``
exists beyond ``w*`` where the challenger *strictly* wins.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ENV_DATA_DIR = "DECISION_LAB_DIR"
DEFAULT_DIR_NAME = ".mcp-decision-lab"
STORE_FILENAME = "decisions.json"

SCORE_MIN = 0.0
SCORE_MAX = 10.0
MIN_RATIONALE_CHARS = 10
ROBUSTNESS_BAND = 0.5  # "reasonable" weight change = ±50% relative
_EPS = 1e-9

_ALLOWED_CRITERION_KEYS = {"name", "weight", "higher_is_better"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _r4(x: float) -> float:
    """Round for display; +0.0 normalizes -0.0."""
    return round(x + 0.0, 4)


def _effective(score: float, higher_is_better: bool) -> float:
    """Effective score: raw score if higher is better, else inverted (10 - score)."""
    return score if higher_is_better else SCORE_MAX - score


class DecisionLab:
    """Stateful decision-matrix engine with JSON persistence.

    Parameters
    ----------
    data_dir:
        Directory where ``decisions.json`` lives. Defaults to the
        ``DECISION_LAB_DIR`` environment variable, or ``~/.mcp-decision-lab``.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            data_dir = os.environ.get(ENV_DATA_DIR) or (Path.home() / DEFAULT_DIR_NAME)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.data_dir / STORE_FILENAME
        self._decisions: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Could not read decision store at {self.store_path} ({exc}). "
                "Fix or delete the file and restart the server."
            ) from exc
        self._decisions = raw.get("decisions", {})

    def _save(self) -> None:
        payload = {"version": 1, "decisions": self._decisions}
        tmp = self.store_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.store_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, decision_id: str) -> dict:
        dec = self._decisions.get(decision_id)
        if dec is None:
            known = ", ".join(sorted(self._decisions)) or "none yet"
            raise ValueError(
                f"Decision '{decision_id}' not found (known ids: {known}). "
                "Call list_decisions to see sessions, or start_decision to create one."
            )
        return dec

    def _new_id(self) -> str:
        while True:
            did = "dec-" + uuid.uuid4().hex[:4]
            if did not in self._decisions:
                return did

    @staticmethod
    def _resolve(name: str, valid: list[str], kind: str) -> str:
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
        raise ValueError(
            f"Unknown {kind} '{name}'. Valid {kind}s: {', '.join(valid)}."
        )

    @staticmethod
    def _missing_cells(dec: dict) -> list[dict]:
        scores = dec["scores"]
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

    # ------------------------------------------------------------------
    # Public API (mirrored 1:1 by server.py tools)
    # ------------------------------------------------------------------

    def start_decision(
        self, question: str, options: list[str], criteria: list[dict]
    ) -> dict:
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
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"Criterion '{name}': 'weight' must be a number.")
            weight = float(weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"Criterion '{name}': weight must be a finite number > 0 "
                    f"(got {weight}). Weights are normalized to sum to 1.0."
                )
            hib = crit.get("higher_is_better", True)
            if not isinstance(hib, bool):
                raise ValueError(
                    f"Criterion '{name}': 'higher_is_better' must be true or false."
                )
            clean_criteria.append({"name": name, "weight": weight, "higher_is_better": hib})

        total_w = sum(c["weight"] for c in clean_criteria)
        for c in clean_criteria:
            c["weight"] = c["weight"] / total_w

        decision_id = self._new_id()
        dec = {
            "id": decision_id,
            "question": question,
            "options": clean_options,
            "criteria": clean_criteria,
            "scores": {},
            "analyzed": False,
            "created_at": _now_iso(),
        }
        self._decisions[decision_id] = dec
        self._save()

        missing = self._missing_cells(dec)
        return {
            "decision_id": decision_id,
            "question": question,
            "options": clean_options,
            "criteria": [
                {
                    "name": c["name"],
                    "weight": _r4(c["weight"]),
                    "higher_is_better": c["higher_is_better"],
                }
                for c in clean_criteria
            ],
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
        dec = self._get(decision_id)
        option = self._resolve(option, dec["options"], "option")
        criterion = self._resolve(
            criterion, [c["name"] for c in dec["criteria"]], "criterion"
        )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score must be a number between 0 and 10.")
        score = float(score)
        if not math.isfinite(score) or not (SCORE_MIN <= score <= SCORE_MAX):
            raise ValueError(
                f"score must be between {SCORE_MIN:g} and {SCORE_MAX:g} (got {score:g})."
            )
        if not isinstance(rationale, str) or len(rationale.strip()) < MIN_RATIONALE_CHARS:
            raise ValueError(
                f"rationale is required (>= {MIN_RATIONALE_CHARS} characters): "
                "briefly justify the score so the final recommendation is defensible."
            )
        rationale = rationale.strip()

        cells = dec["scores"].setdefault(option, {})
        overwrote = criterion in cells
        cells[criterion] = {"score": score, "rationale": rationale}
        if dec.get("analyzed"):
            dec["analyzed"] = False  # scores changed → prior analysis is stale
        self._save()

        missing = self._missing_cells(dec)
        total_cells = len(dec["options"]) * len(dec["criteria"])
        if missing:
            nxt = (
                f"{len(missing)} cell(s) remaining — keep calling score_option. "
                f"Next missing: {missing[0]['option']} / {missing[0]['criterion']}."
            )
        else:
            nxt = (
                f"Matrix complete — call analyze('{decision_id}') for the ranking, "
                "sensitivity analysis and recommendation."
            )
        return {
            "decision_id": decision_id,
            "recorded": {
                "option": option,
                "criterion": criterion,
                "score": score,
                "rationale": rationale,
            },
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
            row: dict[str, dict] = {}
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
            "decision_id": decision_id,
            "question": dec["question"],
            "status": self._status(dec),
            "criteria": [
                {
                    "name": c["name"],
                    "weight": _r4(c["weight"]),
                    "higher_is_better": c["higher_is_better"],
                }
                for c in dec["criteria"]
            ],
            "matrix": matrix,
            "totals": {o: _r4(t) for o, t in totals.items()},
            "totals_note": (
                "Totals cover all cells."
                if not missing
                else f"Partial totals — {len(missing)} cell(s) still unscored."
            ),
            "missing_cells": missing,
        }

    def analyze(self, decision_id: str) -> dict:
        """Ranking, exact sensitivity analysis, robustness verdict, recommendation."""
        dec = self._get(decision_id)
        missing = self._missing_cells(dec)
        if missing:
            preview = ", ".join(
                f"{m['option']}/{m['criterion']}" for m in missing[:6]
            )
            raise ValueError(
                f"Matrix incomplete — {len(missing)} cell(s) missing ({preview}"
                f"{', …' if len(missing) > 6 else ''}). Score them with "
                "score_option before calling analyze."
            )

        options = dec["options"]
        criteria = dec["criteria"]
        eff = {
            o: {
                c["name"]: _effective(
                    dec["scores"][o][c["name"]]["score"], c["higher_is_better"]
                )
                for c in criteria
            }
            for o in options
        }
        totals = {
            o: sum(c["weight"] * eff[o][c["name"]] for c in criteria) for o in options
        }
        ranking = sorted(options, key=lambda o: -totals[o])
        winner, runner_up = ranking[0], ranking[1]
        margin = totals[winner] - totals[runner_up]

        sensitivity = []
        decisive: list[str] = []
        fragile = False
        for c in criteria:
            name, w = c["name"], c["weight"]
            flips: list[tuple[float, float, str, str]] = []
            for challenger in options:
                if challenger == winner:
                    continue
                d = eff[winner][name] - eff[challenger][name]
                rest_diff = (totals[winner] - w * eff[winner][name]) - (
                    totals[challenger] - w * eff[challenger][name]
                )
                r = rest_diff / (1.0 - w)
                slope = d - r  # gap(w') = w' * (d - r) + r
                if abs(slope) < _EPS:
                    continue  # gap is constant in w' — this pair can never flip
                w_star = r / (r - d)
                if slope > 0:
                    # gap grows with w' → challenger wins below w_star
                    if w_star > _EPS:
                        flips.append((abs(w_star - w), w_star, "decrease", challenger))
                else:
                    # gap shrinks with w' → challenger wins above w_star
                    if w_star < 1.0 - _EPS:
                        flips.append((abs(w_star - w), w_star, "increase", challenger))
            entry: dict = {"criterion": name, "weight": _r4(w)}
            if flips:
                flips.sort(key=lambda f: f[0])
                dist, w_star, direction, challenger = flips[0]
                w_star = min(max(w_star, 0.0), 1.0)
                within_band = dist <= ROBUSTNESS_BAND * w + _EPS
                entry.update(
                    {
                        "decisive": True,
                        "flip": {
                            "flip_weight": _r4(w_star),
                            "direction": direction,
                            "new_winner": challenger,
                            "weight_change": _r4(w_star - w),
                            "within_50pct_band": within_band,
                        },
                    }
                )
                decisive.append(name)
                fragile = fragile or within_band
            else:
                entry.update({"decisive": False, "flip": None})
            sensitivity.append(entry)

        strengths = {}
        for o in options:
            best = max(criteria, key=lambda c: eff[o][c["name"]])
            worst = min(criteria, key=lambda c: eff[o][c["name"]])
            strengths[o] = {
                "best_criterion": {
                    "name": best["name"],
                    "effective_score": _r4(eff[o][best["name"]]),
                },
                "worst_criterion": {
                    "name": worst["name"],
                    "effective_score": _r4(eff[o][worst["name"]]),
                },
            }

        robustness = "fragile" if fragile else "robust"
        win_s = strengths[winner]
        rec = (
            f"Choose {winner} ({_r4(totals[winner])} weighted) over {runner_up} "
            f"({_r4(totals[runner_up])}); margin {_r4(margin)}. "
            f"{winner} is strongest on '{win_s['best_criterion']['name']}' "
            f"({win_s['best_criterion']['effective_score']:g}/10) and weakest on "
            f"'{win_s['worst_criterion']['name']}' "
            f"({win_s['worst_criterion']['effective_score']:g}/10). "
        )
        if robustness == "robust":
            rec += (
                "The result is robust: no ±50% relative change to any single "
                "criterion weight changes the winner."
            )
            if decisive:
                examples = []
                for e in sensitivity:
                    if e["flip"]:
                        f = e["flip"]
                        examples.append(
                            f"'{e['criterion']}' would have to {f['direction']} "
                            f"from {e['weight']:g} to {f['flip_weight']:g} for "
                            f"{f['new_winner']} to win"
                        )
                rec += " (Extreme shifts could still flip it: " + "; ".join(examples) + ".)"
        else:
            frag = [e for e in sensitivity if e["flip"] and e["flip"]["within_50pct_band"]]
            parts = [
                f"if '{e['criterion']}' weight {'drops below' if e['flip']['direction'] == 'decrease' else 'rises above'} "
                f"{e['flip']['flip_weight']:g} (now {e['weight']:g}), "
                f"{e['flip']['new_winner']} wins"
                for e in frag
            ]
            rec += (
                "The result is FRAGILE — it hinges on "
                + ", ".join(f"'{e['criterion']}'" for e in frag)
                + ": "
                + "; ".join(parts)
                + ". Double-check those weights before committing."
            )

        dec["analyzed"] = True
        self._save()

        return {
            "decision_id": decision_id,
            "question": dec["question"],
            "ranking": [
                {"rank": i + 1, "option": o, "total": _r4(totals[o])}
                for i, o in enumerate(ranking)
            ],
            "winner": winner,
            "runner_up": runner_up,
            "margin": _r4(margin),
            "sensitivity": sensitivity,
            "decisive_criteria": decisive,
            "strengths_weaknesses": strengths,
            "robustness": robustness,
            "recommendation": rec,
        }

    def list_decisions(self) -> dict:
        """All decision sessions with their status."""
        items = []
        for dec in sorted(self._decisions.values(), key=lambda d: d["created_at"]):
            total_cells = len(dec["options"]) * len(dec["criteria"])
            missing = len(self._missing_cells(dec))
            items.append(
                {
                    "decision_id": dec["id"],
                    "question": dec["question"],
                    "status": self._status(dec),
                    "options": dec["options"],
                    "criteria": [c["name"] for c in dec["criteria"]],
                    "cells_scored": total_cells - missing,
                    "cells_total": total_cells,
                    "created_at": dec["created_at"],
                }
            )
        return {
            "count": len(items),
            "decisions": items,
            "statuses": {
                "pending": "matrix has unscored cells",
                "complete": "fully scored, not yet analyzed",
                "analyzed": "analyze() has been run on the current scores",
            },
        }
