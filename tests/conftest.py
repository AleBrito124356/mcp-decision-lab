import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(ROOT))

from mcp_decision_lab.core import DecisionLab  # noqa: E402


@pytest.fixture()
def lab(tmp_path):
    return DecisionLab(data_dir=tmp_path)


def make_matrix(lab, rows, weights=(1, 1), names=None, higher_is_better=None):
    """Create and fully score a decision. ``rows`` maps option -> scores per criterion."""
    names = names or [f"c{i}" for i in range(len(weights))]
    hib = higher_is_better or [True] * len(weights)
    did = lab.start_decision(
        "test decision?",
        list(rows),
        [{"name": n, "weight": w, "higher_is_better": h} for n, w, h in zip(names, weights, hib)],
    )["decision_id"]
    for opt, scores in rows.items():
        for n, s in zip(names, scores):
            lab.score_option(did, opt, n, s, f"{opt} scores {s} on {n}")
    return did
