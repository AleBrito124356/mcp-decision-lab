"""The README's database example, as data: 3 options x 3 criteria, every cell
scored with a rationale. ``mcp-decision-lab demo`` runs it offline and the
test suite pins its numbers so the README cannot drift from the code."""

from __future__ import annotations

from .core import DecisionLab

QUESTION = "Which database should we use for the new SaaS backend?"
OPTIONS = ["Postgres", "MongoDB", "DynamoDB"]
CRITERIA = [
    {"name": "cost", "weight": 0.5, "higher_is_better": False},
    {"name": "scalability", "weight": 0.3},
    {"name": "team-familiarity", "weight": 0.2},
]
# option -> criterion -> (raw score, rationale). For "cost" (higher_is_better
# false) the raw score is how expensive the option is: 10 = very expensive.
SCORES: dict[str, dict[str, tuple[float, str]]] = {
    "Postgres": {
        "cost": (3, "Managed Postgres (RDS/Neon) is cheap and predictable at our scale"),
        "scalability": (6, "Vertical scaling plus read replicas covers years of growth; sharding is manual"),
        "team-familiarity": (9, "Every backend engineer has run Postgres in production"),
    },
    "MongoDB": {
        "cost": (5, "Atlas is moderate at first but dedicated clusters get pricey fast"),
        "scalability": (7, "Built-in sharding scales writes horizontally with some tuning"),
        "team-familiarity": (5, "Two engineers have used it; the rest only know tutorials"),
    },
    "DynamoDB": {
        "cost": (6, "On-demand pricing gets expensive with our read-heavy access patterns"),
        "scalability": (10, "Effectively unlimited managed horizontal scale"),
        "team-familiarity": (4, "Only one engineer knows single-table design patterns"),
    },
}


def build(lab: DecisionLab) -> str:
    """Create and fully score the example session; returns its decision_id."""
    did = lab.start_decision(QUESTION, OPTIONS, CRITERIA)["decision_id"]
    for option, cells in SCORES.items():
        for criterion, (score, rationale) in cells.items():
            lab.score_option(did, option, criterion, score, rationale)
    return did
