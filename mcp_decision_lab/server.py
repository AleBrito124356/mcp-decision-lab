"""mcp-decision-lab — MCP server entry point.

Thin FastMCP wiring only: every tool delegates to core.DecisionLab.
Run over stdio: python -m mcp_decision_lab.server
"""

from mcp.server.fastmcp import FastMCP

from .core import DecisionLab

mcp = FastMCP("mcp-decision-lab")
lab = DecisionLab()


@mcp.tool()
def start_decision(question: str, options: list[str], criteria: list[dict]) -> dict:
    """Start a structured decision: define the question, the options and the weighted criteria.

    Use this as the first step whenever you (the model) face a choice worth
    reasoning about explicitly — picking a technology, a vendor, a plan, a
    design. It forces the trade-offs into a weighted decision matrix instead
    of a gut call.

    Args:
        question: The decision being made, e.g. "Which database for the new backend?".
        options: At least 2 distinct alternatives, e.g. ["Postgres", "MongoDB"].
        criteria: At least 2 dicts, each {"name": str, "weight": float,
            "higher_is_better": bool (optional, default true)}. Weights are
            relative importances (any positive numbers) and are normalized to
            sum to 1.0. Set higher_is_better=false for criteria like "cost" or
            "risk" where a HIGH raw score means a WORSE option — the effective
            score becomes (10 - score).

    Returns:
        The new decision_id, the normalized criteria, the list of empty matrix
        cells, and the next step: score every option/criterion pair with
        score_option, then call analyze.
    """
    return lab.start_decision(question, options, criteria)


@mcp.tool()
def score_option(
    decision_id: str, option: str, criterion: str, score: float, rationale: str
) -> dict:
    """Score one option against one criterion (one cell of the decision matrix).

    Score each cell independently and honestly — the sensitivity analysis in
    analyze() will reveal how much each judgment matters. Re-scoring an
    existing cell overwrites it (and marks any previous analysis stale).

    Args:
        decision_id: Id returned by start_decision (e.g. "dec-a3f2").
        option: One of the decision's options (case-insensitive match).
        criterion: One of the decision's criteria (case-insensitive match).
        score: 0-10. Always score the raw quantity the criterion names
            (e.g. for "cost", 10 = very expensive); criteria declared with
            higher_is_better=false are inverted automatically.
        rationale: Required justification (>= 10 characters) — this is what
            makes the final recommendation defensible.

    Returns:
        The recorded cell, how many cells remain, which ones, and what to do
        next (keep scoring, or call analyze when the matrix is complete).
    """
    return lab.score_option(decision_id, option, criterion, score, rationale)


@mcp.tool()
def get_matrix(decision_id: str) -> dict:
    """Show the full decision matrix for a session.

    Returns raw scores, effective scores (inverted where higher_is_better is
    false), weighted scores, per-option totals, the rationale behind every
    cell, and any cells still missing. Useful to review the state of the
    decision before or after analyze.

    Args:
        decision_id: Id returned by start_decision.
    """
    return lab.get_matrix(decision_id)


@mcp.tool()
def analyze(decision_id: str) -> dict:
    """Analyze a completed matrix: ranking, sensitivity analysis and recommendation.

    Requires every cell to be scored (fails with the list of missing cells
    otherwise). Returns:

    - ranking: options ordered by weighted total, plus the margin between
      1st and 2nd place.
    - sensitivity: for each criterion, the exact weight (closed-form, with the
      other weights renormalized proportionally) at which the winner would
      flip to another option, if such a weight exists in [0, 1]. Criteria
      that can flip the winner are marked "decisive".
    - strengths_weaknesses: each option's best and worst criterion.
    - robustness: "robust" if no ±50% relative change to any single criterion
      weight flips the winner, "fragile" otherwise.
    - recommendation: a defensible textual recommendation built from the above.

    Args:
        decision_id: Id returned by start_decision.
    """
    return lab.analyze(decision_id)


@mcp.tool()
def list_decisions() -> dict:
    """List all decision sessions with their status.

    Statuses: "pending" (matrix has unscored cells), "complete" (fully scored,
    not yet analyzed), "analyzed" (analyze has been run on the current scores).
    """
    return lab.list_decisions()


def main() -> None:
    """Entry point for the console script."""
    mcp.run()


if __name__ == "__main__":
    main()
