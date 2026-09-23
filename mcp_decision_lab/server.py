"""mcp-decision-lab — MCP server.

Thin MCP wiring only: every tool delegates to :class:`core.DecisionLab`.

Works with both lines of the MCP Python SDK:

* ``mcp`` 2.x — ``mcp.server.mcpserver.MCPServer``
* ``mcp`` 1.x — ``mcp.server.fastmcp.FastMCP``

Every validation problem is raised as the SDK's ``ToolError``. In mcp 2.x any
other exception reaches the model only as "Error executing tool <name>", and
messages such as "Unknown option 'C'. Valid options: A, B" are how the calling
model corrects itself.

Run over stdio: ``mcp-decision-lab`` or ``python -m mcp_decision_lab.server``.
"""

import functools
import inspect
import threading
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _ServerClass
    from mcp.server.mcpserver.exceptions import ToolError

    SDK_LINE = 2
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _ServerClass  # type: ignore[no-redef]
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore[no-redef]

    SDK_LINE = 1

try:  # tool annotations (read-only / destructive hints) need mcp >= 1.6 (1.5 still works, without hints)
    from mcp.types import ToolAnnotations
except ImportError:  # pragma: no cover - very old SDK: tools simply carry no hints
    ToolAnnotations = None  # type: ignore[assignment,misc]

from . import __version__
from . import analysis as A
from .core import MIN_RATIONALE_CHARS, DecisionLab

INSTRUCTIONS = """\
Decision Lab turns a choice into an auditable weighted decision matrix.
Workflow: start_decision (question, >=2 options, >=2 weighted criteria) ->
score_option once per option x criterion cell (raw 0-10 score plus a written
rationale; for criteria declared higher_is_better=false score the raw quantity,
e.g. cost 10 = very expensive) -> analyze. analyze returns the ranking, the
exact weight and score thresholds that would change the winner, a Monte Carlo
win probability and a robustness verdict (robust / fragile / tie). Use what_if
to test other weights or scores without touching the session, and
export_decision for a Markdown decision record. When reporting back, give the
verdict and the cell the decision hinges on; never present a tie as a winner.
"""

# ----------------------------------------------------------------------
# Lazily built engine: importing this module has no side effects, and a
# corrupt store surfaces as a tool error instead of stopping the server.
# ----------------------------------------------------------------------

_lab: Optional[DecisionLab] = None
_lab_dir: Optional[Path] = None
_lab_lock = threading.Lock()


def configure(data_dir: "str | Path | None" = None) -> None:
    """Choose the data directory (default: $DECISION_LAB_DIR or ~/.mcp-decision-lab)."""
    global _lab, _lab_dir
    with _lab_lock:
        _lab = None
        _lab_dir = Path(data_dir) if data_dir is not None else None


def get_lab() -> DecisionLab:
    global _lab
    with _lab_lock:
        if _lab is None:
            _lab = DecisionLab(_lab_dir)  # retried on the next call if it raises
        return _lab


# ----------------------------------------------------------------------
# Server + tool registration
# ----------------------------------------------------------------------


def _supported(fn: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


mcp = _ServerClass(
    **_supported(
        _ServerClass.__init__,
        name="mcp-decision-lab",
        instructions=INSTRUCTIONS,
        version=__version__,
    )
)


def _tool(title: str, *, read_only: bool, destructive: bool = False, idempotent: bool = False):
    """Register ``fn`` as a tool whose expected errors reach the model verbatim."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except ToolError:
                raise
            except ValueError as exc:  # includes store.StoreError
                raise ToolError(str(exc)) from exc
            except OSError as exc:
                raise ToolError(f"Could not access the decision store: {exc}") from exc

        annotations = None
        if ToolAnnotations is not None:
            annotations = ToolAnnotations.model_validate(
                {
                    "readOnlyHint": read_only,
                    "destructiveHint": destructive,
                    "idempotentHint": idempotent,
                    "openWorldHint": False,
                }
            )
        mcp.tool(**_supported(mcp.tool, title=title, annotations=annotations))(wrapper)
        return wrapper

    return decorator


# ----------------------------------------------------------------------
# Typed argument models (these become the tools' JSON schemas)
# ----------------------------------------------------------------------

DecisionId = Annotated[
    str, Field(min_length=1, description="Id returned by start_decision, e.g. 'dec-a3f2'.")
]


class Criterion(BaseModel):
    """One weighted criterion of the decision matrix."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Short unique name, e.g. 'cost'.")
    weight: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Relative importance, any positive number; all weights are "
        "normalized to sum to 1.0.",
    )
    higher_is_better: bool = Field(
        default=True,
        description="false for criteria where a HIGH raw score means a WORSE option "
        "(cost, risk, complexity); the effective score becomes 10 - score.",
    )


class ScoreOverride(BaseModel):
    """A hypothetical score for one cell, used only by what_if."""

    model_config = ConfigDict(extra="forbid")

    option: str = Field(min_length=1)
    criterion: str = Field(min_length=1)
    score: float = Field(ge=0, le=10, allow_inf_nan=False, description="Raw score 0-10.")


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------


@_tool("Start a decision", read_only=False)
def start_decision(
    question: Annotated[
        str,
        Field(min_length=1, description='The decision, e.g. "Which database for the new backend?".'),
    ],
    options: Annotated[
        list[Annotated[str, Field(min_length=1)]],
        Field(min_length=2, description='At least 2 distinct alternatives, e.g. ["Postgres", "MongoDB"].'),
    ],
    criteria: Annotated[
        list[Criterion],
        Field(min_length=2, description="At least 2 weighted criteria."),
    ],
) -> dict[str, Any]:
    """Start a structured decision: the question, the options and the weighted criteria.

    Use this whenever a choice deserves explicit reasoning (a technology, a
    vendor, a plan, a design): it forces the trade-offs into a weighted
    decision matrix instead of a gut call. Returns the new decision_id, the
    normalized weights, the empty cells to fill with score_option, and the
    next step.
    """
    return get_lab().start_decision(question, options, [c.model_dump() for c in criteria])


@_tool("Score one cell", read_only=False, idempotent=True)
def score_option(
    decision_id: DecisionId,
    option: Annotated[str, Field(min_length=1, description="One of the options (case-insensitive).")],
    criterion: Annotated[
        str, Field(min_length=1, description="One of the criteria (case-insensitive).")
    ],
    score: Annotated[
        float,
        Field(
            ge=0,
            le=10,
            allow_inf_nan=False,
            description="Raw score 0-10 for the quantity the criterion names (for "
            "'cost', 10 = very expensive); inverted criteria are handled automatically.",
        ),
    ],
    rationale: Annotated[
        str,
        Field(
            min_length=MIN_RATIONALE_CHARS,
            description="Why this score (at least 10 characters). Rationales are quoted "
            "back when the decision hinges on this cell.",
        ),
    ],
) -> dict[str, Any]:
    """Score one option against one criterion (one cell of the decision matrix).

    Score each cell independently and honestly; analyze reports exactly which
    judgments the result hinges on. Re-scoring a cell overwrites it and marks
    any previous analysis stale.
    """
    return get_lab().score_option(decision_id, option, criterion, score, rationale)


@_tool("Show the matrix", read_only=True, idempotent=True)
def get_matrix(decision_id: DecisionId) -> dict[str, Any]:
    """Show the full decision matrix: raw, effective and weighted scores, totals,
    every rationale, and any cells still missing."""
    return get_lab().get_matrix(decision_id)


@_tool("Analyze the decision", read_only=False, idempotent=True)
def analyze(
    decision_id: DecisionId,
    simulations: Annotated[
        int,
        Field(
            ge=0,
            le=A.MAX_SIMULATIONS,
            description="Monte Carlo samples (0 disables the simulation).",
        ),
    ] = A.DEFAULT_SIMULATIONS,
    seed: Annotated[int, Field(description="Random seed; same seed, same numbers.")] = A.DEFAULT_SEED,
    score_noise: Annotated[
        float,
        Field(ge=0, le=10, description="Uniform ± noise added to every score in the simulation."),
    ] = A.DEFAULT_SCORE_NOISE,
    weight_concentration: Annotated[
        float,
        Field(
            gt=0,
            le=10_000,
            description="Dirichlet concentration of the simulated weights; lower = "
            "more weight uncertainty.",
        ),
    ] = A.DEFAULT_CONCENTRATION,
) -> dict[str, Any]:
    """Analyze a fully scored matrix and give a defensible recommendation.

    Returns the ranking and margin; for each criterion the exact weight at
    which the winner would flip (closed form); the cells whose score alone
    could change the winner, with the threshold and the rationale behind them;
    Monte Carlo win probabilities under joint weight and score perturbation;
    per-option strengths/weaknesses; and a verdict: "robust", "fragile" (a ±50%
    single-weight change, a ≤1-point single-score change, or a win probability
    below 80% flips it) or "tie" (winner is null and the tie-breakers are
    listed). Fails with the list of missing cells if the matrix is incomplete.
    """
    return get_lab().analyze(decision_id, simulations, seed, score_noise, weight_concentration)


@_tool("What if…", read_only=True, idempotent=True)
def what_if(
    decision_id: DecisionId,
    weights: Annotated[
        Optional[dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]]],
        Field(
            description="Criterion -> weight. Name EVERY criterion to give new relative "
            "importances, or name only some to set their exact normalized weight "
            "(0-1, e.g. a flip_weight from analyze) while the rest keep their proportions."
        ),
    ] = None,
    score_overrides: Annotated[
        Optional[list[ScoreOverride]],
        Field(description="Hypothetical raw scores for specific cells."),
    ] = None,
) -> dict[str, Any]:
    """Re-rank the options under hypothetical weights and/or scores, without
    changing the stored session. Answers "what if we cared more about X?" or
    "what if that estimate is wrong?" with the new ranking, totals, margin and
    whether the winner changes."""
    overrides = [o.model_dump() for o in score_overrides] if score_overrides else None
    return get_lab().what_if(decision_id, weights, overrides)


@_tool("Export a decision record", read_only=True, idempotent=True)
def export_decision(
    decision_id: DecisionId,
    format: Annotated[
        Literal["markdown", "json"],
        Field(description="'markdown' (an ADR-style record) or 'json' (machine-readable)."),
    ] = "markdown",
) -> dict[str, Any]:
    """Export the decision as a reviewable record: question, weights, the full
    score table, every rationale, ranking, weight and score sensitivity, Monte
    Carlo result and recommendation. The text is in the "content" field."""
    return get_lab().export_decision(decision_id, format)


@_tool("Delete a decision", read_only=False, destructive=True, idempotent=True)
def delete_decision(decision_id: DecisionId) -> dict[str, Any]:
    """Permanently delete a decision session. Consider export_decision first."""
    return get_lab().delete_decision(decision_id)


@_tool("List decisions", read_only=True, idempotent=True)
def list_decisions() -> dict[str, Any]:
    """List all decision sessions with their status ("pending": cells unscored,
    "complete": scored but not analyzed, "analyzed") and, once analyzed, the
    winner and verdict."""
    return get_lab().list_decisions()


def run(data_dir: "str | Path | None" = None) -> None:
    """Serve over stdio."""
    if data_dir is not None:
        configure(data_dir)
    mcp.run()


def main(argv: "list[str] | None" = None) -> None:
    """Console entry point (see :mod:`mcp_decision_lab.cli`)."""
    from .cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()
