# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-09-23

### Fixed

- **The server did not start on a fresh install.** `mcp>=1.2.0` had no upper bound and now resolves to mcp 2.x, which removed `mcp.server.fastmcp`. The server now supports both SDK lines (mcp 2.x `MCPServer`, mcp 1.x `FastMCP`) and pins `mcp>=1.5,<3`. 1.5 is the oldest release whose stdio transport passes the test suite.
- **Validation messages now reach the model.** Tools raise the SDK's `ToolError`, so the model sees messages like `Unknown option 'C'. Valid options: A, B.` A plain port to mcp 2.x would have reduced every error to "Error executing tool …".
- **Ties were reported as clear, robust wins.** An exact tie used to say "Choose A … The result is robust". A trade-off tie said "Choose A" and reported a flip at the current weight. Ties are now detected with a tolerance. `winner` is `null`, `tied` lists the options, `robustness` is `"tie"`, and the recommendation names which weight change would break the tie.
- `analyze` no longer crashes with `ZeroDivisionError` when one criterion's normalized weight rounds to 1.0 (e.g. weights `[1e20, 1]`). Huge weights (e.g. `[1e308, 1e308]`) no longer overflow normalization into all-zero weights. Weight ratios beyond floating-point range are rejected with a clear error.
- An option that scores the same on every criterion was named as both its best and its worst. It is now reported as `uniform`.
- **Two server processes sharing a store lost data.** Each process kept its own copy in memory and rewrote the whole file, so the last writer silently erased the other's sessions. With 4 processes creating 10 decisions each, 11 of 40 survived, and on Windows some writes crashed with `PermissionError`. Writes now run under a cross-process lock and re-read the store first, and reads refresh when another process changes the file.
- A store file with an unexpected shape (e.g. `[]`) crashed with a raw `AttributeError`. Because the store was opened at import time, a bad store also stopped the server from starting at all. The store is now validated with a clear "Fix or delete the file" error, it is never overwritten, and it is opened lazily on the first tool call, so the problem shows up as a tool error.
- Importing `mcp_decision_lab.server` no longer creates `~/.mcp-decision-lab` as a side effect.

### Added

- **Score sensitivity** in `analyze`: for every cell, the exact (closed-form) score beyond which the winner changes, with that cell's rationale. The recommendation names the judgment the decision hinges on.
- **Monte Carlo robustness** in `analyze`: joint Dirichlet weight and uniform score perturbation, using only the stdlib. Every draw comes from `random.random()`, the one generator Python guarantees across versions, so a seed reproduces the same numbers on any supported Python. It reports win probability (rank-1 acceptability), the rank distribution, expected rank and total percentiles. It takes new optional arguments `simulations`, `seed`, `score_noise` and `weight_concentration`.
- `robustness_checks` in `analyze`, explaining the verdict (weights, scores, Monte Carlo).
- `what_if` tool: re-rank under hypothetical weights and/or scores without changing the stored session.
- `export_decision` tool: an ADR-style Markdown decision record (or JSON) with every rationale and the full analysis.
- `delete_decision` tool.
- Typed tool schemas: a `Criterion` object, 0–10 score bounds, `minItems`, and a minimum rationale length. Tools also carry MCP annotations (read-only / destructive / idempotent) and the server sends usage instructions.
- CLI: `mcp-decision-lab demo | list | report ID | serve | --version`. With no arguments it still serves MCP over stdio. `python -m mcp_decision_lab` works too.
- A store format version (v2) with transparent migration of 0.1.x stores, per-decision `schema_version` and `updated_at`, and the analyzed winner and verdict shown in `list_decisions`.
- Protocol-level tests with a real MCP client (in-process and over stdio) on both SDK lines, a multi-process stress test, and a test that keeps the README example in sync with the code. The suite grew from 18 to 101 tests.

### Changed

- `analyze` now returns `winner: null` for ties (previously the first option listed).
- The `robustness` verdict now also checks single scores (a change of 1 point or less must not flip the winner) and, when simulations run, requires a Monte Carlo win probability of at least 80%. So a result that used to be `"robust"` can now be `"fragile"`, with the reason in `robustness_checks`.
- `strengths_weaknesses[option]` gained a `uniform` flag. For uniform options `best_criterion` and `worst_criterion` are `null`.
- The console script entry point moved to `mcp_decision_lab.cli:main`. `mcp_decision_lab.server:main` still works.
- The README now documents the install paths that actually work (from GitHub via `uvx --from git+…` or `pip install git+…`). The package is not on PyPI yet, so the PyPI badges were removed.

## [0.1.0] — 2026-07-26

### Added

- Five MCP tools: `start_decision`, `score_option`, `get_matrix`, `analyze` and `list_decisions`.
- Weighted decision matrices: criterion weights are normalized to sum to 1.0, and criteria where a high raw score is bad (cost, risk) can be declared `higher_is_better: false` to invert the effective score.
- Mandatory rationale (>= 10 characters) on every scored cell, so the final recommendation is auditable rather than a gut call.
- Exact closed-form sensitivity analysis: for each criterion it solves `gap(w') = w'·d + (1−w')·r = 0` to report the precise weight at which the winner would flip, the direction, and the new winner — no brute-force sweeps.
- Robustness verdict (`robust` / `fragile`) based on whether any ±50% relative change to a single criterion weight flips the winner, plus per-option strengths and weaknesses and a textual recommendation.
- JSON persistence of every session in `~/.mcp-decision-lab/decisions.json` (override with `DECISION_LAB_DIR`), with atomic single-file writes and stale-analysis invalidation on re-score.
- 18 tests covering weight normalization, input validation, hand-computed sensitivity math, score inversion, persistence round-trips and the status lifecycle.
