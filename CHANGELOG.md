# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-26

### Added

- Five MCP tools: `start_decision`, `score_option`, `get_matrix`, `analyze` and `list_decisions`.
- Weighted decision matrices: criterion weights are normalized to sum to 1.0, and criteria where a high raw score is bad (cost, risk) can be declared `higher_is_better: false` to invert the effective score.
- Mandatory rationale (>= 10 characters) on every scored cell, so the final recommendation is auditable rather than a gut call.
- Exact closed-form sensitivity analysis: for each criterion it solves `gap(w') = w'·d + (1−w')·r = 0` to report the precise weight at which the winner would flip, the direction, and the new winner — no brute-force sweeps.
- Robustness verdict (`robust` / `fragile`) based on whether any ±50% relative change to a single criterion weight flips the winner, plus per-option strengths and weaknesses and a textual recommendation.
- JSON persistence of every session in `~/.mcp-decision-lab/decisions.json` (override with `DECISION_LAB_DIR`), with atomic writes and stale-analysis invalidation on re-score.
- 18 tests covering weight normalization, input validation, hand-computed sensitivity math, score inversion, persistence round-trips and the status lifecycle.

[0.1.0]: https://github.com/AleBrito124356/mcp-decision-lab/releases/tag/v0.1.0
