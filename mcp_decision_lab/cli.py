"""Command line for mcp-decision-lab.

With no arguments it serves MCP over stdio, which is how MCP clients launch
it. The subcommands work offline without an MCP client::

    mcp-decision-lab                 # serve MCP over stdio (default)
    mcp-decision-lab serve           # same, explicit
    mcp-decision-lab demo            # run the README example in a temp store
    mcp-decision-lab list            # sessions in the data directory
    mcp-decision-lab report ID       # Markdown (or --format json) decision record
    mcp-decision-lab --version
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from pathlib import Path

from . import __version__


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("mcp")
    except Exception:  # pragma: no cover - mcp missing or not installed as a dist
        return "not installed"


def _utf8_stdout() -> None:
    """Rationales can hold any character; don't crash on a cp1252 pipe."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and not stream.isatty():
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover
                pass


def _lab(data_dir: str | None):
    from .core import DecisionLab

    return DecisionLab(data_dir)


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


def _render_analysis(a: dict, criteria: list[dict]) -> str:
    lines = [f"Question: {a['question']}"]
    weights = " · ".join(
        f"{c['name']} {c['weight']:g}" + ("" if c["higher_is_better"] else " (lower is better)")
        for c in criteria
    )
    lines += [f"Weights:  {weights}", "", "Ranking"]
    width = max(len(r["option"]) for r in a["ranking"])
    for r in a["ranking"]:
        lines.append(f"  {r['rank']}. {r['option']:<{width}}  {r['total']:g}")
    if a["winner"]:
        lines.append(f"Margin: {a['margin']:g}    Verdict: {a['robustness']}")
    else:
        lines.append(f"Verdict: tie between {', '.join(a['tied'])}")
    lines += ["", "Weight sensitivity (exact flip weights)"]
    cw = max(len(e["criterion"]) for e in a["sensitivity"])
    for e in a["sensitivity"]:
        f = e.get("flip")
        if f:
            side = "below" if f["direction"] == "decrease" else "above"
            what = f"{f['new_winner']} wins {side} {f['flip_weight']:g}"
        elif e.get("tie_break"):
            tb = e["tie_break"]
            what = (
                f"more weight favors {', '.join(tb['raise_weight_favors'])}, "
                f"less favors {', '.join(tb['lower_weight_favors'])}"
            )
        else:
            what = "never changes the winner"
        lines.append(f"  {e['criterion']:<{cw}}  {e['weight']:<6g} -> {what}")
    cells = a["score_sensitivity"]["most_fragile"]
    if cells:
        lines += ["", "Most fragile single scores"]
        for c in cells[:5]:
            lines.append(
                f"  {c['option']} / {c['criterion']} = {c['current_score']:g}: "
                f"{c['direction']} {c['flip_score']:g} -> {c['new_winner']} wins "
                f"({c['change_needed']:g} pts away)"
            )
    mc = a.get("monte_carlo")
    if mc:
        probs = " · ".join(
            f"{o} {s['win_probability'] * 100:.1f}%"
            for o, s in sorted(mc["options"].items(), key=lambda kv: -kv[1]["win_probability"])
        )
        lines += [
            "",
            f"Monte Carlo ({mc['simulations']} joint perturbations, seed {mc['seed']}): {probs}",
        ]
    lines += ["", "Recommendation", _wrap(a["recommendation"])]
    return "\n".join(lines)


def cmd_demo(args: argparse.Namespace) -> int:
    from . import demo

    with tempfile.TemporaryDirectory(prefix="decision-lab-demo-") as tmp:
        lab = _lab(tmp)
        did = demo.build(lab)
        analysis = lab.analyze(did, simulations=args.simulations)
        if args.format == "json":
            print(json.dumps(analysis, indent=2, ensure_ascii=False))
        elif args.format == "markdown":
            print(lab.export_decision(did, "markdown")["content"])
        else:
            print("mcp-decision-lab demo: the README example, computed live in a temporary store\n")
            print(_render_analysis(analysis, lab.get_matrix(did)["criteria"]))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    listing = _lab(args.data_dir).list_decisions()
    if args.json:
        print(json.dumps(listing, indent=2, ensure_ascii=False))
        return 0
    if not listing["decisions"]:
        print("No decisions yet.")
        return 0
    rows = []
    for d in listing["decisions"]:
        res = d.get("result")
        if res:
            outcome = res["winner"] or "tie: " + ", ".join(res["tied"])
            outcome += f" ({res['robustness']})"
        else:
            outcome = "-"
        rows.append(
            (d["decision_id"], d["status"], f"{d['cells_scored']}/{d['cells_total']}", outcome, d["question"])
        )
    head = ("ID", "STATUS", "CELLS", "RESULT", "QUESTION")
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths) + "  {}"
    print(fmt.format(*head))
    for r in rows:
        print(fmt.format(*r))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    fmt = "json" if args.format == "json" else "markdown"
    out = _lab(args.data_dir).export_decision(args.decision_id, fmt)["content"]
    if args.output:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(out)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server

    server.run(args.data_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-decision-lab",
        description="Weighted decision matrices with sensitivity analysis. "
        "With no command, serves MCP over stdio.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mcp-decision-lab {__version__} (MCP SDK {_sdk_version()})",
    )
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument(
        "--data-dir",
        default=None,
        help="store directory (default: $DECISION_LAB_DIR or ~/.mcp-decision-lab)",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("serve", parents=[data], help="serve MCP over stdio (the default)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="run the README database example offline")
    p.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    p.add_argument("--simulations", type=int, default=2000, help="Monte Carlo samples (0 = off)")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("list", parents=[data], help="list decision sessions")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("report", parents=[data], help="export a decision record")
    p.add_argument("decision_id")
    p.add_argument("--format", choices=["md", "markdown", "json"], default="md")
    p.add_argument("-o", "--output", help="write to this file instead of stdout")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in (None, "serve"):
        if args.command is None:
            args.data_dir = None
        cmd_serve(args)  # stdout belongs to the MCP protocol from here on
        return
    _utf8_stdout()
    try:
        code = args.func(args)
    except ValueError as exc:  # friendly errors (unknown id, corrupt store...)
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    main()
