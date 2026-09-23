"""CLI (demo / list / report / --version) and decision-record export."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_decision_lab import __version__, cli, demo
from mcp_decision_lab.core import DecisionLab

ROOT = Path(__file__).resolve().parents[1]


def run_cli(capsys, *argv):
    with pytest.raises(SystemExit) as info:
        cli.main(list(argv))
    out, err = capsys.readouterr()
    return info.value.code, out, err


# ----------------------------------------------------------------------
# demo: the README example, pinned
# ----------------------------------------------------------------------


def test_demo_reproduces_the_readme_numbers(capsys):
    code, out, _ = run_cli(capsys, "demo", "--format", "json")
    assert code == 0
    a = json.loads(out)
    assert [(r["option"], r["total"]) for r in a["ranking"]] == [
        ("Postgres", 7.1),
        ("DynamoDB", 5.8),
        ("MongoDB", 5.6),
    ]
    assert a["winner"] == "Postgres" and a["margin"] == 1.3
    flips = {e["criterion"]: e["flip"] for e in a["sensitivity"]}
    assert flips["cost"]["flip_weight"] == 0.1176 and flips["cost"]["direction"] == "decrease"
    assert flips["scalability"]["flip_weight"] == 0.4717
    assert flips["scalability"]["direction"] == "increase"
    assert flips["team-familiarity"] is None
    assert a["robustness"] == "robust"
    hinge = a["score_sensitivity"]["most_fragile"][0]
    assert (hinge["option"], hinge["criterion"], hinge["flip_score"]) == ("Postgres", "cost", 5.6)
    assert a["monte_carlo"]["options"]["Postgres"]["win_probability"] == 0.9285


def test_demo_text_output(capsys):
    code, out, _ = run_cli(capsys, "demo")
    assert code == 0
    assert "1. Postgres  7.1" in out
    assert "cost              0.5    -> DynamoDB wins below 0.1176" in out
    assert "Postgres 92.8%" in out
    assert "Choose Postgres" in out


def test_demo_markdown_output_has_every_rationale(capsys):
    code, out, _ = run_cli(capsys, "demo", "--format", "markdown")
    assert code == 0
    assert out.startswith("# Decision record: " + demo.QUESTION)
    for cells in demo.SCORES.values():
        for _, rationale in cells.values():
            assert rationale in out


# ----------------------------------------------------------------------
# list / report on a real data directory
# ----------------------------------------------------------------------


@pytest.fixture()
def populated(tmp_path):
    lab = DecisionLab(tmp_path)
    did = demo.build(lab)
    lab.analyze(did)
    pending = lab.start_decision(
        "Half-done?", ["A", "B"], [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]
    )["decision_id"]
    return tmp_path, did, pending


def test_list(capsys, populated):
    data, did, pending = populated
    code, out, _ = run_cli(capsys, "list", "--data-dir", str(data))
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["ID", "STATUS", "CELLS", "RESULT", "QUESTION"]
    assert any(did in ln and "analyzed" in ln and "Postgres (robust)" in ln for ln in lines)
    assert any(pending in ln and "pending" in ln and "0/4" in ln for ln in lines)

    code, out, _ = run_cli(capsys, "list", "--data-dir", str(data), "--json")
    assert json.loads(out)["count"] == 2


def test_report_markdown_and_json(capsys, populated, tmp_path):
    data, did, pending = populated
    code, out, _ = run_cli(capsys, "report", did, "--data-dir", str(data))
    assert code == 0
    for cells in demo.SCORES.values():
        for _, rationale in cells.values():
            assert rationale in out
    assert "| cost | 0.5 | 0.1176 | decrease | DynamoDB | no |" in out
    assert "Verdict: **robust**" in out

    code, out, _ = run_cli(capsys, "report", did, "--data-dir", str(data), "--format", "json")
    doc = json.loads(out)
    assert doc["generated_by"] == f"mcp-decision-lab {__version__}"
    assert doc["analysis"]["winner"] == "Postgres"
    assert doc["decision"]["scores"]["Postgres"]["cost"]["score"] == 3

    target = tmp_path / "out" / "record.md"
    target.parent.mkdir()
    code, out, _ = run_cli(capsys, "report", did, "--data-dir", str(data), "-o", str(target))
    assert code == 0 and target.read_text(encoding="utf-8").startswith("# Decision record:")

    # A pending decision still exports, without an analysis.
    code, out, _ = run_cli(capsys, "report", pending, "--data-dir", str(data))
    assert "**pending** — 4 cell(s) still unscored" in out
    assert "Not available until every cell is scored." in out


def test_report_does_not_change_the_session(populated):
    data, did, _ = populated
    before = (data / "decisions.json").read_bytes()
    DecisionLab(data).export_decision(did, "markdown")
    assert (data / "decisions.json").read_bytes() == before


def test_markdown_escapes_table_breaking_text(tmp_path):
    lab = DecisionLab(tmp_path)
    did = lab.start_decision(
        "Pipes | and\nnewlines?",
        ["A|1", "B"],
        [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}],
    )["decision_id"]
    for o in ("A|1", "B"):
        for c in ("x", "y"):
            lab.score_option(did, o, c, 5 if o == "B" else 6, "line one\nline | two here")
    md = lab.export_decision(did)["content"]
    assert md.splitlines()[0] == "# Decision record: Pipes | and newlines?"
    assert "| A\\|1 | 6 | 6 |" in md
    assert "line one line \\| two here" in md


def test_unknown_id_is_a_clean_error(capsys, tmp_path):
    code, out, err = run_cli(capsys, "report", "dec-nope", "--data-dir", str(tmp_path))
    assert code == 2
    assert err.startswith("error: Decision 'dec-nope' not found")


def test_export_rejects_unknown_format(populated):
    data, did, _ = populated
    with pytest.raises(ValueError, match="format must be"):
        DecisionLab(data).export_decision(did, "pdf")


def test_version(capsys):
    code, out, _ = run_cli(capsys, "--version")
    assert code == 0 and out.startswith(f"mcp-decision-lab {__version__} (MCP SDK ")


def test_versions_agree():
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    assert manifest["version"] == __version__
    assert all(p["version"] == __version__ for p in manifest["packages"])


def test_module_entry_point_runs_the_demo():
    """``python -m mcp_decision_lab demo`` in a real process (no MCP client needed)."""
    out = subprocess.run(
        [sys.executable, "-m", "mcp_decision_lab", "demo", "--simulations", "0"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "1. Postgres  7.1" in out.stdout
    assert "Monte Carlo" not in out.stdout
