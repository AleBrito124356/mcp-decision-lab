"""Protocol-level tests: a real MCP client talks to the server.

Run under both SDK lines (mcp 2.x in-process ``Client``; mcp 1.x
``create_connected_server_and_client_session``), plus a stdio smoke test that
launches the server as a subprocess. Skipped when the ``mcp`` package is not
installed, so the core tests still run without it.
"""

import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")
anyio = pytest.importorskip("anyio")

from mcp_decision_lab import __version__, demo, server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    "start_decision",
    "score_option",
    "get_matrix",
    "analyze",
    "what_if",
    "export_decision",
    "delete_decision",
    "list_decisions",
}


class Session:
    """Normalizes the few API differences between SDK 1.x and 2.x clients."""

    def __init__(self, raw):
        self.raw = raw

    async def tools(self):
        return {t.name: t for t in (await self.raw.list_tools()).tools}

    async def call(self, name, args=None):
        res = await self.raw.call_tool(name, args or {})
        is_error = getattr(res, "is_error", None)
        if is_error is None:
            is_error = res.isError
        text = "\n".join(c.text for c in res.content if getattr(c, "text", None))
        if is_error:
            return True, text
        structured = getattr(res, "structured_content", None)
        if structured is None:
            structured = getattr(res, "structuredContent", None)
        return False, structured if structured is not None else json.loads(text)

    async def ok(self, name, args=None):
        err, out = await self.call(name, args)
        assert not err, out
        return out


def schema_of(tool):
    return getattr(tool, "input_schema", None) or tool.inputSchema


def annotations_of(tool):
    a = getattr(tool, "annotations", None)
    if a is None:  # SDKs older than 1.6 have no tool annotations at all
        return None
    get = lambda snake, camel: getattr(a, snake, None) if hasattr(a, snake) else getattr(a, camel)  # noqa: E731
    return {
        "read_only": get("read_only_hint", "readOnlyHint"),
        "destructive": get("destructive_hint", "destructiveHint"),
    }


@asynccontextmanager
async def in_process():
    if server.SDK_LINE == 2:
        from mcp import Client

        async with Client(server.mcp) as client:
            yield Session(client)
    else:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server.mcp._mcp_server) as s:
            yield Session(s)


@asynccontextmanager
async def over_stdio(data_dir, module):
    from mcp.client.stdio import StdioServerParameters, get_default_environment

    # Old 1.x SDKs use a custom env *instead of* the defaults (no SYSTEMROOT on
    # Windows -> the child cannot even open sockets), so always start from them.
    env = {**get_default_environment(), "DECISION_LAB_DIR": str(data_dir), "PYTHONPATH": str(ROOT)}
    params = StdioServerParameters(command=sys.executable, args=["-m", module], env=env)
    if server.SDK_LINE == 2:
        from mcp import Client

        async with Client(params) as client:
            yield Session(client)
    else:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                yield Session(s)


@pytest.fixture()
def data_dir(tmp_path):
    server.configure(tmp_path)
    yield tmp_path
    server.configure(None)


def run(coro_fn):
    anyio.run(coro_fn)


async def build_demo(s):
    did = (
        await s.ok(
            "start_decision",
            {"question": demo.QUESTION, "options": demo.OPTIONS, "criteria": demo.CRITERIA},
        )
    )["decision_id"]
    for option, cells in demo.SCORES.items():
        for criterion, (score, rationale) in cells.items():
            await s.ok(
                "score_option",
                {
                    "decision_id": did,
                    "option": option,
                    "criterion": criterion,
                    "score": score,
                    "rationale": rationale,
                },
            )
    return did


# ----------------------------------------------------------------------
# Tool listing and schemas
# ----------------------------------------------------------------------


def test_lists_all_tools_with_typed_schemas(data_dir):
    async def main():
        async with in_process() as s:
            tools = await s.tools()
            assert set(tools) == TOOLS

            start = schema_of(tools["start_decision"])
            assert start["properties"]["options"]["minItems"] == 2
            crit_items = start["properties"]["criteria"]["items"]
            assert start["properties"]["criteria"]["minItems"] == 2
            crit = start["$defs"][crit_items["$ref"].split("/")[-1]]
            assert set(crit["properties"]) == {"name", "weight", "higher_is_better"}
            assert crit["properties"]["weight"]["exclusiveMinimum"] == 0
            assert crit["properties"]["higher_is_better"]["default"] is True
            assert crit["additionalProperties"] is False
            assert set(crit["required"]) == {"name", "weight"}

            score = schema_of(tools["score_option"])["properties"]
            assert score["score"]["minimum"] == 0 and score["score"]["maximum"] == 10
            assert score["rationale"]["minLength"] == 10

            analyze = schema_of(tools["analyze"])["properties"]
            assert analyze["simulations"]["default"] == 2000
            assert analyze["simulations"]["maximum"] == 50_000

            export = schema_of(tools["export_decision"])["properties"]["format"]
            assert export["enum"] == ["markdown", "json"]

            if annotations_of(tools["delete_decision"]) is not None:
                assert annotations_of(tools["delete_decision"])["destructive"] is True
                assert annotations_of(tools["get_matrix"])["read_only"] is True
                assert annotations_of(tools["what_if"])["read_only"] is True
            else:
                assert server.ToolAnnotations is None  # only acceptable on a very old SDK

    run(main)


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


def test_full_decision_end_to_end(data_dir):
    async def main():
        async with in_process() as s:
            did = await build_demo(s)
            m = await s.ok("get_matrix", {"decision_id": did})
            assert m["status"] == "complete"

            a = await s.ok("analyze", {"decision_id": did})
            assert [r["total"] for r in a["ranking"]] == [7.1, 5.8, 5.6]
            assert a["winner"] == "Postgres" and a["margin"] == 1.3
            assert a["robustness"] == "robust"
            assert a["monte_carlo"]["options"]["Postgres"]["win_probability"] > 0.9

            flip = {e["criterion"]: e["flip"] for e in a["sensitivity"]}["scalability"]
            w = await s.ok(
                "what_if",
                {"decision_id": did, "weights": {"scalability": flip["flip_weight"] + 0.01}},
            )
            assert w["winner"] == "DynamoDB" and w["winner_changed"] is True
            w2 = await s.ok(
                "what_if",
                {
                    "decision_id": did,
                    "score_overrides": [{"option": "Postgres", "criterion": "cost", "score": 6}],
                },
            )
            assert w2["winner"] == "DynamoDB"
            assert await s.ok("get_matrix", {"decision_id": did}) == {**m, "status": "analyzed"}

            md = await s.ok("export_decision", {"decision_id": did})
            assert md["format"] == "markdown"
            for cells in demo.SCORES.values():
                for _, rationale in cells.values():
                    assert rationale in md["content"]
            js = await s.ok("export_decision", {"decision_id": did, "format": "json"})
            assert json.loads(js["content"])["analysis"]["winner"] == "Postgres"

            listing = await s.ok("list_decisions")
            assert listing["decisions"][0]["result"]["winner"] == "Postgres"

            gone = await s.ok("delete_decision", {"decision_id": did})
            assert gone["remaining_decisions"] == 0
            assert (await s.ok("list_decisions"))["count"] == 0

    run(main)


# ----------------------------------------------------------------------
# Error messages must reach the model
# ----------------------------------------------------------------------


def test_validation_messages_reach_the_client(data_dir):
    async def main():
        async with in_process() as s:
            did = (
                await s.ok(
                    "start_decision",
                    {
                        "question": "A or B?",
                        "options": ["A", "B"],
                        "criteria": [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}],
                    },
                )
            )["decision_id"]

            err, text = await s.call(
                "score_option",
                {"decision_id": did, "option": "C", "criterion": "x", "score": 5,
                 "rationale": "C does not exist here"},
            )
            assert err and "Unknown option 'C'. Valid options: A, B." in text

            # Passes the schema (>= 10 chars) but not the stripped check in core.
            err, text = await s.call(
                "score_option",
                {"decision_id": did, "option": "A", "criterion": "x", "score": 5,
                 "rationale": "   short     "},
            )
            assert err and "rationale is required" in text

            err, text = await s.call("analyze", {"decision_id": did})
            assert err and "Matrix incomplete — 4 cell(s) missing" in text

            err, text = await s.call("get_matrix", {"decision_id": "dec-nope"})
            assert err and f"Decision 'dec-nope' not found (known ids: {did})" in text

            err, text = await s.call(
                "start_decision",
                {"question": "q?", "options": ["A", "a"],
                 "criteria": [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}]},
            )
            assert err and "Duplicate option" in text

            # Schema-level rejections carry pydantic's explanation too.
            err, text = await s.call(
                "score_option",
                {"decision_id": did, "option": "A", "criterion": "x", "score": 11,
                 "rationale": "way too high a score"},
            )
            assert err and "less than or equal to 10" in text
            err, text = await s.call(
                "start_decision",
                {"question": "q?", "options": ["A", "B"],
                 "criteria": [{"name": "x", "weight": 1, "colour": "red"},
                              {"name": "y", "weight": 1}]},
            )
            assert err and "colour" in text

    run(main)


def test_corrupt_store_is_a_tool_error_not_a_dead_server(data_dir):
    store = data_dir / "decisions.json"
    store.write_text("[]", encoding="utf-8")

    async def main():
        async with in_process() as s:
            err, text = await s.call("list_decisions")
            assert err and "not a valid mcp-decision-lab store" in text
            assert "Fix or delete the file" in text
            store.unlink()  # the user fixes it; the same server recovers
            assert (await s.ok("list_decisions"))["count"] == 0

    run(main)


def test_importing_the_server_has_no_side_effects(tmp_path):
    """v0.1 built DecisionLab at import time, creating ~/.mcp-decision-lab."""
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "PYTHONPATH": str(ROOT)}
    env.pop("DECISION_LAB_DIR", None)
    code = "import mcp_decision_lab.server as s; assert s._lab is None"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert not (tmp_path / ".mcp-decision-lab").exists()


# ----------------------------------------------------------------------
# stdio: the way MCP clients actually launch it
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module", ["mcp_decision_lab.server", "mcp_decision_lab"])
def test_stdio_smoke(tmp_path, module):
    async def main():
        with anyio.fail_after(60):
            async with over_stdio(tmp_path, module) as s:
                assert set(await s.tools()) == TOOLS
                did = await build_demo(s)
                a = await s.ok("analyze", {"decision_id": did, "simulations": 500})
                assert a["winner"] == "Postgres" and a["margin"] == 1.3
                err, text = await s.call("get_matrix", {"decision_id": "dec-nope"})
                assert err and "not found" in text

    run(main)
    saved = json.loads((tmp_path / "decisions.json").read_text(encoding="utf-8"))
    assert len(saved["decisions"]) == 1


def test_server_reports_package_version():
    if server.SDK_LINE == 2:
        assert server.mcp.version == __version__
    assert server.mcp.name == "mcp-decision-lab"
