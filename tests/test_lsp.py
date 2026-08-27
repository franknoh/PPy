"""The analysis daemon and language server (spec 28)."""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.lsp.protocol import Message, encode, read_messages
from ppy_compiler.lsp.server import LanguageServer, path_to_uri, serve, uri_to_path
from ppy_compiler.lsp.service import AnalysisService, Position

SOURCE = """
import ppy


@ppy.pure
def square(x: ppy.i64) -> ppy.i64:
    return x * x


def double(value):
    return value * 2


def use() -> int:
    return double(21)


def shout(text: str) -> str:
    print(text)
    return text
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "demo.ppy").write_text(textwrap.dedent(SOURCE).lstrip("\n"), encoding="utf-8")
    return tmp_path


@pytest.fixture
def service(workspace: Path) -> AnalysisService:
    return AnalysisService(root=workspace)


def _path(workspace: Path) -> Path:
    return workspace / "demo.ppy"


# -- protocol -------------------------------------------------------------


def test_messages_round_trip_through_the_framing():
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"a": 1}}
    stream = io.BytesIO(encode(payload) + encode({"jsonrpc": "2.0", "method": "exit"}))
    messages = list(read_messages(stream))
    assert len(messages) == 2
    assert messages[0].method == "initialize" and messages[0].is_request
    assert messages[0].params == {"a": 1}
    assert not messages[1].is_request


def test_a_truncated_stream_ends_cleanly():
    assert list(read_messages(io.BytesIO(b"Content-Length: 50\r\n\r\n{}"))) == []
    assert list(read_messages(io.BytesIO(b""))) == []


def test_uri_and_path_round_trip(tmp_path: Path):
    target = tmp_path / "some file.ppy"
    target.write_text("", encoding="utf-8")
    assert uri_to_path(path_to_uri(target)) == target.resolve()


# -- service --------------------------------------------------------------


def test_diagnostics_are_scoped_to_one_document(service: AnalysisService, workspace: Path):
    codes = [d.code for d in service.diagnostics(_path(workspace))]
    assert "E1201" in codes


def test_an_overlay_is_analyzed_before_the_saved_file(service: AnalysisService, workspace: Path):
    path = _path(workspace)
    assert "E1201" in [d.code for d in service.diagnostics(path)]

    service.open(path, "def clean(x: int) -> int:\n    return x\n", version=1)
    assert [d.code for d in service.diagnostics(path)] == []
    assert path.read_text(encoding="utf-8").startswith("import ppy"), "the file itself is untouched"

    service.close(path)
    assert "E1201" in [d.code for d in service.diagnostics(path)]


def test_analysis_is_reused_until_a_document_changes(service: AnalysisService, workspace: Path):
    first = service.bundle()
    assert service.bundle() is first

    service.open(_path(workspace), "x: int = 1\n", version=1)
    assert service.bundle() is not first


def test_hover_reports_type_effects_and_optimization(service: AnalysisService, workspace: Path):
    detail = service.hover(_path(workspace), Position(5, 4))
    assert detail is not None
    assert "square" in detail
    assert "effects:" in detail
    assert "purity: verified pure" in detail
    assert "llvm: native" in detail


def test_hover_reports_why_a_function_is_not_native(service: AnalysisService, workspace: Path):
    detail = service.hover(_path(workspace), Position(17, 4))
    assert detail is not None
    assert "llvm: boxed" in detail
    assert "IO" in detail


def test_hover_shows_an_inferred_expression_type(service: AnalysisService, workspace: Path):
    detail = service.hover(_path(workspace), Position(6, 11))
    assert detail is not None
    assert "`int`" in detail
    assert "representation:" in detail


def test_definition_finds_a_project_function(service: AnalysisService, workspace: Path):
    found = service.definition(_path(workspace), Position(14, 11))
    assert found is not None
    assert found.path == _path(workspace)
    assert found.line == 9


def test_definition_finds_a_parameter(service: AnalysisService, workspace: Path):
    found = service.definition(_path(workspace), Position(6, 11))
    assert found is not None and found.line == 5


def test_references_finds_every_use(service: AnalysisService, workspace: Path):
    found = service.references(_path(workspace), Position(14, 11))
    assert len(found) >= 2
    assert {location.line for location in found} >= {9, 14}


def test_renaming_a_local_is_allowed(service: AnalysisService, workspace: Path):
    found = service.rename(_path(workspace), Position(10, 11), "amount")
    assert found is not None
    assert all(location.path == _path(workspace) for location in found)


def test_renaming_to_an_invalid_identifier_is_refused(service: AnalysisService, workspace: Path):
    assert service.rename(_path(workspace), Position(10, 11), "not an identifier") is None


def test_inlay_hints_show_inferred_types_without_editing(service: AnalysisService, workspace: Path):
    path = _path(workspace)
    before = path.read_text(encoding="utf-8")
    hints = service.inlay_hints(path)

    labels = {hint.label for hint in hints}
    assert any("int" in label for label in labels)
    assert path.read_text(encoding="utf-8") == before


def test_a_code_action_writes_what_the_hint_showed(service: AnalysisService, workspace: Path):
    path = _path(workspace)
    hints = [hint for hint in service.inlay_hints(path) if ":" in hint.label]
    assert hints
    actions = service.code_actions(path, hints[0].line)
    assert actions and actions[0].title.startswith("Insert inferred")
    assert actions[0].edits


def test_dynamic_regions_are_reported(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    target = tmp_path / "dyn.ppy"
    target.write_text(
        textwrap.dedent(
            """
            import ppy


            def f() -> None:
                with ppy.dynamic:
                    eval("1")
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    service = AnalysisService(root=tmp_path)
    spans = service.dynamic_regions(target)
    assert spans and spans[0][0] == 5


def test_remarks_describe_optimization_decisions(service: AnalysisService, workspace: Path):
    messages = [d.message for d in service.remarks(_path(workspace))]
    assert any("lowers to native code" in m for m in messages)


def test_symbols_list_functions_and_classes(service: AnalysisService, workspace: Path):
    names = {name for name, _kind, _located in service.symbols(_path(workspace))}
    assert {"square", "double", "use", "shout"} <= names


# -- server ---------------------------------------------------------------


def _session(root: Path, requests: list[dict]) -> list[dict]:
    stdin = io.BytesIO(b"".join(encode(message) for message in requests))
    stdout = io.BytesIO()
    serve(root, stdin, stdout)
    return [
        json.loads(chunk.split("\r\n\r\n", 1)[1])
        for chunk in stdout.getvalue().decode().split("Content-Length: ")
        if chunk.strip()
    ]


def _open(uri: str, text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {
            "textDocument": {"uri": uri, "languageId": "python", "version": 1, "text": text}
        },
    }


def test_the_server_advertises_its_capabilities(workspace: Path):
    replies = _session(
        workspace,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"rootPath": str(workspace)}},
            {"jsonrpc": "2.0", "method": "exit"},
        ],
    )
    capabilities = replies[0]["result"]["capabilities"]
    for feature in (
        "hoverProvider", "definitionProvider", "referencesProvider",
        "renameProvider", "documentSymbolProvider", "inlayHintProvider",
        "codeActionProvider",
    ):
        assert capabilities[feature]


def test_the_server_publishes_diagnostics_on_open(workspace: Path):
    path = _path(workspace)
    uri = path_to_uri(path)
    replies = _session(
        workspace,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"rootPath": str(workspace)}},
            _open(uri, path.read_text(encoding="utf-8")),
            {"jsonrpc": "2.0", "method": "exit"},
        ],
    )
    published = [r for r in replies if r.get("method") == "textDocument/publishDiagnostics"]
    assert published
    codes = [d["code"] for d in published[0]["params"]["diagnostics"]]
    assert "E1201" in codes
    entry = next(d for d in published[0]["params"]["diagnostics"] if d["code"] == "E1201")
    assert entry["severity"] == 1
    assert entry["source"] == "ppy"
    assert "help:" in entry["message"]


def test_editing_a_buffer_updates_diagnostics(workspace: Path):
    path = _path(workspace)
    uri = path_to_uri(path)
    replies = _session(
        workspace,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"rootPath": str(workspace)}},
            _open(uri, path.read_text(encoding="utf-8")),
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": uri, "version": 2},
                    "contentChanges": [{"text": "def ok(x: int) -> int:\n    return x\n"}],
                },
            },
            {"jsonrpc": "2.0", "method": "exit"},
        ],
    )
    published = [r for r in replies if r.get("method") == "textDocument/publishDiagnostics"]
    assert len(published) == 2
    assert published[1]["params"]["diagnostics"] == []


def test_the_server_answers_hover_and_survives_a_bad_request(workspace: Path):
    path = _path(workspace)
    uri = path_to_uri(path)
    replies = _session(
        workspace,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"rootPath": str(workspace)}},
            _open(uri, path.read_text(encoding="utf-8")),
            {"jsonrpc": "2.0", "id": 2, "method": "textDocument/hover",
             "params": {"textDocument": {"uri": uri}, "position": {"line": 4, "character": 4}}},
            {"jsonrpc": "2.0", "id": 3, "method": "textDocument/nonsense", "params": {}},
            {"jsonrpc": "2.0", "id": 4, "method": "textDocument/documentSymbol",
             "params": {"textDocument": {"uri": uri}}},
            {"jsonrpc": "2.0", "id": 5, "method": "shutdown", "params": {}},
            {"jsonrpc": "2.0", "method": "exit"},
        ],
    )
    answers = {r["id"]: r for r in replies if "id" in r}
    assert "square" in answers[2]["result"]["contents"]["value"]
    assert answers[3]["result"] is None
    assert any(s["name"] == "square" for s in answers[4]["result"])


def test_the_server_reports_inlay_hints_and_code_actions(workspace: Path):
    path = _path(workspace)
    uri = path_to_uri(path)
    replies = _session(
        workspace,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"rootPath": str(workspace)}},
            _open(uri, path.read_text(encoding="utf-8")),
            {"jsonrpc": "2.0", "id": 2, "method": "textDocument/inlayHint",
             "params": {"textDocument": {"uri": uri}}},
            {"jsonrpc": "2.0", "id": 3, "method": "textDocument/codeAction",
             "params": {"textDocument": {"uri": uri},
                        "range": {"start": {"line": 8, "character": 0},
                                  "end": {"line": 8, "character": 0}}}},
            {"jsonrpc": "2.0", "method": "exit"},
        ],
    )
    answers = {r["id"]: r for r in replies if "id" in r}
    assert answers[2]["result"]
    actions = answers[3]["result"]
    assert actions and actions[0]["kind"] == "refactor.rewrite"
    assert actions[0]["edit"]["changes"]
