"""The PPY language server (spec 28.2, 28.3).

Inference is displayed as ghost annotations; only an explicit code action
rewrites a `.ppy` file.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse
from urllib.request import pathname2url

from ..diagnostics import Diagnostic, Severity
from .protocol import Message, read_messages, write_message
from .service import AnalysisService, Position

__all__ = ["LanguageServer", "path_to_uri", "serve", "uri_to_path"]

_SEVERITY = {
    Severity.ERROR: 1,
    Severity.WARNING: 2,
    Severity.NOTE: 3,
    Severity.REMARK: 4,
}


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def path_to_uri(path: Path) -> str:
    return "file://" + pathname2url(str(path.resolve()))


class LanguageServer:
    def __init__(self, service: AnalysisService, out: BinaryIO) -> None:
        self.service = service
        self.out = out
        self.shutdown_requested = False
        self.running = True
        self.show_remarks = False

    def handle(self, message: Message) -> None:
        method = message.method
        if method is None:
            return
        handler = getattr(self, "on_" + method.replace("/", "_").replace("$", "dollar"), None)
        if handler is None:
            if message.is_request:
                self._respond(message.id, None)
            return
        try:
            result = handler(message.params)
        except Exception:  # noqa: BLE001 - a server must not die on one request
            traceback.print_exc(file=sys.stderr)
            if message.is_request:
                self._error(message.id, -32603, "internal error")
            return
        if message.is_request:
            self._respond(message.id, result)

    def _respond(self, identifier: object, result: Any) -> None:
        write_message(self.out, {"jsonrpc": "2.0", "id": identifier, "result": result})

    def _error(self, identifier: object, code: int, message: str) -> None:
        write_message(
            self.out,
            {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}},
        )

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        write_message(self.out, {"jsonrpc": "2.0", "method": method, "params": params})

    def on_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        root = params.get("rootPath")
        folders = params.get("workspaceFolders") or []
        if root is None and folders:
            root = uri_to_path(folders[0]["uri"])
        if root is not None:
            self.service.root = Path(root)
        options = params.get("initializationOptions") or {}
        self.show_remarks = bool(options.get("optimizationRemarks", False))
        return {
            "capabilities": {
                "textDocumentSync": {"openClose": True, "change": 1, "save": True},
                "hoverProvider": True,
                "definitionProvider": True,
                "referencesProvider": True,
                "renameProvider": {"prepareProvider": True},
                "documentSymbolProvider": True,
                "inlayHintProvider": True,
                "codeActionProvider": True,
                "documentHighlightProvider": True,
            },
            "serverInfo": {"name": "ppy", "version": "0.1.0"},
        }

    def on_initialized(self, params: dict[str, Any]) -> None:
        return None

    def on_shutdown(self, params: dict[str, Any]) -> None:
        self.shutdown_requested = True

    def on_exit(self, params: dict[str, Any]) -> None:
        self.running = False

    def on_textDocument_didOpen(self, params: dict[str, Any]) -> None:
        document = params["textDocument"]
        path = uri_to_path(document["uri"])
        self.service.open(path, document.get("text", ""), document.get("version", 0))
        self._publish(path)

    def on_textDocument_didChange(self, params: dict[str, Any]) -> None:
        document = params["textDocument"]
        path = uri_to_path(document["uri"])
        changes = params.get("contentChanges") or []
        if not changes:
            return
        # The server advertises full-document sync, so the last change is the text.
        self.service.change(path, changes[-1].get("text", ""), document.get("version", 0))
        self._publish(path)

    def on_textDocument_didSave(self, params: dict[str, Any]) -> None:
        path = uri_to_path(params["textDocument"]["uri"])
        self.service.invalidate()
        self._publish(path)

    def on_textDocument_didClose(self, params: dict[str, Any]) -> None:
        path = uri_to_path(params["textDocument"]["uri"])
        self.service.close(path)
        self._notify(
            "textDocument/publishDiagnostics",
            {"uri": path_to_uri(path), "diagnostics": []},
        )

    def on_textDocument_hover(self, params: dict[str, Any]) -> dict[str, Any] | None:
        path, position = self._locate(params)
        detail = self.service.hover(path, position)
        if detail is None:
            return None
        return {"contents": {"kind": "markdown", "value": detail}}

    def on_textDocument_definition(self, params: dict[str, Any]) -> dict[str, Any] | None:
        path, position = self._locate(params)
        found = self.service.definition(path, position)
        return None if found is None else self._location(found)

    def on_textDocument_references(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        path, position = self._locate(params)
        return [self._location(found) for found in self.service.references(path, position)]

    def on_textDocument_documentHighlight(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        path, position = self._locate(params)
        return [
            {"range": self._range(found)}
            for found in self.service.references(path, position)
            if found.path.resolve() == path.resolve()
        ]

    def on_textDocument_prepareRename(self, params: dict[str, Any]) -> dict[str, Any] | None:
        path, position = self._locate(params)
        found = self.service.rename(path, position, "_ppy_probe")
        if not found:
            return None
        for location in found:
            if location.path.resolve() == path.resolve() and location.line == position.line:
                return {"range": self._range(location)}
        return {"range": self._range(found[0])}

    def on_textDocument_rename(self, params: dict[str, Any]) -> dict[str, Any] | None:
        path, position = self._locate(params)
        new_name = params.get("newName", "")
        found = self.service.rename(path, position, new_name)
        if not found:
            return None
        changes: dict[str, list[dict[str, Any]]] = {}
        for location in found:
            changes.setdefault(path_to_uri(location.path), []).append(
                {"range": self._range(location), "newText": new_name}
            )
        return {"changes": changes}

    def on_textDocument_documentSymbol(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        path = uri_to_path(params["textDocument"]["uri"])
        kinds = {"function": 12, "class": 5, "method": 6}
        return [
            {
                "name": name,
                "kind": kinds.get(kind, 13),
                "range": self._range(located),
                "selectionRange": self._range(located),
            }
            for name, kind, located in self.service.symbols(path)
        ]

    def on_textDocument_inlayHint(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        path = uri_to_path(params["textDocument"]["uri"])
        return [
            {
                "position": {"line": hint.line - 1, "character": hint.column},
                "label": hint.label,
                "kind": 1,
                "paddingLeft": False,
            }
            for hint in self.service.inlay_hints(path)
        ]

    def on_textDocument_codeAction(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        path = uri_to_path(params["textDocument"]["uri"])
        line = params.get("range", {}).get("start", {}).get("line", 0) + 1
        actions = []
        for action in self.service.code_actions(path, line):
            edits = [
                {
                    "range": {
                        "start": {"line": start_line - 1, "character": start_column},
                        "end": {"line": end_line - 1, "character": end_column},
                    },
                    "newText": text,
                }
                for start_line, start_column, end_line, end_column, text in action.edits
            ]
            actions.append(
                {
                    "title": action.title,
                    "kind": "refactor.rewrite",
                    "edit": {"changes": {path_to_uri(action.path): edits}},
                }
            )
        return actions

    def _publish(self, path: Path) -> None:
        found = list(self.service.diagnostics(path))
        if self.show_remarks:
            found.extend(self.service.remarks(path))
        self._notify(
            "textDocument/publishDiagnostics",
            {
                "uri": path_to_uri(path),
                "diagnostics": [self._diagnostic(d) for d in found],
            },
        )

    def _diagnostic(self, diagnostic: Diagnostic) -> dict[str, Any]:
        span = diagnostic.span
        line = (span.line - 1) if span else 0
        column = span.column if span else 0
        end_line = ((span.end_line or span.line) - 1) if span else 0
        end_column = span.end_column if span and span.end_column is not None else column + 1
        message = diagnostic.message
        if diagnostic.help:
            message += f"\nhelp: {diagnostic.help}"
        for note in diagnostic.notes:
            message += f"\nnote: {note}"
        return {
            "range": {
                "start": {"line": line, "character": column},
                "end": {"line": end_line, "character": end_column},
            },
            "severity": _SEVERITY[diagnostic.severity],
            "code": diagnostic.code,
            "source": "ppy",
            "message": message,
        }

    def _locate(self, params: dict[str, Any]) -> tuple[Path, Position]:
        path = uri_to_path(params["textDocument"]["uri"])
        position = params.get("position", {})
        return path, Position(position.get("line", 0) + 1, position.get("character", 0))

    def _range(self, located) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        return {
            "start": {"line": located.line - 1, "character": located.column},
            "end": {"line": located.end_line - 1, "character": located.end_column},
        }

    def _location(self, located) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        return {"uri": path_to_uri(located.path), "range": self._range(located)}


def serve(root: Path, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    """Run the language server over a byte stream until the client exits."""
    source = stdin or sys.stdin.buffer
    sink = stdout or sys.stdout.buffer
    server = LanguageServer(AnalysisService(root=root), sink)
    for message in read_messages(source):
        server.handle(message)
        if not server.running:
            break
    return 0 if server.shutdown_requested or not server.running else 1
