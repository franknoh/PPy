"""The shared analysis service behind the CLI and the language server (spec 28.1).

The service keeps a project's parsed modules, symbol and type graphs, plugin
state, and open-document overlays, so that an editor keystroke re-analyzes one
module instead of the whole project from scratch.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import types as T
from ..analysis.contracts import ContractReport
from ..analysis.representation import select
from ..analysis.symbols import ClassInfo, FunctionInfo
from ..diagnostics import Diagnostic, Severity
from ..driver.pipeline import AnalysisBundle, analyze_paths, collect_sources, open_project

__all__ = ["Action", "AnalysisService", "Hint", "Located", "Position"]


@dataclass(frozen=True, slots=True)
class Position:
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class Located:
    path: Path
    line: int
    column: int
    end_line: int
    end_column: int

    @staticmethod
    def of(path: Path, node: ast.AST) -> Located:
        return Located(
            path,
            getattr(node, "lineno", 1),
            getattr(node, "col_offset", 0),
            getattr(node, "end_lineno", None) or getattr(node, "lineno", 1),
            getattr(node, "end_col_offset", None) or getattr(node, "col_offset", 0),
        )


@dataclass(frozen=True, slots=True)
class Hint:
    """A ghost annotation shown without rewriting the source (spec 28.3)."""

    line: int
    column: int
    label: str
    insert: str


@dataclass(frozen=True, slots=True)
class Action:
    title: str
    path: Path
    edits: tuple[tuple[int, int, int, int, str], ...]


@dataclass(slots=True)
class Document:
    path: Path
    text: str
    version: int = 0


@dataclass(slots=True)
class AnalysisService:
    root: Path
    documents: dict[Path, Document] = field(default_factory=dict)
    _bundle: AnalysisBundle | None = None
    _stamp: tuple = ()
    _diagnostics: list[Diagnostic] = field(default_factory=list)

    def open(self, path: Path, text: str, version: int = 0) -> None:
        self.documents[path] = Document(path, text, version)
        self.invalidate()

    def change(self, path: Path, text: str, version: int) -> None:
        self.documents[path] = Document(path, text, version)
        self.invalidate()

    def close(self, path: Path) -> None:
        self.documents.pop(path, None)
        self.invalidate()

    def invalidate(self) -> None:
        self._bundle = None

    @property
    def overlays(self) -> dict[Path, str]:
        return {path: document.text for path, document in self.documents.items()}

    def bundle(self) -> AnalysisBundle:
        """Analyze the project, reusing the last result when nothing changed.

        Diagnostics are snapshotted from the strict pass before call-site
        evidence is folded in, so an editor still reports the missing
        annotation while hover and hints show what could be inferred.
        """
        from ..driver.convert import refine_with_call_sites

        stamp = tuple(sorted((str(p), d.version, len(d.text)) for p, d in self.documents.items()))
        if self._bundle is not None and stamp == self._stamp:
            return self._bundle
        project = open_project(self.root)
        entries = sorted(set(collect_sources(self.root, ppy_only=False)) | set(self.documents))
        bundle = analyze_paths(project, entries, backend="llvm", overlays=self.overlays)
        self._diagnostics = bundle.diagnostics.sorted()
        refine_with_call_sites(bundle)
        self._bundle = bundle
        self._stamp = stamp
        return bundle

    def diagnostics(self, path: Path) -> list[Diagnostic]:
        self.bundle()
        resolved = path.resolve()
        return [
            diagnostic
            for diagnostic in self._diagnostics
            if diagnostic.span is not None and diagnostic.span.path.resolve() == resolved
        ]

    def remarks(self, path: Path) -> list[Diagnostic]:
        """Optimization remarks for one document (spec 28.2, 29.2)."""
        bundle = self.bundle()
        module = self._module_named(path)
        if module is None:
            return []
        found: list[Diagnostic] = []
        from ..diagnostics import Span

        analysis = bundle.analysis.modules.get(module)
        if analysis is not None:
            found.extend(
                Diagnostic(
                    "R3001",
                    Severity.REMARK,
                    f"{note.qualname} -> {note.lowering}: {note.reason}",
                    Span(path, note.line, 0),
                )
                for note in analysis.lowerings.values()
            )
        for info in self._functions(path):
            report = bundle.reports.get(info.qualname)
            if report is None:
                continue
            if report.native_ok:
                found.append(
                    Diagnostic(
                        "R3001",
                        Severity.REMARK,
                        f"`{info.name}` lowers to native code",
                        Span(path, info.node.lineno, 0),
                    )
                )
            if report.parallel_ok:
                found.append(
                    Diagnostic(
                        "R3001",
                        Severity.REMARK,
                        f"`{info.name}` is parallelizable",
                        Span(path, info.node.lineno, 0),
                    )
                )
        return found

    def hover(self, path: Path, position: Position) -> str | None:
        """Inferred type, effects, and optimization status at a position."""
        bundle = self.bundle()
        module = self._module_named(path)
        if module is None:
            return None
        analysis = bundle.analysis.modules.get(module)
        symbols = bundle.symbols.modules.get(module)
        if analysis is None or symbols is None:
            return None

        node = self._node_at(symbols.module.tree, position)
        if node is None:
            return None

        info = self._enclosing_function(path, position)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found = self._function_named(path, node.name)
            if found is not None:
                return self._describe_function(bundle, found)

        if isinstance(node, ast.Name) and info is not None:
            for param in info.params:
                if param.name == node.id:
                    return self._describe_binding(
                        f"parameter `{param.name}`", param.type, param.facts
                    )

        if isinstance(node, ast.Name):
            target = self._function_named(path, node.id)
            if target is not None:
                return self._describe_function(bundle, target)
            cls = symbols.classes.get(node.id)
            if cls is not None:
                return self._describe_class(cls)

        node_type = analysis.type_of(node)
        if isinstance(node_type, T.UnknownType):
            return None
        detail = self._describe_binding(
            f"`{ast.unparse(node)}`", node_type, analysis.facts_of(node)
        )
        note = analysis.lowerings.get(id(node))
        if note is not None:
            detail += f"\n\nlowering: {note.lowering} — {note.reason}"
        return detail

    def definition(self, path: Path, position: Position) -> Located | None:
        bundle = self.bundle()
        module = self._module_named(path)
        symbols = bundle.symbols.modules.get(module or "")
        if symbols is None:
            return None
        node = self._node_at(symbols.module.tree, position)
        name = self._identifier(node)
        if name is None:
            return None

        info = self._enclosing_function(path, position)
        if info is not None:
            local = self._local_binding(info, name)
            if local is not None:
                return Located.of(info.path, local)
        target = bundle.symbols.functions.get(f"{module}.{name}") or self._imported(
            bundle, symbols, name
        )
        if target is not None:
            return Located.of(target.path, target.node)
        cls = symbols.classes.get(name)
        if cls is not None:
            return Located.of(cls.path, cls.node)
        return None

    def references(self, path: Path, position: Position) -> list[Located]:
        bundle = self.bundle()
        module = self._module_named(path)
        symbols = bundle.symbols.modules.get(module or "")
        if symbols is None:
            return []
        node = self._node_at(symbols.module.tree, position)
        name = self._identifier(node)
        if name is None:
            return []

        info = self._enclosing_function(path, position)
        if info is not None and self._local_binding(info, name) is not None:
            return [
                Located.of(info.path, child)
                for child in ast.walk(info.node)
                if isinstance(child, ast.Name) and child.id == name
            ]

        def mentions(child: ast.AST) -> bool:
            if isinstance(child, ast.Name):
                return child.id == name
            if isinstance(child, ast.Attribute):
                return child.attr == name
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return child.name == name
            return False

        return [
            Located.of(other.path, child)
            for other in bundle.symbols.modules.values()
            for child in ast.walk(other.module.tree)
            if mentions(child)
        ]

    def rename(self, path: Path, position: Position, new_name: str) -> list[Located] | None:
        """Locations to rewrite, or `None` when the rename is not provably safe."""
        if not new_name.isidentifier():
            return None
        bundle = self.bundle()
        module = self._module_named(path)
        symbols = bundle.symbols.modules.get(module or "")
        if symbols is None:
            return None
        node = self._node_at(symbols.module.tree, position)
        name = self._identifier(node)
        if name is None:
            return None

        info = self._enclosing_function(path, position)
        if info is not None and self._local_binding(info, name) is not None:
            # A local is safe: nothing outside the function can observe it.
            return [
                Located.of(info.path, child)
                for child in ast.walk(info.node)
                if isinstance(child, ast.Name) and child.id == name
            ]
        # A module-level name is renamable only when the whole project is visible
        # and no dynamic boundary could reach it by string.
        if any(
            m.functions and analysis.dynamic_spans
            for m, analysis in zip(
                bundle.symbols.modules.values(), bundle.analysis.modules.values(), strict=False
            )
        ):
            return None
        return self.references(path, position) or None

    def inlay_hints(self, path: Path) -> list[Hint]:
        """Inferred types shown as ghost annotations, never written to source."""
        self.bundle()
        hints: list[Hint] = []
        from ..analysis.render import render_annotation

        for info in self._functions(path):
            for param in info.params:
                if param.kind in {"var_positional", "var_keyword"}:
                    continue
                argument_node = self._argument_node(info, param.name)
                if argument_node is None or argument_node.annotation is not None:
                    continue
                rendered = render_annotation(param.type, param.facts, local_module=info.module)
                if rendered is None:
                    continue
                hints.append(
                    Hint(
                        argument_node.lineno,
                        argument_node.end_col_offset or 0,
                        f": {rendered.text}",
                        f": {rendered.text}",
                    )
                )
            if not info.ret_annotated:
                rendered = render_annotation(info.ret, info.ret_facts, local_module=info.module)
                if rendered is not None:
                    hints.append(
                        Hint(
                            info.node.lineno,
                            self._signature_end(info),
                            f" -> {rendered.text}",
                            f" -> {rendered.text}",
                        )
                    )
        return hints

    def code_actions(self, path: Path, line: int) -> list[Action]:
        """Offer to write the inferred annotations the hints display."""
        hints = [hint for hint in self.inlay_hints(path) if hint.line == line]
        if not hints:
            return []
        edits = tuple(
            (hint.line, hint.column, hint.line, hint.column, hint.insert) for hint in hints
        )
        return [Action("Insert inferred PPY annotations", path, edits)]

    def dynamic_regions(self, path: Path) -> list[tuple[int, int]]:
        """Line spans covered by an explicit `ppy.dynamic` boundary (spec 28.2)."""
        module = self._module_named(path)
        analysis = self.bundle().analysis.modules.get(module or "")
        return list(analysis.dynamic_spans) if analysis else []

    def symbols(self, path: Path) -> list[tuple[str, str, Located]]:
        bundle = self.bundle()
        module = self._module_named(path)
        symbols = bundle.symbols.modules.get(module or "")
        if symbols is None:
            return []
        found: list[tuple[str, str, Located]] = []
        for name, info in symbols.functions.items():
            found.append((name, "function", Located.of(info.path, info.node)))
        for name, cls in symbols.classes.items():
            found.append((name, "class", Located.of(cls.path, cls.node)))
            for method, info in cls.methods.items():
                found.append((f"{name}.{method}", "method", Located.of(info.path, info.node)))
        return found

    def _module_named(self, path: Path) -> str | None:
        resolved = path.resolve()
        for name, symbols in self.bundle().symbols.modules.items():
            if symbols.path.resolve() == resolved:
                return name
        return None

    def _functions(self, path: Path) -> list[FunctionInfo]:
        module = self._module_named(path)
        symbols = self.bundle().symbols.modules.get(module or "")
        if symbols is None:
            return []
        found = list(symbols.functions.values())
        for cls in symbols.classes.values():
            found.extend(cls.methods.values())
        return found

    def _function_named(self, path: Path, name: str) -> FunctionInfo | None:
        for info in self._functions(path):
            if info.name == name:
                return info
        return None

    def _enclosing_function(self, path: Path, position: Position) -> FunctionInfo | None:
        best: FunctionInfo | None = None
        for info in self._functions(path):
            start = info.node.lineno
            end = getattr(info.node, "end_lineno", start) or start
            if start <= position.line <= end and (best is None or start > best.node.lineno):
                best = info
        return best

    def _node_at(self, tree: ast.Module, position: Position) -> ast.AST | None:
        best: ast.AST | None = None
        best_size = None
        for node in ast.walk(tree):
            if not hasattr(node, "lineno"):
                continue
            start_line, start_col = node.lineno, node.col_offset
            end_line = getattr(node, "end_lineno", start_line) or start_line
            end_col = getattr(node, "end_col_offset", start_col) or start_col
            if position.line < start_line or position.line > end_line:
                continue
            if position.line == start_line and position.column < start_col:
                continue
            if position.line == end_line and position.column > end_col:
                continue
            size = (end_line - start_line, end_col - start_col)
            if best_size is None or size < best_size:
                best, best_size = node, size
        return best

    def _identifier(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.arg):
            return node.arg
        return None

    def _local_binding(self, info: FunctionInfo, name: str) -> ast.AST | None:
        for argument in [
            *info.node.args.posonlyargs,
            *info.node.args.args,
            *info.node.args.kwonlyargs,
        ]:
            if argument.arg == name:
                return argument
        for node in ast.walk(info.node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name:
                return node
        return None

    def _imported(self, bundle: AnalysisBundle, symbols, name: str) -> FunctionInfo | None:
        binding = symbols.imports.get(name)
        if binding is None:
            return None
        return bundle.symbols.functions.get(binding.canonical)

    def _argument_node(self, info: FunctionInfo, name: str) -> ast.arg | None:
        arguments = info.node.args
        for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
            for argument in group:
                if argument.arg == name:
                    return argument
        return None

    def _signature_end(self, info: FunctionInfo) -> int:
        """Column just past the parameter list, where `-> T` would go."""
        arguments = info.node.args
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        if arguments.vararg:
            every.append(arguments.vararg)
        if arguments.kwarg:
            every.append(arguments.kwarg)
        if not every:
            return info.node.col_offset + len("def ") + len(info.name) + 2
        last = max(every, key=lambda a: (a.lineno, a.end_col_offset or 0))
        return (last.end_col_offset or 0) + 1

    def _describe_binding(self, subject: str, node_type: T.Type, facts) -> str:
        lines = [f"{subject}: `{node_type}`"]
        described = facts.describe()
        if described:
            lines.append(f"refinements: {', '.join(described)}")
        representation = select(node_type, facts)
        lines.append(f"representation: {representation}")
        return "\n\n".join(lines)

    def _describe_function(self, bundle: AnalysisBundle, info: FunctionInfo) -> str:
        analysis = bundle.analysis.function(info.qualname)
        report: ContractReport | None = bundle.reports.get(info.qualname)
        lines = [f"`{info.name}{info.signature()}`"]
        if analysis is not None:
            lines.append(f"effects: {analysis.effects}")
            purity = "verified pure" if analysis.verified_pure else "impure"
            if info.declared_pure and not analysis.verified_pure:
                purity = "declared `@ppy.pure` but NOT verified"
            lines.append(f"purity: {purity}")
        if report is not None:
            lines.append(
                "llvm: native" if report.native_ok else f"llvm: boxed — {report.native_reason}"
            )
            lines.append(
                "parallel: accepted"
                if report.parallel_ok
                else f"parallel: rejected — {report.parallel_reason}"
            )
        if info.opt_level is not None:
            lines.append(f"optimization: O{info.opt_level}")
        return "\n\n".join(lines)

    def _describe_class(self, cls: ClassInfo) -> str:
        fields = ", ".join(f"{name}: {t}" for name, t in cls.fields.items())
        kind = (
            "pydantic model" if cls.is_pydantic else ("dataclass" if cls.is_dataclass else "class")
        )
        return f"`{cls.name}` ({kind})\n\nfields: {fields or 'none'}\n\nmro: {' -> '.join(cls.mro)}"
