"""Module graph construction (spec 7.4, 5.3)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..diagnostics import Diagnostic, DiagnosticBag, Severity, Span
from .parser import parse_file
from .source import SourceFile

__all__ = ["ImportEdge", "Module", "ModuleGraph", "build_graph", "resolve_module_name", "RUNTIME_MODULES"]

#: The PPY runtime package is never analyzed as project source (spec 6).
RUNTIME_MODULES = frozenset({"ppy"})


@dataclass(slots=True)
class ImportEdge:
    """One resolved or unresolved import in a module."""

    target: str
    node: ast.Import | ast.ImportFrom
    names: tuple[tuple[str, str | None], ...] = ()
    level: int = 0
    is_from: bool = False
    resolved_path: Path | None = None
    external: bool = False
    star: bool = False


@dataclass(slots=True)
class Module:
    name: str
    path: Path
    source: SourceFile
    is_package: bool = False
    imports: list[ImportEdge] = field(default_factory=list)

    @property
    def tree(self) -> ast.Module:
        assert self.source.tree is not None
        return self.source.tree

    @property
    def package(self) -> str:
        if self.is_package:
            return self.name
        return self.name.rpartition(".")[0]

    @property
    def is_ppy(self) -> bool:
        return self.path.suffix == ".ppy"


@dataclass(slots=True)
class ModuleGraph:
    root: Path
    search_paths: list[Path]
    modules: dict[str, Module] = field(default_factory=dict)
    entry: str | None = None
    external: set[str] = field(default_factory=set)

    def order(self) -> list[Module]:
        """Dependency-first order, with cycles broken deterministically."""
        seen: set[str] = set()
        stack: set[str] = set()
        out: list[Module] = []

        def visit(name: str) -> None:
            if name in seen or name in stack:
                return
            module = self.modules.get(name)
            if module is None:
                return
            stack.add(name)
            for edge in module.imports:
                if edge.target in self.modules:
                    visit(edge.target)
            stack.discard(name)
            seen.add(name)
            out.append(module)

        for name in sorted(self.modules):
            visit(name)
        if self.entry:
            visit(self.entry)
        return out

    def dependents_of(self, name: str) -> set[str]:
        return {
            module.name
            for module in self.modules.values()
            if any(edge.target == name for edge in module.imports)
        }


def resolve_module_name(path: Path, search_paths: list[Path]) -> str:
    """Derive a dotted module name for a file inside one of the search paths."""
    path = path.resolve()
    best: tuple[int, str] | None = None
    for search in search_paths:
        try:
            relative = path.relative_to(search.resolve())
        except ValueError:
            continue
        parts = list(relative.parts)
        if parts[-1].startswith("__init__."):
            parts.pop()
        else:
            parts[-1] = Path(parts[-1]).stem
        name = ".".join(parts) if parts else path.stem
        depth = len(search.resolve().parts)
        if best is None or depth > best[0]:
            best = (depth, name)
    return best[1] if best else path.stem


def _candidate_files(name: str, search_paths: list[Path]) -> list[tuple[Path, bool]]:
    tail = name.replace(".", "/")
    found: list[tuple[Path, bool]] = []
    for search in search_paths:
        for suffix in (".ppy", ".py"):
            module_file = search / f"{tail}{suffix}"
            if module_file.is_file():
                found.append((module_file, False))
            package_file = search / tail / f"__init__{suffix}"
            if package_file.is_file():
                found.append((package_file, True))
    return found


def _resolve_relative(module: Module, level: int, target: str) -> str:
    base = module.name if module.is_package else module.name.rpartition(".")[0]
    parts = base.split(".") if base else []
    drop = level - 1
    if drop:
        parts = parts[:-drop] if drop <= len(parts) else []
    if target:
        parts.append(target)
    return ".".join(p for p in parts if p)


def _collect_imports(module: Module) -> list[ImportEdge]:
    edges: list[ImportEdge] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(
                    ImportEdge(
                        target=alias.name,
                        node=node,
                        names=((alias.name, alias.asname),),
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            names = tuple((a.name, a.asname) for a in node.names)
            target = node.module or ""
            if node.level:
                target = _resolve_relative(module, node.level, target)
            edges.append(
                ImportEdge(
                    target=target,
                    node=node,
                    names=names,
                    level=node.level,
                    is_from=True,
                    star=any(a.name == "*" for a in node.names),
                )
            )
    return edges


def build_graph(
    entries: list[Path],
    search_paths: list[Path],
    diagnostics: DiagnosticBag,
    *,
    root: Path | None = None,
    follow_imports: bool = True,
    overlays: dict[Path, str] | None = None,
) -> ModuleGraph:
    """Parse the entry files and, transitively, every project module they import."""
    search_paths = [p for p in search_paths if p.is_dir()]
    graph = ModuleGraph(root=root or (entries[0].parent if entries else Path.cwd()), search_paths=search_paths)
    queue: list[tuple[str, Path, bool]] = []

    for entry in entries:
        name = resolve_module_name(entry, search_paths)
        queue.append((name, entry, entry.stem == "__init__"))
        if graph.entry is None:
            graph.entry = name

    while queue:
        name, path, is_package = queue.pop(0)
        if name in graph.modules:
            continue
        source, diagnostic = parse_file(path, overlays)
        if diagnostic is not None:
            diagnostics.add(diagnostic)
        if source is None or source.tree is None:
            continue
        module = Module(name=name, path=path, source=source, is_package=is_package)
        graph.modules[name] = module
        module.imports = _collect_imports(module)
        if not follow_imports:
            continue
        for edge in module.imports:
            if edge.star:
                diagnostics.add(
                    Diagnostic(
                        "E1103",
                        Severity.ERROR,
                        f"`from {edge.target} import *` leaves the module namespace unanalyzable",
                        Span(path, edge.node.lineno, edge.node.col_offset,
                             edge.node.end_lineno, edge.node.end_col_offset),
                        help="import the names you use explicitly",
                    )
                )
            if edge.target.partition(".")[0] in RUNTIME_MODULES:
                edge.external = True
                graph.external.add(edge.target)
                continue
            candidates = _candidate_files(edge.target, search_paths)
            if not candidates:
                # `from pkg import name` where `name` is a submodule.
                if edge.is_from:
                    for sub, _asname in edge.names:
                        sub_target = f"{edge.target}.{sub}" if edge.target else sub
                        for sub_path, sub_pkg in _candidate_files(sub_target, search_paths):
                            queue.append((sub_target, sub_path, sub_pkg))
                edge.external = True
                graph.external.add(edge.target)
                continue
            if len(candidates) > 1:
                paths = ", ".join(str(p) for p, _ in candidates)
                diagnostics.add(
                    Diagnostic(
                        "E1003",
                        Severity.ERROR,
                        f"module {edge.target!r} is provided by more than one source: {paths}",
                        Span(path, edge.node.lineno, edge.node.col_offset),
                        help="a project must not contain both foo.py and foo.ppy for the same module",
                    )
                )
            target_path, target_is_package = candidates[0]
            edge.resolved_path = target_path
            queue.append((edge.target, target_path, target_is_package))
    return graph
