"""Who in the project looks at annotations at runtime.

Materializing an inferred annotation is invisible -- until someone reads
`f.__annotations__`, calls `inspect.signature(f)`, or prints the module's
`__annotations__`, at which point the conversion has changed the program's
output. This scan finds those readers across the whole project so the
converter can leave the observed objects exactly as their author wrote them.

Resolution is best-effort and failure is conservative: a reflective read
whose target cannot be named blocks materialization everywhere, because the
target could be anything. One shape gets more precision, because it is
everywhere: a function reading annotations off its *own parameter* -- the
classic introspecting decorator -- observes exactly what is passed to it, so
only its call sites' arguments and whatever it decorates are blocked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .lexical import LexicalBindings
from .project_scan import ProjectScan, scan_project

__all__ = ["ReflectionIndex", "build_reflection_index"]

#: Callables whose argument's annotations become observable, by canonical
#: name -- the lexical bindings resolve `sig`, `i.signature`, and the rest
#: down to these before the check.
_READERS = frozenset({"inspect.signature", "inspect.get_annotations", "typing.get_type_hints"})


@dataclass(slots=True)
class ReflectionIndex:
    """Names whose annotations the project observes at runtime."""

    #: Dotted spellings whose `__annotations__`/signature someone reads.
    observed: set[str] = field(default_factory=set)
    #: Modules whose own `__annotations__` mapping is read.
    module_annotations: set[str] = field(default_factory=set)
    #: A reflective read whose target could not be named: everything may be
    #: observed, so nothing may be materialized.
    dynamic: bool = False

    def blocks_function(self, name: str, qualname: str) -> bool:
        del name
        if self.dynamic:
            return True
        # Spellings are canonical (module-qualified), so `homemade.f` blocks
        # only that `f`; matching bare tails would block every `f` there is.
        return any(
            qualname == spelling
            or qualname.endswith("." + spelling)
            or spelling.endswith("." + qualname)
            for spelling in self.observed
        )

    def blocks_module_globals(self, module: str) -> bool:
        if self.dynamic:
            return True
        return any(
            module == seen or module.endswith("." + seen) or seen.endswith("." + module)
            for seen in self.module_annotations
        )


def build_reflection_index(
    root: Path, source_roots: tuple[str, ...] = ("src", "."), *, scan: ProjectScan | None = None
) -> ReflectionIndex:
    """Index every runtime reader of annotations in the project.

    `scan` is the project already walked and parsed, shared with the other
    whole-project indexes; without one this walks the project itself.
    """
    if scan is None:
        scan = scan_project(root, source_roots)
    index = ReflectionIndex()
    # An unparseable file could read anything's annotations.
    index.dynamic = scan.tainted
    # Observers first: a function reading its own parameter's annotations is
    # itself a reader, and its call sites can live in any file.
    observers: set[str] = set()
    for scanned in scan.modules:
        _find_observers(scanned.nodes, scanned.module, scanned.bindings, observers)
    for scanned in scan.modules:
        _scan(scanned.nodes, scanned.module, scanned.bindings, index, frozenset(observers))
    return index


def _own_parameters(nodes: tuple[ast.AST, ...]) -> dict[int, tuple[str, frozenset[str]]]:
    """id(function node) -> (its canonical tail, its parameter names)."""
    found: dict[int, tuple[str, frozenset[str]]] = {}
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            names = frozenset(
                a.arg
                for a in [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *([arguments.vararg] if arguments.vararg else []),
                    *([arguments.kwarg] if arguments.kwarg else []),
                ]
            )
            found[id(node)] = (node.name, names)
    return found


def _reads_a_parameter(fn, params: frozenset[str], bindings) -> bool:  # type: ignore[no-untyped-def]
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "__annotations__":
            if isinstance(node.value, ast.Name) and node.value.id in params:
                return True
        elif isinstance(node, ast.Call) and node.args:
            if not bindings.targets_at(node.func) & _READERS:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id in params:
                return True
    return False


def _find_observers(
    nodes: tuple[ast.AST, ...], module: str, bindings: LexicalBindings, observers: set[str]
) -> None:
    # One walk for every function's parameters, then one per function for its
    # reads. Asking `_own_parameters` inside the loop walked the whole tree
    # once per function, which on a 3000-line module was most of a
    # conversion's time.
    parameters = _own_parameters(nodes)
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _, params = parameters[id(node)]
            if _reads_a_parameter(node, params, bindings):
                observers.add(f"{module}.{node.name}")


def _scan(
    nodes: tuple[ast.AST, ...],
    module: str,
    bindings: LexicalBindings,
    index: ReflectionIndex,
    observers: frozenset[str],
) -> None:
    parameters: dict[str, frozenset[str]] = {}
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            parameters[node.name] = frozenset(
                a.arg
                for a in [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *([arguments.vararg] if arguments.vararg else []),
                    *([arguments.kwarg] if arguments.kwarg else []),
                ]
            )

    def enclosing_params(name_node: ast.Name) -> bool:
        # Reads through *some* function's parameter are the observers' own
        # business, handled by call-site and decoration tracking below.
        return any(name_node.id in names for names in parameters.values())

    def record(target: ast.expr) -> None:
        found = bindings.targets_at(target)
        if found:
            index.observed.update(found)
            index.module_annotations.update(found)
            return
        if isinstance(target, ast.Name) and enclosing_params(target):
            return
        # A value nobody can name had its annotations read; anything could
        # be observed, so nothing may be materialized.
        index.dynamic = True

    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr == "__annotations__":
            record(node.value)
        elif isinstance(node, ast.Name) and node.id == "__annotations__":
            # A bare read reaches the module's own mapping, wherever it
            # appears in the file.
            index.module_annotations.add(module)
        elif isinstance(node, ast.Call):
            readers = bindings.targets_at(node.func)
            if readers & _READERS and node.args:
                record(node.args[0])
            elif readers & observers and node.args:
                # Whatever reaches an observer has its annotations read.
                record(node.args[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if bindings.targets_at(target) & observers:
                    index.observed.add(f"{module}.{node.name}")
