"""What the whole-project indexes need from one file, and nothing else.

`Final` needs to know who writes a name anywhere; materializing an
annotation needs to know who reads annotations anywhere. Both questions are
answered over every source under the project root, and a file that has not
changed answers the same way it did last time. So each file is reduced to a
small record -- the attribute writes it makes on other modules, the
annotation readers it holds and calls -- that the indexes combine, and that
the cache store keeps between runs. Converting one file in a large project
no longer parses the project.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .lexical import LexicalBindings

__all__ = ["READERS", "FileFacts", "facts_of"]

#: Library functions that read annotations off their argument.
READERS = frozenset({"inspect.signature", "inspect.get_annotations", "typing.get_type_hints"})


@dataclass(frozen=True, slots=True)
class FileFacts:
    """One file's contribution to the project-wide indexes.

    Every field is a tuple of plain values, so a record pickles small and
    compares by content.
    """

    #: `module -> attribute names` this file assigns, deletes, or `setattr`s
    #: on that module by a literal name.
    writes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: Modules hit by a `setattr`/`delattr` whose name is not a literal.
    dynamic_writes: tuple[str, ...] = ()
    #: Functions here that read annotations off their own parameter, by
    #: qualified name: what is handed to them is observed.
    observers: tuple[str, ...] = ()
    #: Each read of annotations: what the reader could be -- `None` for a
    #: direct `x.__annotations__` -- what the value read names, and whether
    #: it is a parameter of some function here. A reader that is neither a
    #: library reader nor an observer is not a read.
    reads: tuple[tuple[tuple[str, ...] | None, tuple[str, ...], bool], ...] = ()
    #: `__annotations__` named bare: the module's own.
    names_own_annotations: bool = False
    #: Each decorated definition: what the decorator could be, and the
    #: definition's qualified name. Decorated by an observer, it is observed.
    decorated: tuple[tuple[tuple[str, ...], str], ...] = ()


def facts_of(tree: ast.Module, module: str, bindings: LexicalBindings) -> FileFacts:
    """Reduce one parsed file to its facts."""
    nodes = tuple(ast.walk(tree))
    writes: dict[str, set[str]] = {}
    dynamic: set[str] = set()
    observers: list[str] = []
    reads: set[tuple[tuple[str, ...] | None, tuple[str, ...], bool]] = set()
    decorated: list[tuple[tuple[str, ...], str]] = []
    names_own = False
    parameters = _parameters(nodes)

    def is_parameter(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and any(node.id in names for names in parameters.values())

    def record_write(base: ast.expr, attr: str) -> None:
        for target in bindings.targets_at(base):
            writes.setdefault(target, set()).add(attr)

    def read(reader: tuple[str, ...] | None, value: ast.expr) -> None:
        reads.add((reader, tuple(sorted(bindings.targets_at(value))), is_parameter(value)))

    for node in nodes:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"setattr", "delattr"}
                and node.args
            ):
                written = node.args[1] if len(node.args) > 1 else None
                for target in bindings.targets_at(node.args[0]):
                    if isinstance(written, ast.Constant) and isinstance(written.value, str):
                        writes.setdefault(target, set()).add(written.value)
                    else:
                        dynamic.add(target)
            if node.args:
                readers = bindings.targets_at(node.func)
                if readers:
                    read(tuple(sorted(readers)), node.args[0])
            continue
        elif isinstance(node, ast.Attribute):
            if node.attr == "__annotations__":
                read(None, node.value)
            continue
        elif isinstance(node, ast.Name):
            if node.id == "__annotations__":
                names_own = True
            continue
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualname = f"{module}.{node.name}"
            if not isinstance(node, ast.ClassDef) and _reads_a_parameter(
                node, parameters[id(node)], bindings
            ):
                observers.append(qualname)
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                found = bindings.targets_at(target)
                if found:
                    decorated.append((tuple(sorted(found)), qualname))
            continue
        for target in targets:
            if isinstance(target, ast.Attribute):
                record_write(target.value, target.attr)

    return FileFacts(
        writes=tuple(sorted((name, tuple(sorted(attrs))) for name, attrs in writes.items())),
        dynamic_writes=tuple(sorted(dynamic)),
        observers=tuple(observers),
        reads=tuple(sorted(reads, key=repr)),
        names_own_annotations=names_own,
        decorated=tuple(decorated),
    )


def _parameters(nodes: tuple[ast.AST, ...]) -> dict[int, frozenset[str]]:
    """id(function node) -> its parameter names."""
    found: dict[int, frozenset[str]] = {}
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            found[id(node)] = frozenset(
                a.arg
                for a in [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *([arguments.vararg] if arguments.vararg else []),
                    *([arguments.kwarg] if arguments.kwarg else []),
                ]
            )
    return found


def _reads_a_parameter(fn: ast.AST, params: frozenset[str], bindings: LexicalBindings) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "__annotations__":
            if isinstance(node.value, ast.Name) and node.value.id in params:
                return True
        elif isinstance(node, ast.Call) and node.args:
            if not bindings.targets_at(node.func) & READERS:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id in params:
                return True
    return False
