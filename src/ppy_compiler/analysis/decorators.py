"""What a decorator does to the thing it decorates.

People bend runtimes with decorators, and a converter that rewrites decorated
source has to know which bends it is looking at. Every consumer -- annotation
materialization, class hoisting, purity planning -- asks this table instead of
keeping its own list, so they cannot disagree about what `@lru_cache` is.

Unknown is an answer: `semantics_of` returns None for a decorator nobody can
vouch for, and the caller must treat the decorated object as opaque.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = ["DecoratorSemantics", "definition_time_pure", "semantics_of"]


@dataclass(frozen=True, slots=True)
class DecoratorSemantics:
    """Capabilities of one known decorator."""

    #: Running the decorator at `def`/`class` time has no observable effect.
    pure_at_definition: bool = True
    #: The decorated object comes back `is`-identical.
    preserves_identity: bool = True
    #: Callers see the same parameters and return type.
    preserves_signature: bool = True
    #: The decorator inspects `__annotations__`, so adding annotations to the
    #: source can change what it builds.
    reads_annotations: bool = False


_INERT = DecoratorSemantics()
_WRAPPER = DecoratorSemantics(preserves_identity=False)

#: Decorators whose behavior is known. Keys are the dotted name as written,
#: minus any call arguments; a bare tail (`dataclass`) matches its common
#: `from x import y` spelling.
_KNOWN: dict[str, DecoratorSemantics] = {
    # PPY directives attach metadata and hand the object back (spec 6.1).
    "ppy.pure": _INERT,
    "ppy.jit": _INERT,
    "ppy.parallel": _INERT,
    "ppy.native": _INERT,
    "ppy.inline": _INERT,
    "ppy.noinline": _INERT,
    "ppy.specialize": _INERT,
    "ppy.fastmath": _INERT,
    "ppy.jax": _INERT,
    "ppy.opt": _INERT,
    "ppy.dynamic": _INERT,
    # Python-defined transforms with fixed meaning.
    "staticmethod": _WRAPPER,
    "classmethod": _WRAPPER,
    "property": DecoratorSemantics(preserves_identity=False, preserves_signature=False),
    "abc.abstractmethod": _INERT,
    "abstractmethod": _INERT,
    "typing.final": _INERT,
    "final": _INERT,
    "typing.overload": _INERT,
    "overload": _INERT,
    "typing.runtime_checkable": _INERT,
    "runtime_checkable": _INERT,
    "functools.wraps": _WRAPPER,
    "wraps": _WRAPPER,
    "functools.total_ordering": _INERT,
    "total_ordering": _INERT,
    "functools.cache": _WRAPPER,
    "cache": _WRAPPER,
    "functools.lru_cache": _WRAPPER,
    "lru_cache": _WRAPPER,
    "functools.cached_property": DecoratorSemantics(
        preserves_identity=False, preserves_signature=False
    ),
    "cached_property": DecoratorSemantics(preserves_identity=False, preserves_signature=False),
    # `singledispatch` and `dataclass` read annotations to decide what to
    # build; the object is still the declared one.
    "functools.singledispatch": DecoratorSemantics(
        preserves_identity=False, reads_annotations=True
    ),
    "singledispatch": DecoratorSemantics(preserves_identity=False, reads_annotations=True),
    "dataclasses.dataclass": DecoratorSemantics(reads_annotations=True),
    "dataclass": DecoratorSemantics(reads_annotations=True),
    "enum.unique": _INERT,
    "unique": _INERT,
}


def semantics_of(name: str, plugins=None) -> DecoratorSemantics | None:  # type: ignore[no-untyped-def]
    """The known behavior of a decorator, or None for one nobody vouches for.

    `name` is the decorator as written, arguments stripped: `@lru_cache(64)`
    asks for `lru_cache`.
    """
    found = _KNOWN.get(name)
    if found is not None:
        return found
    if plugins is not None:
        for plugin in getattr(plugins, "_plugins", ()):
            answer = getattr(plugin, "decorator_semantics", lambda _n: None)(name)
            if answer is not None:
                return answer
    return None


def _simple(node: ast.expr) -> bool:
    """An expression whose evaluation cannot be observed."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _simple(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_simple(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is None or _simple(k) for k in node.keys) and all(
            _simple(v) for v in node.values
        )
    if isinstance(node, ast.UnaryOp):
        return _simple(node.operand)
    if isinstance(node, ast.BinOp):
        return _simple(node.left) and _simple(node.right)
    if isinstance(node, ast.Subscript):
        # `list[int]` in an annotation position; evaluated, but observably
        # only for exotic `__class_getitem__` overrides.
        return _simple(node.value) and _simple(node.slice)
    return False


def _decorators_pure(decorator_list: list[ast.expr], plugins=None) -> bool:  # type: ignore[no-untyped-def]
    for decorator in decorator_list:
        call_args_simple = True
        target = decorator
        if isinstance(decorator, ast.Call):
            target = decorator.func
            call_args_simple = all(_simple(a) for a in decorator.args) and all(
                _simple(k.value) for k in decorator.keywords
            )
        known = semantics_of(ast.unparse(target), plugins)
        if known is None or not known.pure_at_definition or not call_args_simple:
            return False
    return True


def definition_time_pure(node: ast.stmt, plugins=None) -> bool:  # type: ignore[no-untyped-def]
    """Does *defining* this statement have observable effects?

    A `def` runs its decorators, default expressions, and annotations; a
    `class` runs its whole body, its bases, and its keywords. Reordering two
    statements is only meaning-preserving when at least one side is free of
    all of that -- which is what a hoist has to prove.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not _decorators_pure(node.decorator_list, plugins):
            return False
        arguments = node.args
        defaults = [*arguments.defaults, *[d for d in arguments.kw_defaults if d is not None]]
        if not all(_simple(d) for d in defaults):
            return False
        annotations = [
            a.annotation
            for a in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *([arguments.vararg] if arguments.vararg else []),
                *([arguments.kwarg] if arguments.kwarg else []),
            ]
            if a.annotation is not None
        ]
        if node.returns is not None:
            annotations.append(node.returns)
        return all(_simple(a) for a in annotations)
    if isinstance(node, ast.ClassDef):
        if not _decorators_pure(node.decorator_list, plugins):
            return False
        if node.keywords:
            # A metaclass or `__init_subclass__` keyword runs arbitrary code.
            return False
        if not all(_simple(base) for base in node.bases):
            return False
        return all(_body_statement_pure(child, plugins) for child in node.body)
    return isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass))


def _body_statement_pure(node: ast.stmt, plugins=None) -> bool:  # type: ignore[no-untyped-def]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return definition_time_pure(node, plugins)
    if isinstance(node, ast.AnnAssign):
        return _simple(node.annotation) and (node.value is None or _simple(node.value))
    if isinstance(node, ast.Assign):
        return _simple(node.value)
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Constant)
    return isinstance(node, ast.Pass)
