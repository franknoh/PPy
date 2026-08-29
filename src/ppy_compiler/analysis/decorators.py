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

__all__ = ["DecoratorSemantics", "definition_time_reorder_safe", "semantics_of"]


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

#: Decorators whose behavior is known, by canonical name. Symbol collection
#: resolves every decorator through the project's imports before it gets
#: here, so a user-defined `cache` arrives as `theirmodule.cache` and matches
#: nothing -- bare spellings are deliberately absent.
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
    "ppy.reflective": _INERT,
    # Python-defined transforms with fixed meaning.
    "builtins.staticmethod": _WRAPPER,
    "builtins.classmethod": _WRAPPER,
    "builtins.property": DecoratorSemantics(preserves_identity=False, preserves_signature=False),
    "abc.abstractmethod": _INERT,
    "typing.final": _INERT,
    "typing.overload": _INERT,
    "typing.runtime_checkable": _INERT,
    "functools.wraps": _WRAPPER,
    "functools.total_ordering": _INERT,
    "functools.cache": _WRAPPER,
    "functools.lru_cache": _WRAPPER,
    "functools.cached_property": DecoratorSemantics(
        preserves_identity=False, preserves_signature=False
    ),
    # `singledispatch` and `dataclass` read annotations to decide what to
    # build; the object is still the declared one.
    "functools.singledispatch": DecoratorSemantics(
        preserves_identity=False, reads_annotations=True
    ),
    "dataclasses.dataclass": DecoratorSemantics(reads_annotations=True),
    "enum.unique": _INERT,
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
    """An expression whose evaluation is provably unobservable.

    This is deliberately narrow. `holder.Base` can run a `__getattribute__`,
    `a + b` an `__add__`, `Base[int]` a `__class_getitem__` -- each is user
    code the reorder would run at a different time. Only literals, bare
    names, and literal containers of the same qualify.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_simple(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is None or _simple(k) for k in node.keys) and all(
            _simple(v) for v in node.values
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.operand, ast.Constant)
    return False


def _decorators_pure(decorator_list: list[ast.expr], plugins=None, identify=None) -> bool:  # type: ignore[no-untyped-def]
    for decorator in decorator_list:
        call_args_simple = True
        if isinstance(decorator, ast.Call):
            call_args_simple = all(_simple(a) for a in decorator.args) and all(
                _simple(k.value) for k in decorator.keywords
            )
        name = (
            identify(decorator)
            if identify is not None
            else ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
        )
        known = semantics_of(name, plugins)
        if known is None or not known.pure_at_definition or not call_args_simple:
            return False
    return True


def definition_time_reorder_safe(node: ast.stmt, plugins=None, identify=None) -> bool:  # type: ignore[no-untyped-def]
    """May this statement change position without changing behavior?

    A `def` runs its decorators, default expressions, and annotations; a
    `class` runs its whole body, its bases, and its keywords. A hoist has to
    prove this for the statement it moves *and* for every statement it
    crosses: a crossed decorator observing `globals()` sees a different world
    once something lands above it.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not _decorators_pure(node.decorator_list, plugins, identify):
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
        if not _decorators_pure(node.decorator_list, plugins, identify):
            return False
        if node.keywords:
            # A metaclass or `__init_subclass__` keyword runs arbitrary code.
            return False
        # Any explicit base can carry an `__init_subclass__` or a metaclass,
        # both of which run at class creation and may look around; a base
        # being *spelled* simply proves nothing about what creating the
        # subclass executes. Only `object` is known inert.
        if any(not (isinstance(base, ast.Name) and base.id == "object") for base in node.bases):
            return False
        return all(_body_statement_pure(child, plugins, identify) for child in node.body)
    return isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass))


def _literal(node: ast.expr) -> bool:
    """A value that is provably not a descriptor.

    A bare name in a class body may be a descriptor whose `__set_name__`
    runs at class creation; a literal cannot be one.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _literal(k) for k in node.keys) and all(
            _literal(v) for v in node.values
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.operand, ast.Constant)
    return False


def _body_statement_pure(node: ast.stmt, plugins=None, identify=None) -> bool:  # type: ignore[no-untyped-def]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return definition_time_reorder_safe(node, plugins, identify)
    if isinstance(node, ast.AnnAssign):
        return _simple(node.annotation) and (node.value is None or _literal(node.value))
    if isinstance(node, ast.Assign):
        return _literal(node.value)
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Constant)
    return isinstance(node, ast.Pass)
