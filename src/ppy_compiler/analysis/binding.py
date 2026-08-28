"""Matching call arguments to the parameters that receive them.

Python's rules for this -- positional order, keywords by name, a bound
receiver that is never written down, `*args` that makes the rest unknowable --
are subtle enough that every place needing them should ask the same code.
Reimplementing them per pass is how one pass ends up disagreeing with another.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Bound", "bind_ast_call", "bind_call", "positional_values", "receiver_offset"]

#: Parameters a positional argument can never land on.
_NOT_POSITIONAL = frozenset({"keyword_only", "var_positional", "var_keyword"})
_NOT_KEYWORD = frozenset({"positional_only", "var_positional", "var_keyword"})


@dataclass(frozen=True, slots=True)
class Bound:
    """One argument, and the parameter it reaches."""

    index: int
    param: Any
    value: Any
    keyword: bool = False


def receiver_offset(info: Any) -> int:
    """How many leading parameters the call site does not write.

    `self` and `cls` are supplied by the attribute access, so the first
    argument written at the call site is the second parameter.
    """
    return 1 if info.is_method and not info.is_static else 0


def positional_values(args: Sequence[ast.expr]) -> list[ast.expr]:
    """The arguments whose position is still known.

    Everything after `*rest` could land anywhere, so it is dropped rather than
    matched to whichever parameter happens to follow.
    """
    kept: list[ast.expr] = []
    for argument in args:
        if isinstance(argument, ast.Starred):
            break
        kept.append(argument)
    return kept


def bind_call(
    params: Sequence[Any],
    positional: Sequence[Any],
    keywords: Iterable[tuple[str | None, Any]] = (),
    *,
    offset: int = 0,
) -> list[Bound]:
    """Match written arguments to parameters, by position and then by name.

    `offset` skips the parameters a call site does not write. An argument that
    reaches no parameter -- past the end, or a `**kwargs` splat, or a name the
    function does not declare -- is left out; deciding it is an error is the
    checker's job, not this one's.
    """
    bound: list[Bound] = []
    index = offset
    for value in positional:
        if index >= len(params):
            break
        param = params[index]
        if param.kind in _NOT_POSITIONAL:
            break
        bound.append(Bound(index, param, value))
        index += 1

    by_name = {p.name: (i, p) for i, p in enumerate(params) if i >= offset}
    for name, value in keywords:
        if name is None:
            continue
        found = by_name.get(name)
        if found is None or found[1].kind in _NOT_KEYWORD:
            continue
        bound.append(Bound(found[0], found[1], value, keyword=True))
    return bound


def bind_ast_call(info: Any, node: ast.Call) -> list[Bound]:
    """`bind_call` for a call as written in the source."""
    return bind_call(
        info.params,
        positional_values(node.args),
        [(k.arg, k.value) for k in node.keywords],
        offset=receiver_offset(info),
    )
