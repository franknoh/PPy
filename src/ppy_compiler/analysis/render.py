"""Rendering semantic types back to valid Python annotation source."""

from __future__ import annotations

from dataclasses import dataclass

from . import types as T
from .refinements import Facts

__all__ = ["Rendered", "render_annotation", "renderable"]


@dataclass(frozen=True, slots=True)
class Rendered:
    text: str
    typing_imports: frozenset[str] = frozenset()
    ppy_imports: frozenset[str] = frozenset()


def renderable(t: T.Type) -> bool:
    """Can this type be written as an annotation without inventing `Any`?"""
    return render_annotation(t) is not None


#: PEP 586 admits these as `Literal` parameters. `float` is not one of them,
#: whatever the runtime accepts.
_LITERAL_BASES = (T.INT, T.STR, T.BYTES, T.BOOL)

#: Past a handful, a closed set reads as noise rather than as a contract.
_MAX_LITERALS = 8


def render_annotation(
    t: T.Type,
    facts: Facts | None = None,
    *,
    local_module: str = "",
    closed_literals: bool = False,
) -> Rendered | None:
    """Write a type as an annotation.

    `closed_literals` keeps a union of literals as `Literal[...]` rather than
    widening it. That is sound for a return type, where the body is the whole
    evidence for what comes back, and unsound for a parameter, where call sites
    are only a sample of what the function accepts.
    """
    marker = _marker(facts)
    if marker is not None and T.strip_literal(t) in (T.INT, T.FLOAT):
        return Rendered(marker, frozenset(), frozenset({marker}))
    if closed_literals:
        closed = _render_closed_set(t)
        if closed is not None:
            return closed
    return _render(t, local_module)


def _render_closed_set(t: T.Type) -> Rendered | None:
    """`Literal['a', 'b']` when a union is exactly a finite set of literals."""
    members = t.members if isinstance(t, T.Union_) else ()
    optional = any(m == T.NONE for m in members)
    literals = [m for m in members if isinstance(m, T.Literal)]
    if len(literals) != len(members) - (1 if optional else 0):
        return None
    if not 2 <= len(literals) <= _MAX_LITERALS:
        return None
    if any(m.base not in _LITERAL_BASES for m in literals):
        return None
    seen: list[str] = []
    for member in literals:
        text = repr(member.value)
        if text not in seen:
            seen.append(text)
    if len(seen) < 2:
        return None
    rendered = f"Literal[{', '.join(seen)}]"
    return Rendered(
        f"{rendered} | None" if optional else rendered, frozenset({"Literal"})
    )


def _marker(facts: Facts | None) -> str | None:
    if facts is None:
        return None
    if facts.width is not None:
        bits, signed = facts.width
        return f"{'i' if signed else 'u'}{bits}"
    if facts.float_bits is not None and facts.float_bits != 64:
        return f"f{facts.float_bits}"
    return None


def _render(t: T.Type, local_module: str) -> Rendered | None:
    if isinstance(t, (T.UnknownType, T.AnyType)):
        return None
    if isinstance(t, T.NeverType):
        return Rendered("Never", frozenset({"Never"}))
    if t == T.NONE:
        return Rendered("None")

    if isinstance(t, T.Literal):
        base = t.base or T.OBJECT
        return _render(base, local_module)

    if isinstance(t, T.Union_):
        parts: list[Rendered] = []
        for member in t.members:
            rendered = _render(member, local_module)
            if rendered is None:
                return None
            parts.append(rendered)
        # `None` reads better last in an optional union. Widening two members
        # to the same text -- two literals of one base -- leaves one of them.
        ordered = [p for p in parts if p.text != "None"] + [p for p in parts if p.text == "None"]
        unique: list[Rendered] = []
        for part in ordered:
            if all(part.text != kept.text for kept in unique):
                unique.append(part)
        return Rendered(
            " | ".join(p.text for p in unique),
            frozenset().union(*[p.typing_imports for p in parts]),
            frozenset().union(*[p.ppy_imports for p in parts]),
        )

    if isinstance(t, T.Tuple_):
        if not t.items:
            return Rendered("tuple[()]")
        parts = []
        for item in t.items:
            rendered = _render(item, local_module)
            if rendered is None:
                return None
            parts.append(rendered)
        inner = ", ".join(p.text for p in parts)
        if t.homogeneous:
            inner = f"{parts[0].text}, ..."
        return Rendered(
            f"tuple[{inner}]",
            frozenset().union(*[p.typing_imports for p in parts]),
            frozenset().union(*[p.ppy_imports for p in parts]),
        )

    if isinstance(t, T.Callable_):
        parts = []
        for param in t.params:
            rendered = _render(param.type, local_module)
            if rendered is None:
                return None
            parts.append(rendered)
        returned = _render(t.ret, local_module)
        if returned is None:
            return None
        inner = ", ".join(p.text for p in parts)
        return Rendered(
            f"Callable[[{inner}], {returned.text}]",
            frozenset({"Callable"}).union(*[p.typing_imports for p in parts], returned.typing_imports),
            frozenset().union(*[p.ppy_imports for p in parts], returned.ppy_imports),
        )

    if isinstance(t, T.ClassObject):
        return Rendered(f"type[{_short(t.name, local_module)}]")

    if isinstance(t, T.Instance):
        name = _short(t.name, local_module)
        if name in {"Sequence", "Iterable", "Iterator", "Mapping", "Generator", "Coroutine"}:
            typing_imports = frozenset({name})
        else:
            typing_imports = frozenset()
        if not t.args:
            return Rendered(name, typing_imports)
        parts = []
        for argument in t.args:
            rendered = _render(argument, local_module)
            if rendered is None:
                return None
            parts.append(rendered)
        return Rendered(
            f"{name}[{', '.join(p.text for p in parts)}]",
            typing_imports.union(*[p.typing_imports for p in parts]),
            frozenset().union(*[p.ppy_imports for p in parts]),
        )
    return None


def _short(qualname: str, local_module: str) -> str:
    if local_module and qualname.startswith(local_module + "."):
        return qualname[len(local_module) + 1 :]
    if "." in qualname and not qualname.startswith(("numpy.", "torch.", "jax.", "pydantic.")):
        return qualname.rpartition(".")[2]
    return qualname
