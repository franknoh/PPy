"""Annotation resolution: Python annotations to semantic type + refinements.

Implements spec 8.1 (Python typing as the baseline), 6.3/6.4 (PPY markers) and
24 (the generic `Annotated` extension protocol).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..diagnostics import Diagnostic, DiagnosticBag, Severity
from ..frontend.source import span_of
from . import types as T
from .refinements import Facts, IntRange, width_range

__all__ = [
    "AnnotationResolver",
    "Resolved",
    "Resolver",
    "narrow_element",
]


class Resolver(Protocol):
    """Name resolution service supplied by the module symbol table."""

    def canonical(self, expr: ast.expr) -> str | None: ...

    def class_instance(self, qualname: str, args: tuple[T.Type, ...]) -> T.Instance | None: ...

    def type_alias(self, name: str) -> ast.expr | None: ...


@dataclass(frozen=True, slots=True)
class Resolved:
    type: T.Type
    facts: Facts = field(default_factory=Facts)

    def with_facts(self, **updates: object) -> Resolved:
        return Resolved(self.type, self.facts.with_(**updates))


_SIMPLE: dict[str, T.Type] = {
    "builtins.int": T.INT,
    "builtins.float": T.FLOAT,
    "builtins.bool": T.BOOL,
    "builtins.str": T.STR,
    "builtins.bytes": T.BYTES,
    "builtins.complex": T.COMPLEX,
    "builtins.object": T.OBJECT,
    "builtins.type": T.instance("type"),
    "builtins.bytearray": T.instance("bytearray"),
    "builtins.memoryview": T.instance("memoryview"),
    "builtins.range": T.instance("range"),
    "builtins.slice": T.instance("slice"),
    "builtins.BaseException": T.instance("BaseException"),
    "builtins.Exception": T.instance("Exception"),
    "types.NoneType": T.NONE,
    "typing.Never": T.NEVER,
    "typing.NoReturn": T.NEVER,
    "typing.Text": T.STR,
}

_BARE_GENERIC = {
    "builtins.list": ("list", 1),
    "builtins.dict": ("dict", 2),
    "builtins.set": ("set", 1),
    "builtins.frozenset": ("frozenset", 1),
    "builtins.tuple": ("tuple", 1),
}

_ABSTRACT = {
    "typing.Sequence": "Sequence",
    "typing.Iterable": "Iterable",
    "typing.Iterator": "Iterator",
    "typing.Mapping": "Mapping",
    "typing.Generator": "Generator",
    "typing.Coroutine": "Coroutine",
    "typing.Awaitable": "Awaitable",
    "typing.AsyncIterator": "AsyncIterator",
    "typing.AsyncIterable": "AsyncIterable",
    "collections.abc.Awaitable": "Awaitable",
    "collections.abc.Coroutine": "Coroutine",
    "collections.abc.AsyncIterator": "AsyncIterator",
    "collections.abc.Sequence": "Sequence",
    "collections.abc.Iterable": "Iterable",
    "collections.abc.Iterator": "Iterator",
    "collections.abc.Mapping": "Mapping",
    "collections.abc.Generator": "Generator",
}

_INT_MARKERS = {
    f"ppy.{'i' if signed else 'u'}{bits}": (bits, signed)
    for bits in (8, 16, 32, 64)
    for signed in (True, False)
}
_FLOAT_MARKERS = {f"ppy.f{bits}": bits for bits in (16, 32, 64)}

_PASSTHROUGH = {"typing.Final", "typing.ClassVar", "typing.Required", "typing.NotRequired"}


#: Element types whose storage is narrower than a machine word, by the width
#: the marker asks for. Only a byte is offered: it is what text and packed
#: data need, and every wider one is the machine word already.
_NARROW_ELEMENTS = {(8, True): "i8", (8, False): "u8"}


def narrow_element(resolved: Resolved) -> T.Type | None:
    """`i8`/`u8` as a buffer element: one byte, sign as declared."""
    width = resolved.facts.width
    if resolved.type != T.INT or width is None:
        return None
    name = _NARROW_ELEMENTS.get(tuple(width))
    return T.Instance(name, (), (name, "int", "object")) if name else None


class AnnotationResolver:
    """Turns an annotation expression into a semantic type plus proven facts."""

    def __init__(
        self, resolver: Resolver, path: Path, diagnostics: DiagnosticBag, *, strict: bool = True
    ) -> None:
        self.resolver = resolver
        self.path = path
        self.diagnostics = diagnostics
        self.strict = strict
        self._expanding: set[str] = set()

    def resolve(self, expr: ast.expr | None) -> Resolved:
        if expr is None:
            return Resolved(T.UNKNOWN)
        return self._resolve(expr)

    def _resolve(self, expr: ast.expr) -> Resolved:
        if isinstance(expr, ast.Constant):
            return self._constant(expr)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            left = self._resolve(expr.left)
            right = self._resolve(expr.right)
            return Resolved(T.union(left.type, right.type))
        if isinstance(expr, ast.Subscript):
            return self._subscript(expr)
        if isinstance(expr, (ast.Name, ast.Attribute)):
            return self._named(expr)
        if isinstance(expr, ast.Tuple):
            items = [self._resolve(e).type for e in expr.elts]
            return Resolved(T.Tuple_(tuple(items)))
        self._error("E1301", "unsupported annotation form", expr)
        return Resolved(T.UNKNOWN)

    def _constant(self, expr: ast.Constant) -> Resolved:
        if expr.value is None:
            return Resolved(T.NONE)
        if isinstance(expr.value, str):
            try:
                inner = ast.parse(expr.value, mode="eval").body
            except SyntaxError:
                self._error("E1301", f"invalid forward reference {expr.value!r}", expr)
                return Resolved(T.UNKNOWN)
            ast.copy_location(inner, expr)
            for node in ast.walk(inner):
                ast.copy_location(node, expr)
            return self._resolve(inner)
        if expr.value is Ellipsis:
            return Resolved(T.ELLIPSIS_T)
        self._error("E1301", "annotation must be a type expression", expr)
        return Resolved(T.UNKNOWN)

    def _named(self, expr: ast.expr) -> Resolved:
        alias = self._alias(expr)
        if alias is not None:
            return alias
        qualname = self.resolver.canonical(expr)
        if qualname is None:
            self._error(
                "E1101",
                f"cannot resolve the annotation `{ast.unparse(expr)}`",
                expr,
                help="import the name, or annotate with a type the project can see",
            )
            return Resolved(T.UNKNOWN)
        if qualname in _SIMPLE:
            return Resolved(_SIMPLE[qualname])
        if qualname == "typing.Any":
            return Resolved(T.ANY)
        if qualname == "ppy.Dynamic":
            return Resolved(T.DYNAMIC)
        if qualname in _INT_MARKERS:
            bits, signed = _INT_MARKERS[qualname]
            return Resolved(T.INT, Facts(width=(bits, signed), int_range=width_range(bits, signed)))
        if qualname in _FLOAT_MARKERS:
            return Resolved(T.FLOAT, Facts(float_bits=_FLOAT_MARKERS[qualname]))
        if qualname in _BARE_GENERIC:
            name, arity = _BARE_GENERIC[qualname]
            self._bare_generic(name, expr)
            return Resolved(
                T.Instance(name, tuple(T.ANY for _ in range(arity)), T.BUILTIN_MRO.get(name, ()))
            )
        if qualname in _ABSTRACT:
            return Resolved(T.instance(_ABSTRACT[qualname]))
        found = self.resolver.class_instance(qualname, ())
        if found is not None:
            return Resolved(found)
        self._error(
            "E1101",
            f"`{qualname}` is not a type the project can analyze",
            expr,
            help="add a stub, a plugin, or an explicit ppy.dynamic boundary",
        )
        return Resolved(T.UNKNOWN)

    def _subscript(self, expr: ast.Subscript) -> Resolved:
        alias = self._alias(expr.value)
        if alias is not None:
            # A subscripted alias is a generic alias; v1 resolves the base only.
            return alias
        qualname = self.resolver.canonical(expr.value)
        args = self._slice_items(expr.slice)
        if qualname is None:
            self._error("E1101", f"cannot resolve `{ast.unparse(expr.value)}`", expr.value)
            return Resolved(T.UNKNOWN)

        if qualname == "typing.Annotated":
            return self._annotated(args, expr)
        if qualname == "typing.Optional":
            inner = self._resolve(args[0]) if args else Resolved(T.UNKNOWN)
            return Resolved(T.union(inner.type, T.NONE), inner.facts)
        if qualname == "typing.Union":
            return Resolved(T.union(*[self._resolve(a).type for a in args]))
        if qualname == "typing.Literal":
            return self._literal(args, expr)
        if qualname in {"typing.Callable", "collections.abc.Callable"}:
            return self._callable(args, expr)
        if qualname in _PASSTHROUGH:
            return self._resolve(args[0]) if args else Resolved(T.UNKNOWN)
        if qualname == "ppy.Array":
            return self._ppy_array(args, expr)
        if qualname == "ppy.Vector":
            element = self._resolve(args[0]).type if args else T.UNKNOWN
            return Resolved(T.list_of(element))
        if qualname == "ppy.Buffer":
            resolved = self._resolve(args[0]) if args else Resolved(T.UNKNOWN)
            # `Buffer[ppy.i8]` is a byte per element, not a 64-bit int with a
            # width note on it: the width is the storage, so it has to survive
            # into the element type the ABI reads.
            element = narrow_element(resolved) or resolved.type
            return Resolved(T.instance("Buffer", element), Facts(contiguous=True))

        resolved_args = tuple(self._resolve(a).type for a in args)
        if qualname in _BARE_GENERIC:
            name, arity = _BARE_GENERIC[qualname]
            if name == "tuple":
                return self._tuple(args, expr)
            if len(resolved_args) != arity:
                self._error("E1301", f"`{name}` takes {arity} type argument(s)", expr)
                return Resolved(T.UNKNOWN)
            return Resolved(T.Instance(name, resolved_args, T.BUILTIN_MRO.get(name, ())))
        if qualname in _ABSTRACT:
            return Resolved(T.instance(_ABSTRACT[qualname], *resolved_args))
        found = self.resolver.class_instance(qualname, resolved_args)
        if found is not None:
            return Resolved(found)
        self._error("E1101", f"`{qualname}` is not a generic type the project can analyze", expr)
        return Resolved(T.UNKNOWN)

    def _annotated(self, args: list[ast.expr], expr: ast.Subscript) -> Resolved:
        if not args:
            self._error("E1301", "Annotated requires a base type", expr)
            return Resolved(T.UNKNOWN)
        base = self._resolve(args[0])
        facts = base.facts
        for meta in args[1:]:
            facts = self._apply_metadata(meta, facts, base.type)
        return Resolved(base.type, facts)

    def _apply_metadata(self, meta: ast.expr, facts: Facts, base: T.Type) -> Facts:
        if not isinstance(meta, ast.Call):
            self._unknown_metadata(meta)
            return facts
        qualname = self.resolver.canonical(meta.func)
        values = [self._literal_value(a) for a in meta.args]
        keywords = {k.arg: self._literal_value(k.value) for k in meta.keywords if k.arg}

        match qualname:
            case "ppy.Range" if len(values) == 2:
                low, high = values
                if isinstance(low, int) and isinstance(high, int):
                    existing = facts.int_range or IntRange()
                    return facts.with_(int_range=existing.meet(IntRange(low, high)))
                return facts
            case "ppy.Length" if len(values) == 1 and isinstance(values[0], int):
                return facts.with_(length=values[0])
            case "ppy.NoAlias":
                return facts.with_(no_alias=True)
            case "ppy.Contiguous":
                return facts.with_(contiguous=True)
            case "ppy.Shape":
                dims = tuple(v for v in values if isinstance(v, (int, str)))
                return facts.with_(shape=dims)
            case "ppy.DType" if len(values) == 1 and isinstance(values[0], str):
                return facts.with_(dtype=values[0])
            case "ppy.IntWidth" if len(values) >= 1 and isinstance(values[0], int):
                signed = bool(values[1]) if len(values) > 1 else bool(keywords.get("signed", True))
                return facts.with_(
                    width=(values[0], signed), int_range=width_range(values[0], signed)
                )
            case "ppy.FloatWidth" if len(values) == 1 and isinstance(values[0], int):
                return facts.with_(float_bits=values[0])
            case "pydantic.Field" | "pydantic.fields.Field":
                return self._pydantic_field(keywords, facts, base)
        self._unknown_metadata(meta)
        return facts

    def _pydantic_field(self, keywords: dict[str, object], facts: Facts, base: T.Type) -> Facts:
        """Pydantic numeric constraints become refinements (spec 23.4)."""
        low = high = None
        if isinstance(v := keywords.get("ge"), int):
            low = v
        if isinstance(v := keywords.get("gt"), int):
            low = v + 1 if low is None else max(low, v + 1)
        if isinstance(v := keywords.get("le"), int):
            high = v
        if isinstance(v := keywords.get("lt"), int):
            high = v - 1 if high is None else min(high, v - 1)
        if (low is not None or high is not None) and T.strip_literal(base) == T.INT:
            existing = facts.int_range or IntRange()
            facts = facts.with_(int_range=existing.meet(IntRange(low, high)))
        if (
            isinstance(length := keywords.get("min_length"), int)
            and keywords.get("max_length") == length
        ):
            facts = facts.with_(length=length)
        return facts

    def _unknown_metadata(self, meta: ast.expr) -> None:
        if self.strict:
            self.diagnostics.add(
                Diagnostic(
                    "W2003",
                    Severity.WARNING,
                    f"unknown Annotated metadata `{ast.unparse(meta)}` "
                    "was preserved but not interpreted",
                    span_of(self.path, meta),
                )
            )

    def _literal(self, args: list[ast.expr], expr: ast.Subscript) -> Resolved:
        members: list[T.Type] = []
        for arg in args:
            value = self._literal_value(arg)
            if value is _MISSING:
                self._error("E1301", "Literal takes constant values only", arg)
                return Resolved(T.UNKNOWN)
            members.append(T.type_of_constant(value))
        return Resolved(T.union(*members) if members else T.UNKNOWN)

    def _callable(self, args: list[ast.expr], expr: ast.Subscript) -> Resolved:
        if len(args) != 2:
            self._error("E1301", "Callable takes a parameter list and a return type", expr)
            return Resolved(T.UNKNOWN)
        params_expr, ret_expr = args
        ret = self._resolve(ret_expr).type
        if isinstance(params_expr, ast.Constant) and params_expr.value is Ellipsis:
            return Resolved(T.Callable_((), ret))
        if not isinstance(params_expr, ast.List):
            self._error("E1301", "Callable parameters must be a list", params_expr)
            return Resolved(T.UNKNOWN)
        params = tuple(
            T.Param(f"a{i}", self._resolve(e).type) for i, e in enumerate(params_expr.elts)
        )
        return Resolved(T.Callable_(params, ret))

    def _tuple(self, args: list[ast.expr], expr: ast.Subscript) -> Resolved:
        if len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis:
            return Resolved(T.Tuple_((self._resolve(args[0]).type,), homogeneous=True))
        if len(args) == 1 and isinstance(args[0], ast.Tuple) and not args[0].elts:
            return Resolved(T.Tuple_(()))
        return Resolved(T.Tuple_(tuple(self._resolve(a).type for a in args)))

    def _ppy_array(self, args: list[ast.expr], expr: ast.Subscript) -> Resolved:
        if len(args) != 2:
            self._error("E1301", "ppy.Array takes an element type and a length: Array[T, N]", expr)
            return Resolved(T.UNKNOWN)
        element = self._resolve(args[0]).type
        length = self._literal_value(args[1])
        facts = Facts(length=length) if isinstance(length, int) else Facts()
        if isinstance(length, int):
            return Resolved(T.Tuple_(tuple(element for _ in range(length))), facts)
        return Resolved(T.Tuple_((element,), homogeneous=True), facts)

    def _alias(self, expr: ast.expr) -> Resolved | None:
        """Expand a module-level type alias, guarding against a cycle."""
        if not isinstance(expr, ast.Name):
            return None
        lookup = getattr(self.resolver, "type_alias", None)
        if lookup is None:
            return None
        target = lookup(expr.id)
        if target is None or expr.id in self._expanding:
            return None
        self._expanding.add(expr.id)
        try:
            return self._resolve(target)
        finally:
            self._expanding.discard(expr.id)

    def _slice_items(self, node: ast.expr) -> list[ast.expr]:
        if isinstance(node, ast.Tuple):
            return list(node.elts)
        return [node]

    def _literal_value(self, expr: ast.expr) -> object:
        try:
            return ast.literal_eval(expr)
        except (ValueError, SyntaxError, TypeError):
            return _MISSING

    def _bare_generic(self, name: str, expr: ast.expr) -> None:
        self.diagnostics.add(
            Diagnostic(
                "E1203" if self.strict else "W2003",
                Severity.ERROR if self.strict else Severity.WARNING,
                f"bare `{name}` would introduce an implicit Any element type",
                span_of(self.path, expr),
                help=f"write `{name}[...]` with explicit type arguments",
            )
        )

    def _error(self, code: str, message: str, node: ast.AST, help: str | None = None) -> None:
        self.diagnostics.add(
            Diagnostic(code, Severity.ERROR, message, span_of(self.path, node), help=help)
        )


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()
