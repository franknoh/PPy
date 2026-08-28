"""Serializing semantic types, facts, and effects.

The analysis is rebuilt from source on every invocation, which is most of what
a fully cached build still costs. Persisting a module's per-function summaries
needs these values to survive a round trip exactly: a summary that decodes to
something subtly different would be worse than no cache at all, so `encode` and
`decode` are exact inverses for every value the checker produces, and a
property test holds them to it.
"""

from __future__ import annotations

from typing import Any

from . import types as T
from .effects import Effect, EffectSet
from .refinements import Facts, IntRange

__all__ = ["encode_type", "decode_type", "encode_facts", "decode_facts",
           "encode_effects", "decode_effects", "CodecError"]


class CodecError(ValueError):
    """A value the codec does not describe, rather than a silent wrong answer."""


def encode_type(t: T.Type) -> Any:
    match t:
        case T.AnyType():
            return ["any"]
        case T.UnknownType():
            return ["unknown", t.reason]
        case T.NeverType():
            return ["never"]
        case T.Literal():
            return ["literal", _encode_value(t.value), encode_type(t.base) if t.base else None]
        case T.Instance():
            return ["instance", t.name, [encode_type(a) for a in t.args], list(t.mro)]
        case T.Tuple_():
            return ["tuple", [encode_type(i) for i in t.items], t.homogeneous]
        case T.Union_():
            return ["union", [encode_type(m) for m in t.members]]
        case T.Callable_():
            return [
                "callable",
                [[p.name, encode_type(p.type), p.has_default, p.kind] for p in t.params],
                encode_type(t.ret),
                t.qualname,
                t.is_async,
                t.is_generator,
            ]
        case T.Module_():
            return ["module", t.name]
        case T.ClassObject():
            return ["class", t.name, encode_type(t.instance_type) if t.instance_type else None]
        case T.TypeVar_():
            return ["typevar", t.name, encode_type(t.bound) if t.bound else None]
    raise CodecError(f"no encoding for {type(t).__name__}")


def decode_type(raw: Any) -> T.Type:
    if not isinstance(raw, list) or not raw:
        raise CodecError(f"not an encoded type: {raw!r}")
    match raw[0]:
        case "any":
            return T.ANY
        case "unknown":
            return T.UnknownType(raw[1])
        case "never":
            return T.NEVER
        case "literal":
            base = decode_type(raw[2]) if raw[2] is not None else None
            return T.Literal(_decode_value(raw[1]), base)  # type: ignore[arg-type]
        case "instance":
            return T.Instance(raw[1], tuple(decode_type(a) for a in raw[2]), tuple(raw[3]))
        case "tuple":
            return T.Tuple_(tuple(decode_type(i) for i in raw[1]), homogeneous=raw[2])
        case "union":
            return T.Union_(tuple(decode_type(m) for m in raw[1]))
        case "callable":
            params = tuple(
                T.Param(name, decode_type(kind), has_default, param_kind)
                for name, kind, has_default, param_kind in raw[1]
            )
            return T.Callable_(params, decode_type(raw[2]), raw[3], raw[4], raw[5])
        case "module":
            return T.Module_(raw[1])
        case "class":
            instance = decode_type(raw[2]) if raw[2] is not None else None
            return T.ClassObject(raw[1], instance)  # type: ignore[arg-type]
        case "typevar":
            return T.TypeVar_(raw[1], decode_type(raw[2]) if raw[2] is not None else None)
    raise CodecError(f"no decoding for {raw[0]!r}")


#: A literal's value is a Python constant. `bool` is checked before `int`
#: because it is a subclass of it and would otherwise come back widened.
def _encode_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return ["v", value.__class__.__name__, value]
    if isinstance(value, bytes):
        return ["v", "bytes", value.hex()]
    raise CodecError(f"no encoding for the literal {value!r}")


def _decode_value(raw: Any) -> object:
    kind, value = raw[1], raw[2]
    if kind == "bytes":
        return bytes.fromhex(value)
    if kind == "NoneType":
        return None
    return {"str": str, "bool": bool, "int": int, "float": float}[kind](value)


def encode_facts(facts: Facts) -> Any:
    return {
        "int_range": [facts.int_range.low, facts.int_range.high] if facts.int_range else None,
        "length": facts.length,
        "constant": _encode_value(facts.constant) if facts.has_constant else None,
        "has_constant": facts.has_constant,
        "exact_class": facts.exact_class,
        "non_null": facts.non_null,
        "contiguous": facts.contiguous,
        "no_alias": facts.no_alias,
        "shape": list(facts.shape) if facts.shape is not None else None,
        "dtype": facts.dtype,
        "width": list(facts.width) if facts.width is not None else None,
        "float_bits": facts.float_bits,
    }


def decode_facts(raw: Any) -> Facts:
    span = raw["int_range"]
    return Facts(
        int_range=IntRange(span[0], span[1]) if span else None,
        length=raw["length"],
        constant=_decode_value(raw["constant"]) if raw["has_constant"] else None,
        has_constant=raw["has_constant"],
        exact_class=raw["exact_class"],
        non_null=raw["non_null"],
        contiguous=raw["contiguous"],
        no_alias=raw["no_alias"],
        shape=tuple(raw["shape"]) if raw["shape"] is not None else None,
        dtype=raw["dtype"],
        width=tuple(raw["width"]) if raw["width"] is not None else None,
        float_bits=raw["float_bits"],
    )


def encode_effects(effects: EffectSet) -> Any:
    return {
        "effects": sorted(e.name for e in effects.effects),
        "raises": sorted(effects.raises),
    }


def decode_effects(raw: Any) -> EffectSet:
    return EffectSet(
        frozenset(Effect[name] for name in raw["effects"]),
        frozenset(raw["raises"]),
    )
