"""The native signature of an IR function, read off the IR.

The frontend records on each function what the binder needs -- the symbol,
the qualified name, whether the GIL may be released, how each buffer
parameter is passed, which class a struct parameter flattens -- so that a
`.ppyir` module carries its own ABI and can be built without the analysis
that produced it.
"""

from __future__ import annotations

from ppy_runtime.abi import NativeParam, NativeSignature

from ..ir import (
    ArrayType,
    BoolType,
    BufferType,
    FloatType,
    IntType,
    IRFunction,
    IRType,
    PtrType,
    StructType,
    TupleType,
)

__all__ = ["signature_from_ir"]


def _kind(t: IRType) -> str:
    if isinstance(t, BoolType):
        return "bool"
    if isinstance(t, FloatType):
        return "float"
    if isinstance(t, IntType):
        if t.width == 64 and t.signed:
            return "int"
        return str(t)
    raise ValueError(f"{t} has no native ABI atom")


def signature_from_ir(function: IRFunction) -> NativeSignature:
    parameters: list[NativeParam] = []
    for (name, t), attributes in zip(function.params, function.param_attributes, strict=True):
        if isinstance(t, BufferType):
            kind = str(attributes.get("ppy.kind", "view"))
            parameters.append(NativeParam(name, kind, _kind(t.element)))
        elif isinstance(t, PtrType):
            element = t.pointee
            shape = ()
            if isinstance(element, ArrayType):
                shape = element.shape
                element = element.element
            parameters.append(
                NativeParam(name, "ptr" if t.mutable else "const_ptr", _kind(element), shape=shape)
            )
        elif isinstance(t, TupleType):
            parameters.append(NativeParam(name, "tuple", elements=tuple(_kind(i) for i in t.items)))
        elif isinstance(t, StructType):
            parameters.append(
                NativeParam(
                    name,
                    "object",
                    fields=tuple((field, _kind(item)) for field, item in t.fields),
                    class_name=str(attributes.get("ppy.class", t.name)),
                )
            )
        else:
            parameters.append(NativeParam(name, _kind(t)))
    atoms: list[str] = []
    for t in function.results:
        if isinstance(t, TupleType):
            atoms.extend(_kind(item) for item in t.items)
        else:
            atoms.append(_kind(t))
    if not atoms:
        atoms = ["int"]
    from ppy_runtime.abi import _abi_name

    return NativeSignature(
        qualname=str(function.attributes.get("ppy.qualname", function.name)),
        symbol=str(function.attributes.get("ppy.symbol", function.name)),
        parameters=tuple(parameters),
        returns=tuple(_abi_name(a) for a in atoms),
        releases_gil=bool(function.attributes.get("ppy.releases_gil", False)),
        cpu_features=tuple(str(f) for f in function.attributes.get("cpu.features", ())),  # type: ignore[union-attr]
        returned="None" if not function.results else "",
        result_kinds=tuple(atoms) if function.results and function.results[0] == IntType(8) else (),
    )
