"""Stable PPY diagnostic codes (spec 29.1)."""

from __future__ import annotations

__all__ = ["CODES", "describe"]

CODES: dict[str, str] = {
    "E1001": "The source could not be parsed by the configured CPython grammar.",
    "E1002": "The source file could not be read.",
    "E1003": "A module is provided by both a .py and a .ppy source under the same qualified name.",
    "E1101": "A name is used before any binding reaches it.",
    "E1102": "An import target could not be resolved to a module in the project or environment.",
    "E1103": "A `from ... import *` makes the module namespace unanalyzable.",
    "E1201": (
        "A parameter has no annotation and no inferable type, "
        "which would introduce an implicit `Any`."
    ),
    "E1202": "An attribute could not be resolved on a statically known type.",
    "E1203": "A generic type was used bare, which would introduce an implicit `Any` element type.",
    "E1204": "A decorator applies an unknown transform, so the decorated signature is unknown.",
    "E1206": "An attribute was read through a value that may be `None`.",
    "E1205": "A `ppy.` decorator names a directive the runtime does not export.",
    "E1301": "An assignment or argument does not match the declared type.",
    "E1302": "An operator has no definition for the operand types.",
    "E1303": "A returned value does not match the declared return type.",
    "E1304": "A stable type could not be inferred for a parameter or local.",
    "E1305": "A call does not match the callee's parameter list.",
    "E1306": "A callable's signature is unknown, so the call cannot be typed.",
    "E1401": "A value provably leaves the range promised by a fixed-width marker.",
    "E1402": (
        "A fixed-width contract needs a runtime check that the selected contract mode forbids."
    ),
    "E1501": "`eval`, `exec`, or runtime code-object construction is not statically analyzable.",
    "E1502": "Mutation of `globals()`, `locals()`, or frame locals is not statically analyzable.",
    "E1503": "An import target is not a compile-time constant.",
    "E1504": "A dynamic Python feature requires an explicit `ppy.dynamic` boundary.",
    "E1505": "A dynamic boundary is forbidden by the project configuration.",
    "E1506": "Attribute or class mutation after analysis is not supported.",
    "E1507": "A class is constructed dynamically, from computed bases or an unvouched metaclass.",
    "E1508": "A dynamic value crosses into typed code without a checked conversion.",
    "E1601": "A function declared `@ppy.pure` performs a forbidden effect.",
    "E1602": "A function declared `@ppy.pure` calls a function with unknown effects.",
    "E1611": "A function returns a parameter it only borrows.",
    "E1612": "A function stores a borrowed parameter where it outlives the call.",
    "E1613": "A function writes through a parameter it borrows read-only.",
    "E1630": "A `ppy.native` operation was given something that is not a native pointer.",
    "E1631": "A write through a `ppy.native.const_ptr`.",
    "E1632": "A C export needs a signature C can spell, with one scalar result.",
    "E1633": "A C binding needs every parameter and its return annotated.",
    "E1640": "A `ppy.simd` operation is misused; the message says how.",
    "E1641": "A `ppy.atomic` operation is misused; the message says how.",
    "E1642": "A `ppy.concurrent` operation is misused; the message says how.",
    "E1643": "A `ppy.cpu` operation is misused; the message says how.",
    "E1644": "A `ppy.cuda` or `ppy.hip` operation is misused; the message says how.",
    "E1645": "A `ppy.aio` operation is misused; the message says how.",
    "E1650": "A `ppy.parallel` loop is misused; the message says how.",
    "E1660": "`ppy.grad` or `ppy.value_and_grad` is misused; the message says how.",
    "E1661": (
        "A function cannot be differentiated as typed: it must return `float`, and `argnums` "
        "names `float` parameters."
    ),
    "E1662": (
        "A function with an effect no derivative follows -- I/O, a write, a thread -- is "
        "differentiated."
    ),
    "E1701": "`@ppy.parallel(require=True)` could not be satisfied for the selected backend.",
    "E1702": "`@ppy.native(require=True)` could not be satisfied without an opaque Python call.",
    "E1720": "A type parameter form other than `T` or `T: Bound`.",
    "E1721": "A type argument does not satisfy its parameter's bound.",
    "E1722": "A generic is specialized more, or on a deeper type, than the project allows.",
    "E1723": "A generic calls itself with its own type parameter nested in a type.",
    "E1801": "The selected backend is unavailable in this environment.",
    "E1802": "A construct is not supported by the selected backend.",
    "E1803": "A standalone build requires a fully native reachable graph.",
    "E1804": "A header-only unit cannot carry runtime state; the feature needing it is named.",
    "E1805": "A library build has nothing to export; `@ppy.native.export` names what to publish.",
    "E1806": "A C header could not be imported; the reason is named.",
    "E1901": "A plugin the project asked for could not be loaded, or two plugins claim one module.",
    "E1902": "A plugin's compiler pass left the IR invalid; the pass is named.",
    "E1903": (
        "A backend could not be used: unknown, registered twice, written against another "
        "interface version, or not loadable; the reason is named."
    ),
    "E1904": "A backend's compiler pass left the IR invalid; the pass is named.",
    "W2001": "A module is shadowed by a same-named source with a different extension.",
    "W2002": "A `bool` value takes part in arithmetic, which is legal but usually unintended.",
    "W2003": "Unknown `Annotated` metadata was preserved but not interpreted.",
    "W2005": "Conversion left both a .py and a .ppy source for the same module.",
    "W2006": (
        "Errors that only restated a type the analysis could not resolve were withheld; "
        "the count and the unresolved origins are reported once."
    ),
    "W2004": "A directive had no effect for the selected backend.",
    "W2007": "A function marked `@ppy.xla.jit` cannot be taken by XLA; the reason is named.",
    "W2008": "A kernel will not run on the device; the reason is named, and the reference runs.",
    "W2009": (
        "A function changed since the profile given to `--pgo` was recorded; its counts were "
        "ignored and it was built as without a profile."
    ),
    "R3003": "A list parameter is close to being a borrowed buffer but something blocks it.",
    "W2101": (
        "The build cache index was damaged; it was quarantined and rebuilt, "
        "and compilation continued with cache misses."
    ),
    "E9001": "An internal analysis failed to converge; the result cannot be trusted.",
    "R3002": "The converter promoted a list parameter to a borrowed buffer.",
    "R3004": (
        "A native-eligible function stays on the Python boundary; "
        "crossing costs more than it saves."
    ),
    "R3001": "An optimization remark.",
}


def describe(code: str) -> str | None:
    return CODES.get(code.upper())
