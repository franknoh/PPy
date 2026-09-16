"""Analyses and integer models for standalone source programs."""

from __future__ import annotations

from ...ir import (
    BoolType,
    BufferType,
    FloatType,
    IntType,
    IRFunction,
    IRModule,
    IRType,
    PtrType,
)


def narrow_integers(module: IRModule) -> None:
    """Select i32/u32 before shared optimization, without changing foreign ABIs.

    Only scalar programs, local slots, and literal strings participate. Other
    storage and runtime interfaces retain their fixed widths and are refused.
    """
    externs = {
        "ppy_rt_input_int",
        "ppy_rt_scan_int",
        "ppy_rt_print_i64",
        "ppy_rt_print_u64",
        "ppy_rt_print_bool",
        "ppy_rt_print_str",
        "ppy_rt_print_nl",
        "ppy_rt_print_sep",
        "ppy_rt_flush_stdout",
    }
    operations = {
        "const",
        "alloca",
        "load",
        "store",
        "add",
        "sub",
        "mul",
        "neg",
        "div",
        "mod",
        "and",
        "or",
        "xor",
        "shl",
        "shr",
        "cmp",
        "select",
        "cast",
        "br",
        "cond_br",
        "ret",
        "guard",
        "call",
        "call_extern",
        "call_intrinsic",
    }

    def refuse(detail: str) -> None:
        raise ValueError(f"32-bit standalone source: {detail}; use --int-width 64")

    def narrowed(t: IRType) -> IRType:
        if isinstance(t, IntType):
            return IntType(32, t.signed) if t.width == 64 else t
        if isinstance(t, (BoolType, FloatType)):
            return t
        if isinstance(t, PtrType) and t.address_space == "stack":
            return PtrType(narrowed(t.pointee), t.address_space, t.mutable)
        if t == PtrType(IntType(8, False)):
            return t  # A literal string's data, never integer storage.
        refuse(f"{t} has a fixed storage ABI")
        return t

    # Validate everything before rewriting any type.
    for item in module.globals.values():
        if item.type != BufferType(IntType(8, False)) or not isinstance(item.value, str):
            refuse("only literal string globals are supported")
    for function in module.functions.values():
        if function.is_declaration or function.attributes.get("ppy.abi") == "resume":
            refuse(f"@{function.name} has an external or asynchronous ABI")
        if len(function.results) > 1:
            refuse(f"@{function.name} returns multiple values")
        for _, t in function.params:
            if isinstance(t, PtrType):
                refuse("pointer parameters have a fixed storage ABI")
            narrowed(t)
        for t in function.results:
            if isinstance(t, PtrType):
                refuse("pointer results have a fixed storage ABI")
            narrowed(t)
        for op in function.operations():
            if op.dialect != "core" or op.local_name not in operations or op.regions:
                refuse(f"{op.name} requires a fixed runtime ABI")
            if op.name == "core.call_extern" and op.attributes.get("callee") not in externs:
                refuse(f"{op.attributes.get('callee')} requires a fixed runtime ABI")
            if op.name == "core.call" and op.attributes.get("capture_status"):
                refuse("captured call status requires the runtime ABI")
            if (
                op.name == "core.call_intrinsic"
                and op.attributes.get("intrinsic") != "ppy.string_data"
            ):
                refuse(f"{op.attributes.get('intrinsic')} requires a fixed runtime ABI")
            for value in op.results:
                narrowed(value.type)
            if op.name == "core.const" and isinstance(op.result.type, IntType):
                t = narrowed(op.result.type)
                assert isinstance(t, IntType)
                number = op.attributes["value"]
                assert isinstance(number, int)
                if not t.fits(number):
                    refuse(f"constant {number} is outside {t}'s range")
    for function in module.functions.values():
        function.params = tuple((name, narrowed(t)) for name, t in function.params)
        function.results = tuple(narrowed(t) for t in function.results)
        for block in function.blocks():
            for value in block.arguments:
                value.type = narrowed(value.type)
            for op in block.operations:
                for value in op.results:
                    value.type = narrowed(value.type)
                for key, value in op.attributes.items():
                    if isinstance(value, IRType):
                        op.attributes[key] = narrowed(value)
    module.attributes["ppy.source_int_width"] = 32


def definition_order(functions: list[IRFunction]) -> tuple[list[IRFunction], set[str]]:
    """Callees first, with prototypes only for cycles or callback ABI users."""
    by_name = {function.name: function for function in functions}
    dependencies: dict[str, list[str]] = {}
    prototypes: set[str] = set()
    for function in functions:
        names = []
        for op in function.operations():
            callee = getattr(op.attributes.get("callee"), "name", None)
            if callee in by_name:
                if callee not in names:
                    names.append(callee)
                if op.name != "core.call":
                    prototypes.add(callee)
        dependencies[function.name] = names
    ordered = []
    active: set[str] = set()
    done: set[str] = set()

    def visit(name: str) -> None:
        if name in active:
            prototypes.add(name)
            return
        if name in done:
            return
        active.add(name)
        for callee in dependencies[name]:
            visit(callee)
        active.remove(name)
        done.add(name)
        ordered.append(by_name[name])

    for function in functions:
        visit(function.name)
    return ordered, prototypes
