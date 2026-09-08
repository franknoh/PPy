"""The prof dialect and profile attributes in LLVM (spec 84).

An instrumented module gets a counter array and its legend as globals --
`__ppy_prof_counters_<module>` and `__ppy_prof_map_<module>` -- and each
`prof.hit` is an atomic add on one slot, so counting is right under the
threads a parallel loop runs. A profile-guided module writes its counts
where LLVM reads them: `function_entry_count` and `branch_weights` as
`!prof` metadata, and `hot` and `cold` as function attributes.
"""

from __future__ import annotations

import json

from ...ir.transforms.profile import counters_identifier
from .dialect_lowerings import EmitError

__all__ = ["branch_weights", "function_profile", "lower_prof", "profile_globals"]

COUNTERS = "__ppy_prof_counters_"
LEGEND = "__ppy_prof_map_"


def profile_globals(emitter, legend_text: str) -> None:  # type: ignore[no-untyped-def]
    """The counter array and its legend, for a module the instrument pass marked."""
    ir = emitter.ir
    table = json.loads(legend_text)
    ident = counters_identifier(emitter.module.name)
    count = max(int(table["counters"]), 1)
    array = ir.ArrayType(ir.IntType(64), count)
    counters = ir.GlobalVariable(emitter.llvm, array, name=COUNTERS + ident)
    counters.initializer = ir.Constant(array, None)
    data = bytearray(legend_text.encode("utf-8")) + b"\0"
    legend_type = ir.ArrayType(ir.IntType(8), len(data))
    legend = ir.GlobalVariable(emitter.llvm, legend_type, name=LEGEND + ident)
    legend.global_constant = True
    legend.initializer = ir.Constant(legend_type, data)
    emitter.profile_counters = counters


def lower_prof(emitter, op) -> None:  # type: ignore[no-untyped-def]
    counters = getattr(emitter.owner, "profile_counters", None)
    if counters is None:
        raise EmitError(
            f"{op.name} in a module without a profile legend; `instrument-profile` places both"
        )
    ir = emitter.ir
    b = emitter.builder
    i32 = ir.IntType(32)
    i64 = ir.IntType(64)
    slot = b.gep(counters, [ir.Constant(i32, 0), ir.Constant(i32, int(op.attributes["counter"]))])
    if op.local_name == "hit":
        amount = ir.Constant(i64, 1)
    elif op.local_name == "hit_if":
        amount = b.zext(emitter.value(op.operands[0]), i64)
    else:
        raise EmitError(f"{op.name} has no LLVM lowering")
    b.atomic_rmw("add", slot, amount, "monotonic")


def branch_weights(emitter, instruction, weights) -> None:  # type: ignore[no-untyped-def]
    """`!prof branch_weights` from (taken, not taken) counts; never a zero, never past 32 bits."""
    ir = emitter.ir
    taken, not_taken = (max(int(w), 0) + 1 for w in weights)
    limit = 2**31 - 1
    largest = max(taken, not_taken)
    if largest > limit:
        taken = max(taken * limit // largest, 1)
        not_taken = max(not_taken * limit // largest, 1)
    i32 = ir.IntType(32)
    instruction.set_metadata(
        "prof",
        emitter.owner.llvm.add_metadata(
            ["branch_weights", ir.Constant(i32, taken), ir.Constant(i32, not_taken)]
        ),
    )


def function_profile(emitter, declared, function) -> None:  # type: ignore[no-untyped-def]
    """The entry count and the hotness of a function the profile measured."""
    ir = emitter.ir
    calls = function.attributes.get("ppy.profile.calls")
    if isinstance(calls, int) and not isinstance(calls, bool):
        declared.set_metadata(
            "prof",
            emitter.llvm.add_metadata(["function_entry_count", ir.Constant(ir.IntType(64), calls)]),
        )
    if function.attributes.get("ppy.profile.hot"):
        # llvmlite's list of attributes predates `hot`; the set behind its
        # check takes the spelling LLVM knows.
        set.add(declared.attributes, "hot")
    if function.attributes.get("ppy.profile.cold"):
        declared.attributes.add("cold")
