"""The prof dialect in C: a file-static counter array the instrumented functions add to."""

from __future__ import annotations

import json

from ...ir import Operation
from ...ir.transforms.profile import PROFILE_MAP, counters_identifier
from .dialects import EmitError

__all__ = ["emit_prof"]


def emit_prof(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    owner = fe.owner
    legend = owner.module.attributes.get(PROFILE_MAP)
    if not legend:
        raise EmitError(
            f"{op.name} in a module without a profile legend; `instrument-profile` places both"
        )
    ident = counters_identifier(owner.module.name)
    count = max(int(json.loads(str(legend))["counters"]), 1)
    owner.unit.prelude[f"prof:{ident}"] = f"static int64_t __ppy_prof_counters_{ident}[{count}];"
    index = int(op.attributes["counter"])
    if op.local_name == "hit":
        fe.body.append(f"    __ppy_prof_counters_{ident}[{index}] += 1;")
    elif op.local_name == "hit_if":
        fe.body.append(
            f"    __ppy_prof_counters_{ident}[{index}] += ({fe.value(op.operands[0])}) ? 1 : 0;"
        )
    else:
        raise EmitError(f"{op.name} has no C lowering")
