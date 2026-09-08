"""Running a compiled `ppy.xla` function: the payload the build staged, on a device.

The payload is what the compiler wrote for an `@ppy.xla.jit` function: the
StableHLO module and the kinds of its parameters and results. `runtime_call`
turns it into a callable that places the arguments on the device, runs the
executable, and hands back Python values -- a `float` for a `float`, an
array for a tensor -- so the program sees what the Python path would have
given it (spec 64, 65, 66).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .pjrt import available, client

__all__ = ["KIND", "available", "runtime_call"]

#: The payload's `kind`, distinguishing it from a JAX export.
KIND = "ppy.xla"


def runtime_call(payload: bytes) -> Callable[..., Any]:
    """A callable running the compiled function `payload` describes."""
    import numpy

    described = json.loads(payload.decode("utf-8"))
    if described.get("kind") != KIND:
        raise ValueError("not a ppy.xla payload")
    text = described["stablehlo"]
    params = described["params"]
    results = described["results"]
    executable = client().compile(text)
    dtypes = {"float": numpy.float64, "int": numpy.int64, "bool": numpy.bool_}

    def call(*arguments: Any) -> Any:
        if len(arguments) != len(params):
            raise TypeError(f"expected {len(params)} argument(s), got {len(arguments)}")
        placed = []
        for value, kind in zip(arguments, params, strict=True):
            if kind in dtypes:
                placed.append(numpy.asarray(value, dtype=dtypes[kind]))
            else:
                placed.append(numpy.asarray(value))
        outputs = client().execute(executable, placed)
        values = []
        for output, kind in zip(outputs, results, strict=True):
            if kind == "float":
                values.append(float(output))
            elif kind == "int":
                values.append(int(output))
            elif kind == "bool":
                values.append(bool(output))
            else:
                values.append(output)
        return values[0] if len(values) == 1 else tuple(values)

    return call
