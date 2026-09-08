"""`ppy.xla`: a function's arithmetic compiled by XLA and run on a PJRT device.

```python
from ppy import xla

@xla.jit
def f(x: float, y: float) -> float:
    return math.sin(x) * y + x * x

xla.devices()        # ["cpu:0"] where a device is present
```

`@xla.jit` marks a function of floats, ints, and bools whose body is one
block of arithmetic and math. The compiler lowers it to the IR, emits
StableHLO, and the build stages it; at run time the staged module is
compiled by XLA once and each call runs on the device. Under plain CPython
-- and wherever no device is present -- the function runs as written: the
directive is inert, like every `ppy` directive (spec 61, 64). `compile(f)`
is the same as `jit(f)`; `devices()`, `default_device()`, and
`device_put()` ask the PJRT bridge, and answer nothing, `None`, and the
value itself when there is no bridge to ask.
"""

from __future__ import annotations

from typing import Any

from ._directives import _flexible

__all__ = ["compile", "default_device", "device_put", "devices", "jit"]

jit = _flexible("xla.jit")
jit.__qualname__ = "ppy.xla.jit"


def compile(function: Any) -> Any:  # pylint: disable=redefined-builtin
    """`jit`, as a call: the function marked for XLA."""
    return jit(function)


def _bridge():  # type: ignore[no-untyped-def]
    try:
        from ppy_runtime.xla import pjrt
    except ImportError:
        return None
    if not pjrt.available():
        return None
    try:
        return pjrt.client()
    except Exception:  # noqa: BLE001 - no device is an answer, not an error
        return None


def devices() -> list[str]:
    """The XLA devices a program can run on; none without the bridge."""
    bridge = _bridge()
    return [] if bridge is None else bridge.devices()


def default_device() -> str | None:
    found = devices()
    return found[0] if found else None


def device_put(value: Any) -> Any:
    """`value` placed on the default device, or itself when there is none."""
    bridge = _bridge()
    if bridge is None:
        return value
    import numpy

    return bridge._jax.device_put(numpy.asarray(value))
