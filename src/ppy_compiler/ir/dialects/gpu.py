"""The gpu dialect: one execution model every GPU backend meets (spec 71).

A function is `host` -- the default -- `device`, or `kernel`, by its
`gpu.kind` attribute. A kernel returns nothing and is launched from the
host over a grid of blocks of threads; a device function is called from a
kernel or another device function. A thread learns where it is from
`thread_id`, `block_id`, `block_dim`, and `grid_dim`, each `.x`, `.y`, or
`.z`; waits for its block at `barrier` and for its subgroup at
`subgroup_barrier`; trades a value across the subgroup with
`subgroup_shuffle`; and takes memory of its own in the `shared` and
`private` spaces with `shared_alloc` and `private_alloc`. Memory it is
handed is `ptr<T, global>` or `ptr<T, constant>`, and atomics on it are the
atomic dialect's. The host launches with `gpu.launch @kernel(%args...)`
after the sizes of the grid and the block.

The verifier holds what each kind may contain: a device operation in a
host function is refused, as is a host operation in device code -- a guard,
which falls back to Python where there is none; a buffer, which is a host
object; a call to a host function; another dialect's operation -- and a
kernel that returns, or takes stack memory. The CUDA and HIP frontends
lower to this dialect and the source, NVVM, and PTX backends lower from it;
the CPU backends leave device code alone (spec 72-75).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Block, Builder, IRFunction, Operation, Region, SymbolRef, Value
from ..types import INDEX, IRType, PtrType, is_integer, is_scalar

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "ADDRESS_SPACES",
    "DEVICE_DIALECTS",
    "DIMENSIONS",
    "HOST_ONLY",
    "KINDS",
    "SHUFFLE_MODES",
    "GpuDialect",
    "barrier",
    "block_dim",
    "block_id",
    "grid_dim",
    "kind_of",
    "launch",
    "mark",
    "private_alloc",
    "shared_alloc",
    "subgroup_barrier",
    "subgroup_shuffle",
    "thread_id",
]

KINDS = ("host", "device", "kernel")
DIMENSIONS = ("x", "y", "z")
SHUFFLE_MODES = ("idx", "up", "down", "xor")
ADDRESS_SPACES = frozenset({"global", "shared", "private", "constant"})
#: Where a kernel's pointer parameters may point.
PARAMETER_SPACES = frozenset({"global", "constant", "generic"})
#: The dialects device code is written in; every other is the host's.
DEVICE_DIALECTS = frozenset({"core", "math", "atomic", "gpu", "simd"})
#: Core operations with no device form, and why.
HOST_ONLY = {
    "core.guard": "a guard falls back to Python, and a device has none",
    "core.call_extern": "device code calls device functions, not C",
    "core.call_intrinsic": "an intrinsic is the host runtime's",
    "core.buffer_data": "a buffer is a host object; a kernel is handed a pointer and a length",
    "core.buffer_len": "a buffer is a host object; a kernel is handed a pointer and a length",
    "core.buffer_load": "a buffer is a host object; a kernel is handed a pointer and a length",
    "gpu.device_alloc": "device memory is made by the host and handed to a launch",
    "core.buffer_store": "a buffer is a host object; a kernel is handed a pointer and a length",
}
#: The operations that say where a thread is; each is `.x`, `.y`, or `.z`.
POSITIONS = ("thread_id", "block_id", "block_dim", "grid_dim")


def kind_of(function: IRFunction) -> str:
    """`host`, `device`, or `kernel`: what the function's `gpu.kind` says."""
    return str(function.attributes.get("gpu.kind", "host"))


def mark(function: IRFunction, kind: str) -> IRFunction:
    """`function` as a `kind` of function; `host` is the absence of the mark."""
    if kind not in KINDS:
        raise ValueError(f"a function is one of {', '.join(KINDS)}, not {kind!r}")
    if kind == "host":
        function.attributes.pop("gpu.kind", None)
    else:
        function.attributes["gpu.kind"] = kind
    return function


# -- verification ----------------------------------------------------------------


def _verify_position(op: Operation, checker: Checker) -> None:
    if op.results[0].type != INDEX:
        checker.error(op, f"{op.name} is an index, not {op.results[0].type}")


def _verify_alloc(space: str):  # type: ignore[no-untyped-def]
    def verify(op: Operation, checker: Checker) -> None:
        result = op.results[0].type
        if not isinstance(result, PtrType) or result.address_space != space:
            checker.error(op, f"{op.name} yields ptr<T, {space}>, not {result}")
            return
        if not is_scalar(result.pointee):
            checker.error(op, f"{op.name} holds scalars, not {result.pointee}")
        count = op.attributes.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            checker.error(op, f"`count` is a positive integer, not {count!r}")

    return verify


def _verify_shuffle(op: Operation, checker: Checker) -> None:
    value, lane = op.operands
    if not is_scalar(value.type):
        checker.error(op, f"subgroup_shuffle moves a scalar, not {value.type}")
    if not is_integer(lane.type):
        checker.error(op, f"a lane is an integer, not {lane.type}")
    if op.results[0].type != value.type:
        checker.error(op, f"subgroup_shuffle gives {value.type}, not {op.results[0].type}")


def _verify_device_alloc(op: Operation, checker: Checker) -> None:
    result = op.results[0].type
    if not isinstance(result, PtrType) or result.address_space != "generic" or not result.mutable:
        checker.error(op, f"{op.name} yields a mutable ptr<T, generic>, not {result}")
        return
    if not is_scalar(result.pointee):
        checker.error(op, f"{op.name} holds scalars, not {result.pointee}")
    if not is_integer(op.operands[0].type):
        checker.error(op, f"a count is an integer, not {op.operands[0].type}")


def _verify_launch(op: Operation, checker: Checker) -> None:
    callee = op.attributes.get("callee")
    if not isinstance(callee, SymbolRef):
        checker.error(op, "launch needs a `callee` symbol")
        return
    if len(op.operands) < 6:
        checker.error(
            op, "launch takes the grid and the block -- six integers -- before the arguments"
        )
        return
    for size in op.operands[:6]:
        if not is_integer(size.type):
            checker.error(op, f"a grid or block size is an integer, not {size.type}")
    module = checker.module
    if module is None:
        return
    target = module.functions.get(callee.name)
    if target is None:
        checker.error(op, f"launch of undefined @{callee.name}")
        return
    if kind_of(target) != "kernel":
        checker.error(
            op, f"@{callee.name} is a {kind_of(target)} function; only a kernel is launched"
        )
        return
    given = tuple(v.type for v in op.operands[6:])
    expected = tuple(t for _name, t in target.params)
    if given != expected:
        checker.error(
            op,
            f"@{callee.name} takes ({', '.join(map(str, expected))}), "
            f"launched with ({', '.join(map(str, given))})",
        )


def _operations(region: Region) -> Iterator[tuple[Block, Operation]]:
    for block in region.blocks:
        for op in block.operations:
            yield block, op
            for inner in op.regions:
                yield from _operations(inner)


class GpuDialect(Dialect):
    name = "gpu"
    version = 1

    def address_spaces(self) -> frozenset[str]:
        return ADDRESS_SPACES

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        for name in POSITIONS:
            add(
                OpSpec(
                    f"gpu.{name}",
                    pure=True,
                    variant_attribute="dim",
                    variants=DIMENSIONS,
                    verify=_verify_position,
                    operands=0,
                    results=1,
                )
            )
        add(OpSpec("gpu.barrier", operands=0, results=0))
        add(OpSpec("gpu.subgroup_barrier", operands=0, results=0))
        for space in ("shared", "private"):
            add(
                OpSpec(
                    f"gpu.{space}_alloc",
                    verify=_verify_alloc(space),
                    operands=0,
                    results=1,
                    required_attributes=("count",),
                )
            )
        add(
            OpSpec(
                "gpu.device_alloc",
                verify=_verify_device_alloc,
                operands=1,
                results=1,
            )
        )
        add(
            OpSpec(
                "gpu.subgroup_shuffle",
                variant_attribute="mode",
                variants=SHUFFLE_MODES,
                verify=_verify_shuffle,
                operands=2,
                results=1,
            )
        )
        add(
            OpSpec(
                "gpu.launch",
                verify=_verify_launch,
                operands=None,
                results=0,
                required_attributes=("callee",),
            )
        )

    def verify_function(self, function: IRFunction, checker: Checker) -> None:
        kind = kind_of(function)
        if kind not in KINDS:
            checker.error(None, f"`gpu.kind` is one of {', '.join(KINDS)}, not {kind!r}")
            return
        if kind == "kernel":
            if function.results:
                checker.error(None, "a kernel returns nothing; its results are global memory")
            for name, t in function.params:
                if isinstance(t, PtrType):
                    if t.address_space not in PARAMETER_SPACES:
                        checker.error(
                            None, f"parameter %{name} is {t}; a kernel is handed global memory"
                        )
                elif not is_scalar(t):
                    checker.error(
                        None, f"parameter %{name} is {t}; a kernel takes scalars and pointers"
                    )
        if function.is_declaration:
            return
        for block, op in _operations(function.body):
            checker.block = block
            if kind == "host":
                self._in_host(op, checker)
            else:
                self._in_device(op, kind, checker)
        checker.block = None

    def _in_host(self, op: Operation, checker: Checker) -> None:
        if op.dialect == "gpu" and op.local_name not in {"launch", "device_alloc"}:
            checker.error(op, f"{op.name} runs on a device; this is a host function")
        elif op.name == "core.call":
            target = self._callee(op, checker)
            if target is not None and kind_of(target) != "host":
                checker.error(
                    op,
                    f"@{target.name} is a {kind_of(target)} function; a kernel is launched, "
                    "and a device function is called from device code",
                )

    def _in_device(self, op: Operation, kind: str, checker: Checker) -> None:
        if op.name == "gpu.launch":
            checker.error(op, "a kernel is launched from the host, not from device code")
        elif op.dialect not in DEVICE_DIALECTS:
            checker.error(
                op,
                f"{op.name} is a host operation; a {kind} function is "
                "core, math, atomic, gpu, simd",
            )
        elif op.name in HOST_ONLY:
            checker.error(op, f"{op.name} has no device form: {HOST_ONLY[op.name]}")
        elif op.name == "core.call":
            target = self._callee(op, checker)
            if target is not None and kind_of(target) != "device":
                checker.error(
                    op,
                    f"@{target.name} is a {kind_of(target)} function; "
                    "device code calls device functions",
                )

    @staticmethod
    def _callee(op: Operation, checker: Checker) -> IRFunction | None:
        callee = op.attributes.get("callee")
        if checker.module is None or not isinstance(callee, SymbolRef):
            return None
        return checker.module.functions.get(callee.name)


# -- builders --------------------------------------------------------------------


def _position(b: Builder, what: str, dim: str, name: str | None) -> Value:
    if dim not in DIMENSIONS:
        raise ValueError(f"a dimension is one of {', '.join(DIMENSIONS)}, not {dim!r}")
    return b.create(f"gpu.{what}", (), (INDEX,), {"dim": dim}, result_names=(name,)).result


def thread_id(b: Builder, dim: str = "x", name: str | None = None) -> Value:
    """This thread's index within its block along `dim`."""
    return _position(b, "thread_id", dim, name)


def block_id(b: Builder, dim: str = "x", name: str | None = None) -> Value:
    """This block's index within the grid along `dim`."""
    return _position(b, "block_id", dim, name)


def block_dim(b: Builder, dim: str = "x", name: str | None = None) -> Value:
    """How many threads a block has along `dim`."""
    return _position(b, "block_dim", dim, name)


def grid_dim(b: Builder, dim: str = "x", name: str | None = None) -> Value:
    """How many blocks the grid has along `dim`."""
    return _position(b, "grid_dim", dim, name)


def barrier(b: Builder) -> Operation:
    """Every thread of the block arrives before any leaves; shared memory is visible."""
    return b.create("gpu.barrier")


def subgroup_barrier(b: Builder) -> Operation:
    return b.create("gpu.subgroup_barrier")


def shared_alloc(b: Builder, t: IRType, count: int, name: str | None = None) -> Value:
    """`count` elements of `t` in the block's shared memory."""
    return b.create(
        "gpu.shared_alloc", (), (PtrType(t, "shared"),), {"count": count}, result_names=(name,)
    ).result


def private_alloc(b: Builder, t: IRType, count: int, name: str | None = None) -> Value:
    """`count` elements of `t` private to this thread."""
    return b.create(
        "gpu.private_alloc", (), (PtrType(t, "private"),), {"count": count}, result_names=(name,)
    ).result


def device_alloc(b: Builder, t: IRType, count: Value, name: str | None = None) -> Value:
    """`count` elements of `t` in device memory the host may also read and write."""
    return b.create(
        "gpu.device_alloc", (count,), (PtrType(t, "generic", True),), result_names=(name,)
    ).result


def subgroup_shuffle(
    b: Builder, value: Value, lane: Value, mode: str = "idx", name: str | None = None
) -> Value:
    """`value` as another lane of the subgroup holds it: lane `lane`, or this
    lane offset `up`, `down`, or `xor` by it."""
    if mode not in SHUFFLE_MODES:
        raise ValueError(f"a shuffle is one of {', '.join(SHUFFLE_MODES)}, not {mode!r}")
    return b.create(
        "gpu.subgroup_shuffle", (value, lane), (value.type,), {"mode": mode}, result_names=(name,)
    ).result


def launch(
    b: Builder,
    callee: str,
    grid: tuple[Value, Value, Value],
    block: tuple[Value, Value, Value],
    arguments: tuple[Value, ...] = (),
) -> Operation:
    """Run kernel `@callee` over `grid` blocks of `block` threads with `arguments`."""
    return b.create("gpu.launch", (*grid, *block, *arguments), (), {"callee": SymbolRef(callee)})
