"""The gpu dialect spelled as CUDA and as HIP, for the source backends (spec 74).

A thread's position is `threadIdx.x` and its kin, a block barrier
`__syncthreads()`, shared memory a `__shared__` array declared in the
kernel, private memory a local array, a subgroup shuffle `__shfl_sync` on
CUDA and `__shfl` on HIP, a launch the triple chevron followed by a
device synchronization whose status is the host function's, and device
memory a managed allocation freed with the host function. One IR is
written both ways; nothing here is vendor-specific beyond the spelling.
"""

from __future__ import annotations

from ...ir import BoolType, Operation, PtrType
from .dialects import EmitError

__all__ = ["emit_gpu"]

_POSITIONS = {
    "thread_id": "threadIdx",
    "block_id": "blockIdx",
    "block_dim": "blockDim",
    "grid_dim": "gridDim",
}
_SHUFFLES = {
    "cuda": {
        "idx": "__shfl_sync(0xffffffffu, {v}, (int){lane})",
        "up": "__shfl_up_sync(0xffffffffu, {v}, (unsigned){lane})",
        "down": "__shfl_down_sync(0xffffffffu, {v}, (unsigned){lane})",
        "xor": "__shfl_xor_sync(0xffffffffu, {v}, (int){lane})",
    },
    "hip": {
        "idx": "__shfl({v}, (int){lane})",
        "up": "__shfl_up({v}, (unsigned){lane})",
        "down": "__shfl_down({v}, (unsigned){lane})",
        "xor": "__shfl_xor({v}, (int){lane})",
    },
}
_SYNCHRONIZE = {
    "cuda": ("cudaDeviceSynchronize", "cudaSuccess", "__syncwarp()"),
    "hip": ("hipDeviceSynchronize", "hipSuccess", "__builtin_amdgcn_wave_barrier()"),
}
#: Device memory a host function makes: managed, so the host reads and writes
#: it as the reference does, and freed with the function, however it leaves.
_DEVICE_MEMORY = """template <typename T> struct ppy_device_memory {{
    T *data = nullptr;
    bool allocate(int64_t count) {{
        size_t bytes = count > 0 ? (size_t)count * sizeof(T) : 1;
        return {malloc}((void **)&data, bytes) == {success};
    }}
    ~ppy_device_memory() {{
        if (data != nullptr) {{
            {free}(data);
        }}
    }}
}};
"""
_MEMORY_CALLS = {
    "cuda": ("cudaMallocManaged", "cudaSuccess", "cudaFree"),
    "hip": ("hipMallocManaged", "hipSuccess", "hipFree"),
}


def emit_gpu(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    owner = fe.owner
    if not owner.gpu:
        raise EmitError(f"{op.name} is device code; `ppy emit cuda` or `ppy emit hip` writes it")
    api = str(owner.language)
    name = op.local_name
    if name in _POSITIONS:
        fe.define(op.result, f"(int64_t){_POSITIONS[name]}.{op.attributes['dim']}")
    elif name == "barrier":
        fe.body.append("    __syncthreads();")
    elif name == "subgroup_barrier":
        fe.body.append(f"    {_SYNCHRONIZE[api][2]};")
    elif name in {"shared_alloc", "private_alloc"}:
        pointer = op.result.type
        assert isinstance(pointer, PtrType)
        slot = fe.fresh(op.result.name or name.partition("_")[0])
        count = int(op.attributes["count"])  # type: ignore[call-overload]
        qualifier = "__shared__ " if name == "shared_alloc" else ""
        fe.declarations.append(f"    {qualifier}{owner.c_type(pointer.pointee)} {slot}[{count}];")
        fe.scalars[id(op.result)] = slot
    elif name == "subgroup_shuffle":
        value, lane = (fe.value(v) for v in op.operands)
        spelled = _SHUFFLES[api][str(op.attributes["mode"])]
        if isinstance(op.result.type, BoolType):
            fe.define(op.result, "(bool)" + spelled.format(v=f"(int){value}", lane=lane))
        else:
            fe.define(op.result, spelled.format(v=value, lane=lane))
    elif name == "device_alloc":
        pointer = op.result.type
        assert isinstance(pointer, PtrType)
        malloc, success, free = _MEMORY_CALLS[api]
        owner.unit.prelude.setdefault(
            "device_memory", _DEVICE_MEMORY.format(malloc=malloc, success=success, free=free)
        )
        slot = fe.fresh(op.result.name or "device")
        fe.declarations.append(f"    ppy_device_memory<{owner.c_type(pointer.pointee)}> {slot};")
        fe.fail_unless(f"{slot}.allocate({fe.bare(op.operands[0])})", "device_alloc.ok")
        fe.scalars[id(op.result)] = f"{slot}.data"
    elif name == "launch":
        _launch(fe, op, api)
    else:
        raise EmitError(f"{op.name} has no {api.upper()} lowering")


def _launch(fe, op: Operation, api: str) -> None:  # type: ignore[no-untyped-def]
    callee = op.attributes["callee"].name  # type: ignore[union-attr]
    target = fe.owner.module.functions.get(callee)
    if target is None:
        raise EmitError(f"launch of @{callee}, which was not emitted")
    sizes = [fe.value(v) for v in op.operands[:6]]
    arguments = [atom for v in op.operands[6:] for atom in fe.flatten(v)]
    grid = ", ".join(f"(unsigned){s}" for s in sizes[:3])
    block = ", ".join(f"(unsigned){s}" for s in sizes[3:])
    symbol = fe.owner.symbol_of(target)
    fe.body.append(f"    {symbol}<<<dim3({grid}), dim3({block})>>>({', '.join(arguments)});")
    synchronize, success, _ = _SYNCHRONIZE[api]
    fe.fail_unless(f"{synchronize}() == {success}", "launch.ok")
