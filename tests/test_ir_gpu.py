"""The gpu dialect: kinds, address spaces, and what device code may hold."""

from __future__ import annotations

import ctypes

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.ir import (
    F64,
    I64,
    INDEX,
    Builder,
    IRModule,
    PtrType,
    Successor,
    decode,
    encode,
    verify,
)
from ppy_compiler.ir.dialects import atomic, core, gpu, tensor
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

GLOBAL_F64 = PtrType(F64, "global")
CONST_F64 = PtrType(F64, "global", mutable=False)


def _module() -> IRModule:
    module = IRModule("kernels")
    module.require("gpu", 1)
    module.require("atomic", 1)
    return module


def _saxpy_module() -> IRModule:
    """`saxpy`: y[i] = a * x[i] + y[i] for i below n, one thread each, through
    the device function `fma`; `tally`: each thread parks x[tid] in shared
    memory, the block waits, and every lane below lane zero's value counts
    one into *count; `run`: the host launches `saxpy` over ceil(n / 256)
    blocks of 256 threads."""
    module = _module()
    fma = gpu.mark(
        module.add_function("fma", [("a", F64), ("x", F64), ("y", F64)], [F64]), "device"
    )
    b = Builder(fma.add_entry_block())
    a, x, y = fma.entry.arguments
    core.ret(b, core.add(b, core.mul(b, a, x), y))

    saxpy = module.add_function(
        "saxpy", [("n", I64), ("a", F64), ("x", CONST_F64), ("y", GLOBAL_F64)], []
    )
    gpu.mark(saxpy, "kernel")
    entry = saxpy.add_entry_block()
    body = saxpy.body.add_block("body")
    done = saxpy.body.add_block("done")
    b = Builder(entry)
    n, a, x, y = saxpy.entry.arguments
    position = core.add(
        b,
        core.mul(b, gpu.block_id(b, "x"), gpu.block_dim(b, "x"), overflow="wrap"),
        gpu.thread_id(b, "x"),
        overflow="wrap",
        name="i",
    )
    i = core.cast(b, position, I64)
    core.cond_br(b, core.cmp(b, "lt", i, n), Successor(body), Successor(done))
    b = Builder(body)
    slot = core.ptr_offset(b, y, i)
    summed = core.call(
        b, "fma", (a, core.load(b, core.ptr_offset(b, x, i)), core.load(b, slot)), (F64,)
    )
    core.store(b, summed.results[0], slot)
    core.br(b, Successor(done))
    core.ret(Builder(done))

    tally = module.add_function("tally", [("x", CONST_F64), ("count", PtrType(I64, "global"))], [])
    gpu.mark(tally, "kernel")
    b = Builder(tally.add_entry_block())
    x, count = tally.entry.arguments
    tid = gpu.thread_id(b, "x", name="tid")
    parked = gpu.shared_alloc(b, F64, 256, name="parked")
    mine = core.ptr_offset(b, parked, tid)
    core.store(b, core.load(b, core.ptr_offset(b, x, tid)), mine)
    gpu.barrier(b)
    value = core.load(b, mine)
    first = gpu.subgroup_shuffle(b, value, core.const(b, 0, INDEX), "idx", name="first")
    scratch = gpu.private_alloc(b, I64, 1, name="scratch")
    core.store(
        b,
        core.select(
            b, core.cmp(b, "lt", value, first), core.const(b, 1, I64), core.const(b, 0, I64)
        ),
        scratch,
    )
    atomic.fetch(b, "add", count, core.load(b, scratch), "relaxed")
    core.ret(b)

    run = module.add_function(
        "run", [("n", I64), ("a", F64), ("x", CONST_F64), ("y", GLOBAL_F64)], []
    )
    b = Builder(run.add_entry_block())
    n, a, x, y = run.entry.arguments
    one = core.const(b, 1, I64)
    threads = core.const(b, 256, I64)
    blocks = core.div(
        b, core.add(b, n, core.const(b, 255, I64), overflow="wrap"), threads, overflow="wrap"
    )
    gpu.launch(b, "saxpy", (blocks, one, one), (threads, one, one), (n, a, x, y))
    core.ret(b)
    return module


def _errors(module: IRModule) -> str:
    return "\n".join(str(error) for error in verify(module))


def test_a_kernel_module_verifies_and_round_trips():
    module = _saxpy_module()
    assert not verify(module), _errors(module)
    text = encode(module)
    assert 'attrs {gpu.kind = "kernel"}' in text and 'attrs {gpu.kind = "device"}' in text
    assert "gpu.thread_id.x : index" in text and "gpu.block_dim.x : index" in text
    assert (
        "ptr<f64, global, const>" in text
        and "gpu.shared_alloc {count = 256} : ptr<f64, shared>" in text
    )
    assert "gpu.subgroup_shuffle.idx %" in text and "gpu.barrier" in text
    assert "gpu.launch %" in text and "{callee = @saxpy}" in text
    stored = decode(text)
    assert encode(stored) == text
    assert not verify(stored)
    assert [gpu.kind_of(f) for f in stored.functions.values()] == [
        "device",
        "kernel",
        "kernel",
        "host",
    ]
    assert gpu.kind_of(module.functions["run"]) == "host"
    assert gpu.kind_of(gpu.mark(module.functions["fma"], "host")) == "host"


def _host_and_kernel() -> tuple[IRModule, Builder, Builder]:
    """A host function and a kernel, each with a builder at an open entry block."""
    module = _module()
    host = module.add_function("host", [("x", F64)], [F64])
    kernel = gpu.mark(module.add_function("kernel", [("p", GLOBAL_F64)], []), "kernel")
    return module, Builder(host.add_entry_block()), Builder(kernel.add_entry_block())


def test_a_device_operation_in_a_host_function_is_refused():
    module, host, kernel = _host_and_kernel()
    core.ret(host, core.cast(host, gpu.thread_id(host, "x"), F64))
    core.ret(kernel)
    assert "gpu.thread_id runs on a device; this is a host function" in _errors(module)


def test_host_operations_have_no_device_form():
    module, host, kernel = _host_and_kernel()
    core.ret(host, module.functions["host"].entry.arguments[0])
    p = module.functions["kernel"].entry.arguments[0]
    core.guard(
        kernel, core.cmp(kernel, "gt", core.load(kernel, p), core.const(kernel, 0.0, F64)), "bounds"
    )
    core.call_extern(kernel, "puts", (), ())
    tensor.fill(kernel, core.const(kernel, 1.0, F64), tensor.tensor_type(F64, (2,)))
    core.call(kernel, "host", (core.const(kernel, 1.0, F64),), (F64,))
    core.ret(kernel)
    errors = _errors(module)
    assert "core.guard has no device form: a guard falls back to Python" in errors
    assert "core.call_extern has no device form" in errors
    assert "tensor.fill is a host operation; a kernel function is core, math, atomic, gpu" in errors
    assert "@host is a host function; device code calls device functions" in errors


def test_a_kernel_is_launched_and_a_device_function_is_called():
    module, host, kernel = _host_and_kernel()
    x = module.functions["host"].entry.arguments[0]
    p = module.functions["kernel"].entry.arguments[0]
    device = gpu.mark(module.add_function("helper", [], []), "device")
    core.ret(Builder(device.add_entry_block()))
    core.call(host, "kernel", (core.alloca(host, F64),), ())
    core.call(host, "helper", (), ())
    one = core.const(host, 1, I64)
    gpu.launch(host, "helper", (one, one, one), (one, one, one))
    gpu.launch(host, "kernel", (one, one, one), (one, one, one), (x,))
    gpu.launch(host, "kernel", (one, one), (one, one, one))
    core.ret(host, x)
    unit = core.const(kernel, 1, I64)
    gpu.launch(kernel, "kernel", (unit, unit, unit), (unit, unit, unit), (p,))
    core.ret(kernel)
    errors = _errors(module)
    assert "@kernel is a kernel function; a kernel is launched" in errors
    assert "@helper is a device function; a kernel is launched" in errors
    assert "@helper is a device function; only a kernel is launched" in errors
    assert "@kernel takes (ptr<f64, global>), launched with (f64)" in errors
    assert "launch takes the grid and the block -- six integers" in errors
    assert "a kernel is launched from the host, not from device code" in errors


def test_a_kernel_signature_is_scalars_and_global_memory():
    module = _module()
    returning = gpu.mark(module.add_function("returning", [("x", F64)], [F64]), "kernel")
    b = Builder(returning.add_entry_block())
    core.ret(b, returning.entry.arguments[0])
    module.add_function(
        "shared_in", [("s", PtrType(F64, "shared"))], [], attributes={"gpu.kind": "kernel"}
    )
    module.add_function(
        "tensor_in", [("t", tensor.tensor_type(F64, (2,)))], [], attributes={"gpu.kind": "kernel"}
    )
    module.add_function("odd", [("x", F64)], [], attributes={"gpu.kind": "wavefront"})
    errors = _errors(module)
    assert "a kernel returns nothing; its results are global memory" in errors
    assert "parameter %s is ptr<f64, shared>; a kernel is handed global memory" in errors
    assert "parameter %t is tensor.tensor<f64, 2>; a kernel takes scalars and pointers" in errors
    assert "`gpu.kind` is one of host, device, kernel, not 'wavefront'" in errors


def test_allocations_and_shuffles_are_typed():
    module, host, kernel = _host_and_kernel()
    core.ret(host, module.functions["host"].entry.arguments[0])
    kernel.create("gpu.shared_alloc", (), (PtrType(F64, "private"),), {"count": 4})
    kernel.create("gpu.private_alloc", (), (PtrType(F64, "private"),), {"count": 0})
    kernel.create(
        "gpu.shared_alloc", (), (PtrType(tensor.tensor_type(F64, (2,)), "shared"),), {"count": 1}
    )
    value = core.const(kernel, 1.0, F64)
    kernel.create("gpu.subgroup_shuffle", (value, value), (F64,), {"mode": "xor"})
    kernel.create(
        "gpu.subgroup_shuffle", (value, core.const(kernel, 1, I64)), (I64,), {"mode": "up"}
    )
    kernel.create("gpu.thread_id", (), (I64,), {"dim": "x"})
    kernel.create("gpu.thread_id", (), (INDEX,), {"dim": "w"})
    core.ret(kernel)
    errors = _errors(module)
    assert "gpu.shared_alloc yields ptr<T, shared>, not ptr<f64, private>" in errors
    assert "`count` is a positive integer, not 0" in errors
    assert "gpu.shared_alloc holds scalars, not tensor.tensor<f64, 2>" in errors
    assert "a lane is an integer, not f64" in errors
    assert "subgroup_shuffle gives f64, not i64" in errors
    assert "gpu.thread_id is an index, not i64" in errors
    assert "dim" in errors and "w" in errors


def test_a_pointer_into_device_memory_needs_the_dialect():
    module = IRModule("plain")
    f = module.add_function("f", [("p", PtrType(F64, "shared"))], [F64])
    b = Builder(f.add_entry_block())
    core.ret(b, core.load(b, f.entry.arguments[0]))
    assert not verify(module), "the builtin registry knows the gpu dialect's spaces"
    module.require("gpu", 1)
    assert "gpu.thread_id" in {spec.name for spec in module_registry().ops_of("gpu")}


def module_registry():  # type: ignore[no-untyped-def]
    from ppy_compiler.ir import registry

    return registry()


@requires_llvm
def test_the_llvm_backend_leaves_device_code_alone():
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    module = _saxpy_module()
    module.functions.pop("run")  # a launch has no CPU lowering until the launch runtime
    twice = module.add_function("twice", [("x", F64)], [F64])
    b = Builder(twice.add_entry_block())
    (x,) = twice.entry.arguments
    core.ret(b, core.add(b, x, x))
    llvm = str(emit_module(module))
    assert "twice" in llvm and "saxpy" not in llvm and "tally" not in llvm and "fma" not in llvm
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    out = ctypes.c_double(0.0)
    call = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_double, ctypes.POINTER(ctypes.c_double))(
        engine.address("twice")
    )
    assert call(2.5, ctypes.byref(out)) == STATUS_OK and out.value == 5.0
