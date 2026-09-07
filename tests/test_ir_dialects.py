"""The simd, cpu, atomic, and concurrency dialects: verified, printed,
read back, and run the same way by the LLVM and C backends."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.ir import (
    F64,
    I64,
    Builder,
    IRModule,
    PtrType,
    Successor,
    SymbolRef,
    VectorType,
    decode,
    encode,
    verify,
)
from ppy_compiler.ir.dialects import atomic, concurrency, core, cpu, simd
from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

V4 = VectorType(F64, 4)
V4I = VectorType(I64, 4)


def _module() -> IRModule:
    module = IRModule("lanes")
    for name in ("simd", "cpu", "atomic", "concurrency"):
        module.require(name, 1)
    return module


def _lanes_module() -> IRModule:
    """`dot(a, b)`: the lanes at a and b multiplied and summed, in order;
    `spread(p, x)`: p[0..4] set to x, x+1, x+2, x+3 and their max returned;
    `mix(p)`: the even lanes of p[0..4] shuffled ahead of the odd ones,
    stored back, and lane 3 returned."""
    module = _module()
    dot = module.add_function(
        "dot", [("a", PtrType(F64, mutable=False)), ("b", PtrType(F64, mutable=False))], [F64]
    )
    b = Builder(dot.add_entry_block())
    a, c = dot.entry.arguments
    product = core.mul(b, simd.load(b, a, 4), simd.load(b, c, 4), overflow="wrap")
    core.ret(b, simd.reduce(b, "add", product))

    spread = module.add_function("spread", [("p", PtrType(I64)), ("x", I64)], [I64])
    b = Builder(spread.add_entry_block())
    p, x = spread.entry.arguments
    ramp = simd.splat(b, x, 4)
    for lane in range(1, 4):
        ramp = simd.insert(
            b,
            ramp,
            core.add(b, x, core.const(b, lane, I64), overflow="wrap"),
            core.const(b, lane, I64),
        )
    simd.store(b, ramp, p)
    cpu.prefetch(b, p, rw="write", locality=1)
    core.ret(b, simd.reduce(b, "max", ramp))

    mix = module.add_function("mix", [("p", PtrType(F64))], [F64])
    b = Builder(mix.add_entry_block())
    (p,) = mix.entry.arguments
    loaded = simd.load(b, p, 4)
    shuffled = simd.shuffle(b, loaded, loaded, (0, 2, 1, 3))
    doubled = core.add(b, shuffled, shuffled, overflow="wrap")
    simd.store(b, doubled, p)
    core.ret(b, simd.extract(b, doubled, core.const(b, 3, I64)))

    minimum = module.add_function("least", [("p", PtrType(F64, mutable=False))], [F64])
    b = Builder(minimum.add_entry_block())
    (p,) = minimum.entry.arguments
    core.ret(b, simd.reduce(b, "min", simd.load(b, p, 4)))
    assert not verify(module)
    return module


def _threads_module() -> IRModule:
    """`count(slot, n)`: n threads each add 1 to *slot one thousand times
    under a mutex, then all pass a barrier; the joiner returns *slot."""
    module = _module()
    worker = module.add_function(
        "worker",
        [("slot", PtrType(I64)), ("mutex", PtrType(I64)), ("gate", PtrType(I64)), ("parties", I64)],
        [],
    )
    b = Builder(worker.add_entry_block())
    slot, mutex, gate, parties = worker.entry.arguments
    head = worker.body.add_block("loop", [("i", I64)])
    body = worker.body.add_block("body")
    done = worker.body.add_block("done")
    core.br(b, Successor(head, (core.const(b, 0, I64),)))
    b = Builder(head)
    i = head.arguments[0]
    core.cond_br(
        b, core.cmp(b, "lt", i, core.const(b, 1000, I64)), Successor(body), Successor(done)
    )
    b = Builder(body)
    concurrency.mutex_lock(b, mutex)
    old = core.load(b, slot)
    core.store(b, core.add(b, old, core.const(b, 1, I64), overflow="wrap"), slot)
    concurrency.mutex_unlock(b, mutex)
    core.br(b, Successor(head, (core.add(b, i, core.const(b, 1, I64), overflow="wrap"),)))
    b = Builder(done)
    atomic.fetch(b, "add", slot, core.const(b, 0, I64), "acq_rel")
    concurrency.barrier(b, gate, parties)
    core.ret(b)

    count = module.add_function(
        "count",
        [("slot", PtrType(I64)), ("mutex", PtrType(I64)), ("gate", PtrType(I64)), ("n", I64)],
        [I64],
    )
    b = Builder(count.add_entry_block())
    slot, mutex, gate, n = count.entry.arguments
    handles = [concurrency.spawn(b, "worker", (slot, mutex, gate, n)) for _ in range(3)]
    for handle in handles:
        status = concurrency.join(b, handle)
        core.guard(
            b, core.cmp(b, "eq", status, core.const(b, 0, I64)), "contract", "a thread failed"
        )
    atomic.fence(b, "seq_cst")
    core.ret(b, atomic.load(b, slot, "acquire"))

    ident = module.add_function("ident", [], [I64])
    b = Builder(ident.add_entry_block())
    core.ret(b, concurrency.thread_id(b))
    assert not verify(module)
    return module


def test_the_dialects_print_and_read_back():
    for module in (_lanes_module(), _threads_module()):
        text = encode(module)
        again = decode(text)
        assert not verify(again)
        assert encode(again) == text
    text = encode(_lanes_module())
    assert "simd.shuffle" in text and "mask = [0, 2, 1, 3]" in text
    assert "cpu.prefetch" in text and 'rw = "write"' in text
    threads = encode(_threads_module())
    assert "concurrency.spawn @worker" in threads or "concurrency.spawn" in threads
    assert "atomic.fetch_add" in threads and 'order = "acq_rel"' in threads


def test_the_verifier_holds_each_dialect_to_its_rules():
    module = _module()
    f = module.add_function("f", [("p", PtrType(I64)), ("q", PtrType(F64, mutable=False))], [I64])
    b = Builder(f.add_entry_block())
    p, q = f.entry.arguments
    v = simd.load(b, p, 4)
    b.create("simd.reduce_add", (v,), (F64,))  # wrong result type
    b.create("simd.shuffle", (v, v), (V4I,), {"mask": (0, 9, 1, 2)})  # lane out of range
    b.create("atomic.load", (p,), (I64,), {"order": "release"})  # a load is not release
    b.create(
        "atomic.store", (core.const(b, 1.0, F64), q), (), {"order": "seq_cst"}
    )  # const pointer
    b.create("atomic.fetch_add", (q, core.const(b, 1.0, F64)), (F64,), {"order": "seq_cst"})
    b.create("concurrency.mutex_lock", (q,), ())
    b.create("concurrency.spawn", (p,), (I64,), {"callee": SymbolRef("nowhere")})
    core.ret(b, core.const(b, 0, I64))
    messages = [e.message for e in verify(module)]
    assert any("reducing vector<i64, 4> gives i64" in m for m in messages)
    assert any("shuffle lane is in [0, 8)" in m for m in messages)
    assert any("a load is not `release`" in m for m in messages)
    assert any("writes through a const pointer" in m for m in messages)
    assert any("pointer to an integer" in m for m in messages)
    assert any("a mutex is a mutable pointer to i64" in m for m in messages)
    assert any("not a function of this module" in m for m in messages)


def _jit(module: IRModule):  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    return engine


def _library(module: IRModule, directory: Path):  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.c import Language, emit_module

    source = directory / f"{module.name}.c"
    source.write_text(emit_module(module, Language.C), encoding="utf-8")
    path = directory / f"lib{module.name}.so"
    done = subprocess.run(
        [
            c_compiler(),
            "-std=c11",
            "-Wall",
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(path),
            str(source),
            "-lpthread",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr + source.read_text()
    assert "warning:" not in done.stderr, done.stderr
    return ctypes.CDLL(str(path))


class _Address:
    def __init__(self, engine_or_library) -> None:  # type: ignore[no-untyped-def]
        self.target = engine_or_library

    def __call__(self, symbol: str) -> int:
        if hasattr(self.target, "address"):
            return self.target.address(symbol)
        return ctypes.cast(getattr(self.target, symbol), ctypes.c_void_p).value or 0


def _lanes_answers(address: _Address) -> tuple:
    double4 = ctypes.c_double * 4
    out = ctypes.c_double(0.0)
    dot = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )(address("dot"))
    assert (
        dot(double4(1.0, 2.0, 3.0, 4.0), double4(0.5, 0.25, 2.0, 10.0), ctypes.byref(out))
        == STATUS_OK
    )
    first = out.value
    ints = (ctypes.c_int64 * 4)()
    out_i = ctypes.c_int64(0)
    spread = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    )(address("spread"))
    assert spread(ints, 2**63 - 2, ctypes.byref(out_i)) == STATUS_OK
    second = (list(ints), out_i.value)
    values = double4(1.0, 2.0, 3.0, 4.0)
    mix = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    )(address("mix"))
    assert mix(values, ctypes.byref(out)) == STATUS_OK
    third = (list(values), out.value)
    least = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    )(address("least"))
    nan = float("nan")
    assert least(double4(3.0, nan, -1.0, 2.0), ctypes.byref(out)) == STATUS_OK
    fourth = out.value
    assert least(double4(nan, 3.0, -1.0, 2.0), ctypes.byref(out)) == STATUS_OK
    fifth = out.value
    return first, second, third, fourth, fifth


@requires_llvm
def test_vectors_compute_lane_by_lane_in_order_on_the_llvm_backend():
    answers = _lanes_answers(_Address(_jit(_lanes_module())))
    assert answers[0] == 1.0 * 0.5 + 2.0 * 0.25 + 3.0 * 2.0 + 4.0 * 10.0
    assert answers[1] == ([2**63 - 2, 2**63 - 1, -(2**63), -(2**63) + 1], 2**63 - 1), "lanes wrap"
    assert answers[2] == ([2.0, 6.0, 4.0, 8.0], 8.0)
    assert answers[3] == -1.0, "a NaN lane after the first is passed over, as Python's min does"
    assert answers[4] != answers[4], "a NaN first lane stays, as Python's min does"


@requires_llvm
@requires_cc
def test_the_c_backend_agrees_with_llvm_on_vectors(tmp_path: Path):
    module = _lanes_module()
    from_llvm = _lanes_answers(_Address(_jit(module)))
    from_c = _lanes_answers(_Address(_library(module, tmp_path)))
    assert from_c[:4] == from_llvm[:4]
    assert from_c[4] != from_c[4]


def _count(address: _Address) -> tuple[int, int, int]:
    slot = ctypes.c_int64(0)
    mutex = ctypes.c_int64(0)
    gate = (ctypes.c_int64 * 2)(0, 0)
    out = ctypes.c_int64(0)
    count = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    )(address("count"))
    status = count(ctypes.byref(slot), ctypes.byref(mutex), gate, 3, ctypes.byref(out))
    ident = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(ctypes.c_int64))(address("ident"))
    me = ctypes.c_int64(0)
    assert ident(ctypes.byref(me)) == STATUS_OK
    return status, out.value, me.value


@requires_llvm
def test_threads_count_under_a_mutex_and_meet_at_a_barrier():
    status, total, me = _count(_Address(_jit(_threads_module())))
    assert (status, total) == (STATUS_OK, 3000)
    assert me != 0


@requires_llvm
@requires_cc
def test_the_c_backend_runs_the_same_threads(tmp_path: Path):
    status, total, me = _count(_Address(_library(_threads_module(), tmp_path)))
    assert (status, total) == (STATUS_OK, 3000)
    assert me != 0


@requires_llvm
def test_a_thread_that_fails_a_guard_fails_its_joiner():
    module = _module()
    worker = module.add_function("worker", [("n", I64)], [])
    b = Builder(worker.add_entry_block())
    (n,) = worker.entry.arguments
    core.guard(b, core.cmp(b, "gt", n, core.const(b, 0, I64)), "range", "positive")
    core.ret(b)
    run = module.add_function("run", [("n", I64)], [I64])
    b = Builder(run.add_entry_block())
    (n,) = run.entry.arguments
    status = concurrency.join(b, concurrency.spawn(b, "worker", (n,)))
    core.guard(b, core.cmp(b, "eq", status, core.const(b, 0, I64)), "contract", "thread ok")
    core.ret(b, status)
    assert not verify(module)
    engine = _jit(module)
    out = ctypes.c_int64(0)
    run_c = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))(
        engine.address("run")
    )
    assert run_c(5, ctypes.byref(out)) == STATUS_OK and out.value == 0
    assert run_c(-5, ctypes.byref(out)) == STATUS_FALLBACK
