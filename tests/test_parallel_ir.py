"""The parallel dialect: one meaning, lowered every way, verified by running."""

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
    decode,
    encode,
    verify,
)
from ppy_compiler.ir.dialects import core, parallel
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import LowerParallel
from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_openmp = (
    c_compiler() is not None
    and subprocess.run(
        [c_compiler() or "cc", "-fopenmp", "-x", "c", "-", "-o", "/dev/null"],
        input="int main(void) { return 0; }",
        capture_output=True,
        text=True,
        check=False,
    ).returncode
    == 0
)
requires_openmp = pytest.mark.skipif(not _openmp, reason="the C compiler has no OpenMP")


def _module() -> IRModule:
    """`fill(out, n)`: out[i] = i * i in parallel; `total(xs, n)`: the sum of
    xs[i] * 2, a reduction; `squares(xs, out, n)`: out[i] = xs[i] * xs[i], a
    map; `least(xs, n, init)`: the minimum from init."""
    module = IRModule("par")
    module.require("parallel", 1)
    body = module.add_function(
        "fill_body", [("out", PtrType(I64)), ("begin", I64), ("end", I64)], []
    )
    entry = body.add_entry_block()
    out = entry.arguments[0]
    begin = entry.arguments[1]
    end = entry.arguments[2]
    head = body.body.add_block("head", [("i", I64)])
    loop = body.body.add_block("loop")
    done = body.body.add_block("done")
    b = Builder(entry)
    core.br(b, Successor(head, (begin,)))
    b = Builder(head)
    i = head.arguments[0]
    core.cond_br(b, core.cmp(b, "lt", i, end), Successor(loop), Successor(done))
    b = Builder(loop)
    core.store(b, core.mul(b, i, i, overflow="wrap"), core.ptr_offset(b, out, i))
    core.br(b, Successor(head, (core.add(b, i, core.const(b, 1, I64), overflow="wrap"),)))
    core.ret(Builder(done))

    fill = module.add_function("fill", [("out", PtrType(I64)), ("n", I64)], [I64])
    b = Builder(fill.add_entry_block())
    out = fill.entry.arguments[0]
    n = fill.entry.arguments[1]
    parallel.loop(b, "fill_body", (out,), core.const(b, 0, I64), n)
    core.ret(
        b,
        core.load(
            b, core.ptr_offset(b, out, core.sub(b, n, core.const(b, 1, I64), overflow="wrap"))
        ),
    )

    sum_body = module.add_function(
        "sum_body",
        [("xs", PtrType(I64, mutable=False)), ("begin", I64), ("end", I64), ("acc", I64)],
        [I64],
    )
    entry = sum_body.add_entry_block()
    xs = entry.arguments[0]
    begin = entry.arguments[1]
    end = entry.arguments[2]
    acc = entry.arguments[3]
    head = sum_body.body.add_block("head", [("i", I64), ("acc", I64)])
    loop = sum_body.body.add_block("loop")
    done = sum_body.body.add_block("done")
    b = Builder(entry)
    core.br(b, Successor(head, (begin, acc)))
    b = Builder(head)
    i = head.arguments[0]
    running = head.arguments[1]
    core.cond_br(b, core.cmp(b, "lt", i, end), Successor(loop), Successor(done))
    b = Builder(loop)
    doubled = core.mul(
        b, core.load(b, core.ptr_offset(b, xs, i)), core.const(b, 2, I64), overflow="wrap"
    )
    core.br(
        b,
        Successor(
            head,
            (
                core.add(b, i, core.const(b, 1, I64), overflow="wrap"),
                core.add(b, running, doubled, overflow="wrap"),
            ),
        ),
    )
    core.ret(Builder(done), running)

    total = module.add_function("total", [("xs", PtrType(I64, mutable=False)), ("n", I64)], [I64])
    b = Builder(total.add_entry_block())
    xs = total.entry.arguments[0]
    n = total.entry.arguments[1]
    core.ret(
        b,
        parallel.reduce(
            b, "sum_body", (xs,), core.const(b, 0, I64), n, core.const(b, 100, I64), "add"
        ),
    )

    square = module.add_function(
        "square_one", [("xs", PtrType(F64, mutable=False)), ("i", I64)], [F64]
    )
    b = Builder(square.add_entry_block())
    xs = square.entry.arguments[0]
    i = square.entry.arguments[1]
    x = core.load(b, core.ptr_offset(b, xs, i))
    core.ret(b, core.mul(b, x, x))

    squares = module.add_function(
        "squares", [("xs", PtrType(F64, mutable=False)), ("out", PtrType(F64)), ("n", I64)], [F64]
    )
    b = Builder(squares.add_entry_block())
    xs = squares.entry.arguments[0]
    out = squares.entry.arguments[1]
    n = squares.entry.arguments[2]
    parallel.map_(b, "square_one", (xs,), core.const(b, 0, I64), n, out)
    core.ret(b, core.load(b, out))

    min_body = module.add_function(
        "min_body",
        [("xs", PtrType(F64, mutable=False)), ("begin", I64), ("end", I64), ("acc", F64)],
        [F64],
    )
    entry = min_body.add_entry_block()
    xs = entry.arguments[0]
    begin = entry.arguments[1]
    end = entry.arguments[2]
    acc = entry.arguments[3]
    head = min_body.body.add_block("head", [("i", I64), ("acc", F64)])
    loop = min_body.body.add_block("loop")
    done = min_body.body.add_block("done")
    b = Builder(entry)
    core.br(b, Successor(head, (begin, acc)))
    b = Builder(head)
    i = head.arguments[0]
    running = head.arguments[1]
    core.cond_br(b, core.cmp(b, "lt", i, end), Successor(loop), Successor(done))
    b = Builder(loop)
    x = core.load(b, core.ptr_offset(b, xs, i))
    smaller = core.select(b, core.cmp(b, "lt", x, running), x, running)
    core.br(b, Successor(head, (core.add(b, i, core.const(b, 1, I64), overflow="wrap"), smaller)))
    core.ret(Builder(done), running)

    least = module.add_function(
        "least", [("xs", PtrType(F64, mutable=False)), ("n", I64), ("init", F64)], [F64]
    )
    b = Builder(least.add_entry_block())
    xs = least.entry.arguments[0]
    n = least.entry.arguments[1]
    init = least.entry.arguments[2]
    core.ret(
        b,
        parallel.reduce(
            b, "min_body", (xs,), core.const(b, 0, I64), n, init, "min", reassociate=False
        ),
    )
    assert not verify(module), verify(module)
    return module


def _ops(text: str, dialect: str) -> list[str]:
    """The operation lines of `dialect` in printed IR (not its attributes)."""
    return [
        line
        for line in text.splitlines()
        if line.strip().startswith(f"{dialect}.") or f" = {dialect}." in line
    ]


def _lowered(backend: str, threads: int = 4, minimum: int = 8) -> IRModule:
    module = _module()
    ctx = PassContext()
    manager = PassManager(ctx)
    manager.add(LowerParallel(backend, threads=threads, minimum=minimum))
    manager.run(module)
    problems = verify(module)
    assert not problems, problems
    return module


def test_the_dialect_prints_reads_back_and_is_held_to_its_rules():
    module = _module()
    text = encode(module)
    again = decode(text)
    assert not verify(again) and encode(again) == text
    assert (
        'parallel.reduce %xs, %0, %n, %1 {callee = @sum_body, op = "add", reassociate = true} : i64'
        in text
    )
    bad = IRModule("bad")
    bad.require("parallel", 1)
    body = bad.add_function("body", [("p", PtrType(I64)), ("i", I64)], [I64])
    core.ret(Builder(body.add_entry_block()), core.const(Builder(body.entry), 1, I64))
    f = bad.add_function("f", [("p", PtrType(I64)), ("n", I64)], [I64])
    b = Builder(f.add_entry_block())
    p = f.entry.arguments[0]
    n = f.entry.arguments[1]
    parallel.loop(b, "body", (p,), core.const(b, 0, I64), n)  # body ends in (i), not (begin, end)
    b.create(
        "parallel.reduce",
        (p, core.const(b, 0, I64), n, core.const(b, 0, I64)),
        (I64,),
        {"callee": SymbolRef("body"), "op": "avg"},
    )
    core.ret(b, core.const(b, 0, I64))
    messages = [e.message for e in verify(bad)]
    assert any("the body takes 2 more" in m for m in messages)
    assert any("`op` is one of add, mul, min, max" in m for m in messages)


def test_lowering_keeps_the_meaning_whatever_the_backend():
    serial = encode(_lowered("serial"))
    threads = encode(_lowered("threads"))
    assert not _ops(serial, "parallel") and not _ops(threads, "parallel")
    spawns = [line for line in threads.splitlines() if "concurrency.spawn" in line]
    assert "concurrency.spawn" not in serial and len(spawns) == 9, "three loops, three chunks each"
    assert "func @square_one__map" in serial and "func @sum_body__task" in threads
    assert sum("@sum_body__task" in line for line in spawns) == 3, "three of four chunks spawn"
    assert sum("@fill_body" in line for line in spawns) == 3
    assert not any("@min_body" in line for line in spawns), "an order-keeping reduction stays whole"
    untouched = encode(_lowered("openmp"))
    assert _ops(untouched, "parallel"), "OpenMP is the C backend's to spell"


def _run(address, name: str):  # type: ignore[no-untyped-def]
    ints = (ctypes.c_int64 * 1000)(*range(1000))
    out_i = ctypes.c_int64(0)
    fill = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    )(address("fill"))
    filled = (ctypes.c_int64 * 1000)()
    assert fill(filled, 1000, ctypes.byref(out_i)) == STATUS_OK
    assert list(filled) == [i * i for i in range(1000)] and out_i.value == 999 * 999
    total = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    )(address("total"))
    assert total(ints, 1000, ctypes.byref(out_i)) == STATUS_OK
    assert out_i.value == 100 + 2 * sum(range(1000))
    assert total(ints, 3, ctypes.byref(out_i)) == STATUS_OK and out_i.value == 100 + 6, (
        "a small range runs serially"
    )
    floats = (ctypes.c_double * 1000)(*(0.5 * i for i in range(1000)))
    squared = (ctypes.c_double * 1000)()
    out_f = ctypes.c_double(0.0)
    squares = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
    )(address("squares"))
    assert squares(floats, squared, 1000, ctypes.byref(out_f)) == STATUS_OK
    assert list(squared) == [(0.5 * i) ** 2 for i in range(1000)]
    least = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    )(address("least"))
    values = (ctypes.c_double * 1000)(*(1000.0 - i for i in range(1000)))
    assert least(values, 1000, 5000.0, ctypes.byref(out_f)) == STATUS_OK and out_f.value == 1.0
    assert least(values, 1000, -3.0, ctypes.byref(out_f)) == STATUS_OK and out_f.value == -3.0
    return name


@requires_llvm
@pytest.mark.parametrize("backend", ["serial", "threads", "simd"])
def test_the_llvm_backend_runs_every_lowering(backend: str):
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(_lowered(backend)))
    engine.finalize()
    _run(engine.address, backend)


def _library(module: IRModule, directory: Path, *flags: str) -> ctypes.CDLL:
    from ppy_compiler.backend.c import Language, emit_module

    source = directory / "par.c"
    source.write_text(emit_module(module, Language.C), encoding="utf-8")
    path = directory / "libpar.so"
    done = subprocess.run(
        [
            c_compiler(),
            "-std=c11",
            "-Wall",
            "-O2",
            "-shared",
            "-fPIC",
            *flags,
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


def _symbol(library: ctypes.CDLL):  # type: ignore[no-untyped-def]
    return lambda name: ctypes.cast(getattr(library, name), ctypes.c_void_p).value or 0


@requires_cc
@pytest.mark.parametrize("backend", ["serial", "threads"])
def test_the_c_backend_runs_the_lowered_chunks(tmp_path: Path, backend: str):
    _run(_symbol(_library(_lowered(backend), tmp_path)), backend)


@requires_openmp
def test_the_c_backend_spells_openmp_when_selected(tmp_path: Path):
    from ppy_compiler.backend.c import Language, emit_module

    module = _lowered("openmp")
    text = emit_module(module, Language.C)
    assert "#pragma omp parallel" in text and "#include <omp.h>" in text
    assert "/* compile with: -fopenmp */" in text
    library = _library(module, tmp_path, "-fopenmp")
    _run(_symbol(library), "openmp")


def test_a_chunk_that_fails_a_guard_fails_the_whole_loop():
    pytest.importorskip("llvmlite")
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    module = IRModule("guarded")
    module.require("parallel", 1)
    body = module.add_function("body", [("limit", I64), ("begin", I64), ("end", I64)], [])
    b = Builder(body.add_entry_block())
    limit = body.entry.arguments[0]
    _begin = body.entry.arguments[1]
    end = body.entry.arguments[2]
    core.guard(b, core.cmp(b, "le", end, limit), "range", "past the limit")
    core.ret(b)
    f = module.add_function("f", [("limit", I64), ("n", I64)], [I64])
    b = Builder(f.add_entry_block())
    limit = f.entry.arguments[0]
    n = f.entry.arguments[1]
    parallel.loop(b, "body", (limit,), core.const(b, 0, I64), n)
    core.ret(b, n)
    assert not verify(module)
    manager = PassManager(PassContext())
    manager.add(LowerParallel("threads", threads=3, minimum=4))
    manager.run(module)
    assert not verify(module)
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    out = ctypes.c_int64(0)
    run = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)
    )(engine.address("f"))
    assert run(1000, 300, ctypes.byref(out)) == STATUS_OK and out.value == 300
    assert run(100, 300, ctypes.byref(out)) == STATUS_FALLBACK
