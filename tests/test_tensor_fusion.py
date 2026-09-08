"""Tensor canonicalization and fusion: fewer operations, one loop, no temporaries."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from ppy_compiler.backend.c import Language
from ppy_compiler.backend.c import emit_module as emit_c
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.from_ir import emit_module
from ppy_compiler.backend.llvm.fusion import FusedLoop, kernel_module
from ppy_compiler.backend.llvm.ir_pipeline import optimize
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.ir import F64, I64, BufferType, Builder, IRModule, decode, encode, verify
from ppy_compiler.ir import shape as shapes
from ppy_compiler.ir.dialects import core, tensor
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import (
    Canonicalize,
    DeadCodeElimination,
    FuseTensor,
    LowerTensor,
    TensorCanonicalize,
)
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

F64S = BufferType(F64)
N = shapes.Symbol("N")


def _ops(function) -> list[str]:  # type: ignore[no-untyped-def]
    return [op.name for op in function.body.blocks[0].operations]


def _run(module: IRModule, *passes) -> PassContext:  # type: ignore[no-untyped-def]
    problems = verify(module)
    assert not problems, problems
    ctx = PassContext()
    manager = PassManager(ctx)
    for pass_ in passes:
        manager.add(pass_)
    manager.run(module)
    problems = verify(module)
    assert not problems, problems
    return ctx


# -- canonicalization ------------------------------------------------------------------------


def test_views_that_change_nothing_go_away():
    module = IRModule("views")
    module.require("tensor", 1)
    t43 = tensor.tensor_type(F64, (4, 3))
    f = module.add_function("f", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, out = f.entry.arguments[0], f.entry.arguments[1]
    t = tensor.load(b, a, t43)
    same = tensor.broadcast(b, t, (4, 3))
    same = tensor.reshape(b, same, (4, 3))
    same = tensor.convert(b, same, F64)
    same = tensor.transpose(b, same, (0, 1))
    twice = tensor.transpose(b, tensor.transpose(b, same, (1, 0)), (1, 0))
    whole = tensor.slice_(b, twice, (0, 0), (4, 3), (1, 1))
    nothing = tensor.reduce(b, whole, (), "add")
    tensor.store(b, tensor.unary(b, "neg", tensor.unary(b, "neg", nothing)), out)
    core.ret(b, core.const(b, 0, I64))
    ctx = _run(module, TensorCanonicalize(), DeadCodeElimination())
    assert _ops(f) == ["tensor.load", "tensor.store", "core.const", "core.ret"]
    assert any("two undo each other" in r for r in ctx.remarks)
    assert any("double negation removed" in r for r in ctx.remarks)


def test_transposes_and_reshapes_compose_and_fills_fold():
    module = IRModule("fold")
    module.require("tensor", 1)
    t234 = tensor.tensor_type(F64, (2, 3, 4))
    f = module.add_function("f", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, out = f.entry.arguments[0], f.entry.arguments[1]
    t = tensor.load(b, a, t234)
    turned = tensor.transpose(b, tensor.transpose(b, t, (1, 2, 0)), (1, 0, 2))
    flat = tensor.reshape(b, tensor.reshape(b, turned, (6, 4)), (24,))
    two = tensor.fill(b, core.const(b, 2.0, F64), tensor.tensor_type(F64, (24,)))
    three = tensor.fill(b, core.const(b, 3.0, F64), tensor.tensor_type(F64, (24,)))
    scale = tensor.unary(b, "sqrt", tensor.elementwise(b, "mul", two, three))
    one = tensor.fill(b, core.const(b, 1.0, F64), tensor.tensor_type(F64, (24,)))
    zero = tensor.fill(b, core.const(b, 0.0, F64), tensor.tensor_type(F64, (24,)))
    scaled = tensor.elementwise(b, "mul", flat, one)
    tensor.store(
        b, tensor.elementwise(b, "add", tensor.elementwise(b, "mul", scaled, scale), zero), out
    )
    core.ret(b, core.const(b, 0, I64))
    ctx = _run(module, Canonicalize(), DeadCodeElimination())
    names = _ops(f)
    assert names.count("tensor.transpose") == 1, "two transposes composed into one"
    assert names.count("tensor.reshape") == 1, "two reshapes are one"
    assert names.count("tensor.fill") == 2, "sqrt(2 * 3) is one fill; 0.0 stays: -0.0 + 0.0"
    assert "tensor.unary" not in names, "the constant is computed once, as a scalar"
    assert names.count("tensor.mul") == 1, "x * fill 1.0 is x"
    assert names.count("tensor.add") == 1, "x + fill 0.0 is kept for floats"
    turned = next(op for op in f.body.blocks[0].operations if op.name == "tensor.transpose")
    assert tuple(turned.attributes["perm"]) == (2, 1, 0)
    assert any("computed once" in r for r in ctx.remarks)


def test_integer_neutral_elements_are_removed():
    module = IRModule("ints")
    module.require("tensor", 1)
    ints = tensor.tensor_type(I64, (5,))
    f = module.add_function("f", [("a", BufferType(I64)), ("out", BufferType(I64))], [I64])
    b = Builder(f.add_entry_block())
    a, out = f.entry.arguments[0], f.entry.arguments[1]
    t = tensor.load(b, a, ints)
    zero = tensor.fill(b, core.const(b, 0, I64), ints)
    one = tensor.fill(b, core.const(b, 1, I64), ints)
    value = tensor.elementwise(b, "sub", tensor.elementwise(b, "add", zero, t), zero)
    tensor.store(b, tensor.elementwise(b, "mul", one, value), out)
    core.ret(b, core.const(b, 0, I64))
    _run(module, TensorCanonicalize(), DeadCodeElimination())
    assert _ops(f) == ["tensor.load", "tensor.store", "core.const", "core.ret"]


# -- fusion --------------------------------------------------------------------------------


def _chain() -> IRModule:
    module = IRModule("chain")
    module.require("tensor", 1)
    vector = tensor.tensor_type(F64, (N,))
    f = module.add_function("blend", [("a", F64S), ("b", F64S), ("s", F64), ("out", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, c, s, out = (f.entry.arguments[i] for i in range(4))
    ta = tensor.load(b, a, vector, "a")
    tb = tensor.load(b, c, vector, "b")
    root = tensor.unary(b, "sqrt", ta)
    scaled = tensor.elementwise(b, "mul", root, tb)
    shifted = tensor.elementwise(b, "add", scaled, tensor.fill(b, s, vector))
    tensor.store(b, tensor.elementwise(b, "max", shifted, tensor.unary(b, "neg", tb)), out)
    core.ret(b, core.const(b, 0, I64))
    g = module.add_function("energy", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(g.add_entry_block())
    a, out = g.entry.arguments[0], g.entry.arguments[1]
    ta = tensor.load(b, a, tensor.tensor_type(F64, (N, 2)), "a")
    squared = tensor.elementwise(b, "mul", ta, ta)
    tensor.store(b, tensor.reduce(b, squared, (1,), "add"), out)
    core.ret(b, core.const(b, 0, I64))
    h = module.add_function("shared", [("a", F64S), ("out", F64S), ("out2", F64S)], [I64])
    b = Builder(h.add_entry_block())
    a, out, out2 = (h.entry.arguments[i] for i in range(3))
    ta = tensor.load(b, a, vector, "a")
    root = tensor.unary(b, "sqrt", ta)
    tensor.store(b, tensor.elementwise(b, "add", root, ta), out)
    tensor.store(b, tensor.elementwise(b, "sub", root, ta), out2)
    core.ret(b, core.const(b, 0, I64))
    return module


def test_chains_of_elementwise_operations_fuse_into_one_region():
    module = _chain()
    ctx = _run(module, FuseTensor())
    blend = module.functions["blend"]
    assert _ops(blend) == [
        "tensor.load",
        "tensor.load",
        "tensor.fused",
        "tensor.store",
        "core.const",
        "core.ret",
    ]
    fused = next(op for op in blend.body.blocks[0].operations if op.name == "tensor.fused")
    assert [str(v.type) for v in fused.operands] == [
        "tensor.tensor<f64, N>",
        "tensor.tensor<f64, N>",
        "f64",
    ]
    body = fused.regions[0].blocks[0]
    assert [str(a.type) for a in body.arguments] == ["f64", "f64", "f64"]
    assert [op.name for op in body.operations] == [
        "math.sqrt",
        "core.mul",
        "core.add",
        "core.neg",
        "core.cmp",
        "core.select",
        "core.cmp",
        "core.select",
        "tensor.yield",
    ]
    energy = module.functions["energy"]
    assert _ops(energy) == ["tensor.load", "tensor.fused", "tensor.store", "core.const", "core.ret"]
    reduced = next(op for op in energy.body.blocks[0].operations if op.name == "tensor.fused")
    assert reduced.attributes["reduce"] == "add" and tuple(reduced.attributes["axes"]) == (1,)
    assert str(reduced.results[0].type) == "tensor.tensor<f64, N>"
    shared = module.functions["shared"]
    assert _ops(shared).count("tensor.unary") == 1, "a value two groups read stays a value"
    assert _ops(shared).count("tensor.fused") == 0, "a group of one is not fused"
    assert sum("tensor ops fused" in r for r in ctx.remarks) == 2
    assert any("sqrt, mul, fill, add, neg, max (6 operations)" in r for r in ctx.remarks)
    text = encode(module)
    assert "^body(%" in text and "tensor.yield" in text
    again = decode(text)
    assert not verify(again) and encode(again) == text


def test_a_fused_region_is_verified():
    module = IRModule("bad")
    module.require("tensor", 1)
    module.require("math", 1)
    vector = tensor.tensor_type(F64, (4,))
    f = module.add_function("f", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, out = f.entry.arguments[0], f.entry.arguments[1]
    ta = tensor.load(b, a, vector)
    wrong_arity = b.create("tensor.fused", (ta,), (vector,))
    body = wrong_arity.add_region().add_block("body", [(None, F64), (None, F64)])
    tensor.yield_(Builder(body), body.arguments[0])
    wrong_yield = b.create("tensor.fused", (ta,), (vector,))
    body = wrong_yield.add_region().add_block("body", [(None, F64)])
    bb = Builder(body)
    tensor.yield_(bb, core.cmp(bb, "lt", body.arguments[0], body.arguments[0]))
    impure = b.create("tensor.fused", (ta,), (vector,))
    body = impure.add_region().add_block("body", [(None, F64)])
    bb = Builder(body)
    slot = core.alloca(bb, F64)
    core.store(bb, body.arguments[0], slot)
    tensor.yield_(bb, core.load(bb, slot))
    wrong_shape = b.create("tensor.fused", (ta,), (tensor.tensor_type(F64, (5,)),))
    body = wrong_shape.add_region().add_block("body", [(None, F64)])
    tensor.yield_(Builder(body), body.arguments[0])
    tensor.store(b, ta, out)
    core.ret(b, core.const(b, 0, I64))
    messages = [e.message for e in verify(module)]
    assert any("takes 2 elements for 1 operands" in m for m in messages)
    assert any("yields bool, the result holds f64" in m for m in messages)
    assert any("core.alloca is not scalar arithmetic" in m for m in messages)
    assert any("tensor.fused computes shape" in m for m in messages)


def _lowered() -> IRModule:
    module = _chain()
    _run(module, FuseTensor(), LowerTensor())
    return module


def test_a_fused_region_lowers_to_one_loop_that_writes_the_destination():
    module = _lowered()
    text = encode(module)
    blend = text[text.index("func @blend") : text.index("func @energy")]
    assert "tensor." not in blend.replace("tensor.tensor", "")
    assert "malloc" not in blend and "core.alloca" not in blend, "no temporary: out is written"
    assert blend.count("\n^t.head") == 1, "one loop"
    energy = text[text.index("func @energy") : text.index("func @shared")]
    assert "malloc" in energy, "the reduction's accumulator is the result tensor"
    assert energy.count("\n^t.head") == 4, "init (1), fold over N by 2 (2), copy out (1)"


def _call(address, name: str, *arguments):  # type: ignore[no-untyped-def]
    atoms: list = []
    kinds: list = []
    double_pointer = ctypes.POINTER(ctypes.c_double)
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            atoms.extend([argument.ctypes.data_as(double_pointer), argument.size])
            kinds.extend([double_pointer, ctypes.c_int64])
        else:
            atoms.append(float(argument))
            kinds.append(ctypes.c_double)
    out = ctypes.c_int64(0)
    function = ctypes.CFUNCTYPE(ctypes.c_int32, *kinds, ctypes.POINTER(ctypes.c_int64))(address)
    return function(*atoms, ctypes.byref(out))


def _check(address) -> None:  # type: ignore[no-untyped-def]
    a = np.linspace(0.0, 9.0, 10)
    c = np.linspace(-1.0, 1.0, 10)
    out = np.zeros(10)
    assert _call(address("blend"), "blend", a, c, 0.25, out) == STATUS_OK
    np.testing.assert_allclose(out, np.maximum(np.sqrt(a) * c + 0.25, -c))
    pairs = np.arange(8, dtype=np.float64)
    sums = np.zeros(4)
    assert _call(address("energy"), "energy", pairs, sums) == STATUS_OK
    np.testing.assert_array_equal(sums, (pairs.reshape(4, 2) ** 2).sum(axis=1))
    plus, minus = np.zeros(10), np.zeros(10)
    assert _call(address("shared"), "shared", a, plus, minus) == STATUS_OK
    np.testing.assert_allclose(plus, np.sqrt(a) + a)
    np.testing.assert_allclose(minus, np.sqrt(a) - a)


@requires_llvm
def test_fused_regions_agree_with_numpy_on_llvm():
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(_lowered()))
    engine.finalize()
    _check(engine.address)


@requires_cc
def test_fused_regions_agree_with_numpy_in_c(tmp_path):  # type: ignore[no-untyped-def]
    source = tmp_path / "fused.c"
    source.write_text(emit_c(_lowered(), Language.C), encoding="utf-8")
    library = tmp_path / "fused.so"
    compiler = c_compiler()
    assert compiler is not None
    result = subprocess.run(
        [compiler, "-std=c11", "-Wall", "-O2", "-shared", "-fPIC", "-o", str(library), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "warning:" not in result.stderr, result.stderr
    handle = ctypes.CDLL(str(library))
    _check(lambda name: ctypes.cast(getattr(handle, name), ctypes.c_void_p).value)


@requires_llvm
def test_a_fused_kernel_is_one_loop_with_no_temporaries():
    loop = FusedLoop("k", ("a", "b"), ("s",), expression="(add (mul (sin a0) s0) (cos a1))")
    total = FusedLoop("t", ("a",), (), reduction="add", expression="(mul a0 a0)")
    module = kernel_module([loop, total], "kernels")
    ctx = optimize(module, 2)
    assert any("tensor ops fused" in r for r in ctx.remarks)
    text = encode(module)
    kernel = text[text.index("func @k(") : text.index("func @t(")]
    assert "malloc" not in kernel and "core.alloca" not in kernel
    assert kernel.count("\n^t.head") == 1
    reduction = text[text.index("func @t(") :]
    assert "malloc" not in reduction, "a rank-0 result lives on the stack"
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    values = np.linspace(-1.0, 1.0, 17)
    out = np.zeros(17)
    double = ctypes.POINTER(ctypes.c_double)
    placeholder = ctypes.c_int64(0)
    run = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        double,
        double,
        double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    )(engine.address("k"))
    status = run(
        out.ctypes.data_as(double),
        values.ctypes.data_as(double),
        values.ctypes.data_as(double),
        0.5,
        17,
        ctypes.byref(placeholder),
    )
    assert status == STATUS_OK
    np.testing.assert_allclose(out, np.sin(values) * 0.5 + np.cos(values))
