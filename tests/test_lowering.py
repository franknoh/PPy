"""AST -> canonical IR -> LLVM, held against the AST -> LLVM road it replaces.

The two roads must agree on every answer, including which calls fall back
to CPython. The differential test here JIT-compiles the same functions
both ways and calls them on the same inputs; `PPY_LOWERING=ir` runs the
whole suite the same way.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.backend.llvm.lowering import lower_module
from ppy_compiler.ir import encode, verify
from ppy_compiler.lowering import lower_module_to_ir
from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

KERNELS = """
    import math

    import ppy


    def collatz(n: int) -> int:
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            steps += 1
        return steps


    def floor_mod(a: int, b: int) -> int:
        return a // b + a % b


    def cube(n: int) -> int:
        return n * n * n


    def shifted(n: int, k: int) -> int:
        return (n << k) + (n >> 1)


    def mixed(x: float, n: int) -> float:
        return x * n + 0.5 if n > 0 else -x


    def parity(n: int) -> bool:
        return n % 2 == 0 and n > 0 or n == -7


    def hypot(x: float, y: float) -> float:
        return math.sqrt(x * x + y * y)


    def clamp(x: int, lo: int, hi: int) -> int:
        result = x
        if result < lo:
            result = lo
        if result > hi:
            result = hi
        return result


    def triangle(n: int) -> int:
        total = 0
        for i in range(n):
            total += i
        return total


    def twice(n: int) -> int:
        return cube(n) + cube(n)
    """

CASES: dict[str, list[tuple]] = {
    "collatz": [(1,), (27,), (97,), (871,)],
    "floor_mod": [(7, 2), (-7, 2), (7, -2), (-7, -2), (7, 0)],
    "cube": [(10,), (-10,), (3037000500,), (2**21,)],
    "shifted": [(1, 3), (-8, 2), (1, 62), (1, 63), (5, 64), (5, -1)],
    "mixed": [(1.5, 3), (2.0, 0), (-1.0, -2)],
    "parity": [(4,), (3,), (-7,), (0,)],
    "hypot": [(3.0, 4.0), (0.0, 0.0), (-5.0, 12.0)],
    "clamp": [(5, 0, 3), (-2, 0, 3), (2, 0, 3)],
    "triangle": [(0,), (10,), (100000,)],
    "twice": [(3,), (2**21,)],
}


def _c_type(kind: str):  # type: ignore[no-untyped-def]
    return {"int": ctypes.c_int64, "float": ctypes.c_double, "bool": ctypes.c_int8}[kind]


def _call(engine, signature, arguments):  # type: ignore[no-untyped-def]
    """Call one native entry point through ctypes; (status, result)."""
    argument_types = [_c_type(p.kind) for p in signature.parameters]
    result_kind = {"i64": "int", "double": "float", "i8": "bool"}[signature.returns[0]]
    out = _c_type(result_kind)()
    prototype = ctypes.CFUNCTYPE(ctypes.c_int32, *argument_types, ctypes.POINTER(type(out)))
    function = prototype(engine.address(signature.symbol))
    converted = [
        t(v if p.kind != "bool" else int(v))
        for t, v, p in zip(argument_types, arguments, signature.parameters, strict=True)
    ]
    status = function(*converted, ctypes.byref(out))
    value = out.value
    if result_kind == "bool":
        value = bool(value)
    return status, value


def _lower_both(analyze, path: Path):  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.llvm import _definitions, _value_class_layouts
    from ppy_compiler.backend.llvm.ir_pipeline import lower_module_via_ir

    bundle = analyze(path, backend="llvm")
    symbols = bundle.symbols.modules["kernels"]
    analysis = bundle.analysis.modules["kernels"]
    tree = symbols.module.tree
    candidates = {}
    for owner, node in _definitions(tree):
        info = symbols.functions.get(node.name) if not owner else None
        if info is None:
            continue
        function_analysis = analysis.functions.get(info.qualname)
        if function_analysis is not None:
            candidates[info.qualname] = (info, function_analysis, node)
    layouts = _value_class_layouts(bundle)
    old = lower_module(analysis, candidates, layouts, safeguards="inline")
    new = lower_module_via_ir(analysis, candidates, layouts, safeguards="inline", opt_level=2)
    return old, new


@requires_llvm
def test_both_roads_lower_the_same_functions(write, analyze):
    path = write("kernels.ppy", KERNELS)
    old, new = _lower_both(analyze, path)
    assert sorted(new.functions) == sorted(old.functions)
    assert new.rejected == old.rejected
    for qualname, lowered in new.functions.items():
        assert lowered.signature == old.functions[qualname].signature


@requires_llvm
def test_both_roads_answer_alike_on_every_input(write, analyze):
    """Same value, same status -- including the calls that fall back to CPython."""
    path = write("kernels.ppy", KERNELS)
    old, new = _lower_both(analyze, path)
    old_engine = JitEngine(opt_level=2).open()
    old_engine.add(old.ir)
    old_engine.finalize()
    new_engine = JitEngine(opt_level=2).open()
    new_engine.add(new.ir)
    new_engine.finalize()
    checked = 0
    for name, cases in CASES.items():
        signature = new.functions[f"kernels.{name}"].signature
        for arguments in cases:
            expected = _call(old_engine, signature, arguments)
            got = _call(new_engine, signature, arguments)
            assert got == expected, f"{name}{arguments}: ir road {got}, ast road {expected}"
            checked += 1
    assert checked == sum(len(cases) for cases in CASES.values())


@requires_llvm
def test_fallbacks_happen_where_python_semantics_demand_them(write, analyze):
    path = write("kernels.ppy", KERNELS)
    _old, new = _lower_both(analyze, path)
    engine = JitEngine(opt_level=2).open()
    engine.add(new.ir)
    engine.finalize()
    functions = new.functions
    assert _call(engine, functions["kernels.cube"].signature, (10,)) == (STATUS_OK, 1000)
    assert _call(engine, functions["kernels.cube"].signature, (3037000500,))[0] == STATUS_FALLBACK
    assert _call(engine, functions["kernels.floor_mod"].signature, (-7, 2)) == (STATUS_OK, -3)
    assert _call(engine, functions["kernels.floor_mod"].signature, (7, 0))[0] == STATUS_FALLBACK
    # The one quotient a machine word cannot hold: Python answers 2**63, so
    # the guard takes it to CPython rather than trapping in the division.
    minimum = -(2**63)
    assert _call(engine, functions["kernels.floor_mod"].signature, (minimum, -1))[0] == (
        STATUS_FALLBACK
    )
    assert _call(engine, functions["kernels.shifted"].signature, (1, 63))[0] == STATUS_FALLBACK
    assert _call(engine, functions["kernels.shifted"].signature, (1, 3)) == (STATUS_OK, 8)
    assert _call(engine, functions["kernels.parity"].signature, (-7,)) == (STATUS_OK, True)
    assert _call(engine, functions["kernels.twice"].signature, (2**21,))[0] == STATUS_FALLBACK


def test_the_frontend_writes_verified_ir_with_the_semantics_spelled(write, analyze):
    path = write(
        "kernels.ppy",
        """
        def halve(n: int) -> int:
            return n // 2 + 1
        """,
    )
    bundle = analyze(path, backend="llvm")
    symbols = bundle.symbols.modules["kernels"]
    analysis = bundle.analysis.modules["kernels"]
    info = symbols.functions["halve"]
    node = next(n for n in symbols.module.tree.body if getattr(n, "name", "") == "halve")
    lowered = lower_module_to_ir(
        analysis, {info.qualname: (info, analysis.functions[info.qualname], node)}
    )
    assert not verify(lowered.module)
    text = encode(lowered.module)
    assert "func @kernels_halve(%n: i64) -> i64 attrs {" in text
    assert '"ppy.symbol" = "ppy_kernels_halve"' in text.replace("ppy.symbol =", '"ppy.symbol" =')
    assert "core.guard %" in text and 'kind = "zero_division"' in text
    assert "core.div %" in text and 'overflow = "python"' in text and 'rounding = "floor"' in text
    assert list(lowered.functions) == ["kernels.halve"]


def test_what_the_subset_excludes_is_refused_with_the_reason(write, analyze):
    path = write(
        "kernels.ppy",
        """
        def words(text: str) -> int:
            return len(text)


        def chained(a: int, b: int, c: int) -> bool:
            return a < b < c


        def caller(a: int, b: int, c: int) -> int:
            return 1 if chained(a, b, c) else 0
        """,
    )
    bundle = analyze(path, backend="llvm", strict=False)
    symbols = bundle.symbols.modules["kernels"]
    analysis = bundle.analysis.modules["kernels"]
    candidates = {}
    for node in symbols.module.tree.body:
        name = getattr(node, "name", None)
        if name and name in symbols.functions:
            info = symbols.functions[name]
            candidates[info.qualname] = (info, analysis.functions[info.qualname], node)
    lowered = lower_module_to_ir(analysis, candidates)
    assert not lowered.functions
    assert "no native ABI" in lowered.rejected["kernels.words"]
    assert lowered.rejected["kernels.chained"] == "chained comparison has no native lowering"
    assert "`kernels.chained` has no native lowering" in lowered.rejected["kernels.caller"]
    assert not verify(lowered.module), "dropped functions leave no half-built bodies behind"
