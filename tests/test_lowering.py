"""AST -> canonical IR -> LLVM: the one road, held to Python's own answers.

The IR road must answer as CPython does on every input, including which
calls fall back to it. The test here JIT-compiles the same functions and
calls them on the same inputs Python is given; the functions themselves
are the reference.
"""

from __future__ import annotations

import ctypes
import math
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.jit import JitEngine
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

_WORD = 2**63


def _python(name: str, arguments: tuple) -> tuple[int, object]:
    """What CPython answers, and whether the native code may answer it or must fall back."""
    namespace: dict = {}
    exec(textwrap.dedent(KERNELS), namespace)  # the kernels above, the reference
    try:
        value = namespace[name](*arguments)
    except (ZeroDivisionError, ValueError, OverflowError):
        return STATUS_FALLBACK, None
    if isinstance(value, bool) or not isinstance(value, int):
        return STATUS_OK, value
    if -_WORD <= value < _WORD and _intermediate_fits(name, arguments):
        return STATUS_OK, value
    return STATUS_FALLBACK, value


def _intermediate_fits(name: str, arguments: tuple) -> bool:
    if name == "shifted":
        n, k = arguments
        return 0 <= k < 64 and -_WORD <= n << k < _WORD
    if name == "cube":
        (n,) = arguments
        return -_WORD <= n * n < _WORD and -_WORD <= n * n * n < _WORD
    if name == "twice":
        (n,) = arguments
        return _intermediate_fits("cube", (n,)) and -_WORD <= 2 * n**3 < _WORD
    if name == "floor_mod":
        a, b = arguments
        return b != 0 and not (a == -_WORD and b == -1)
    return True


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


def _lower(analyze, path: Path):  # type: ignore[no-untyped-def]
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
    return lower_module_via_ir(analysis, candidates, layouts, safeguards="inline", opt_level=2)


@requires_llvm
def test_the_road_lowers_every_kernel_with_its_abi(write, analyze):
    path = write("kernels.ppy", KERNELS)
    lowered = _lower(analyze, path)
    assert sorted(lowered.functions) == sorted(f"kernels.{name}" for name in CASES)
    assert not lowered.rejected
    assert lowered.functions["kernels.mixed"].signature.returns == ("double",)
    assert lowered.functions["kernels.parity"].signature.returns == ("i8",)
    assert [p.kind for p in lowered.functions["kernels.clamp"].signature.parameters] == ["int"] * 3
    assert lowered.ppyir.startswith("ppyir"), "the module's own IR travels with the lowering"


@requires_llvm
def test_the_road_answers_as_python_does_on_every_input(write, analyze):
    """Same value, same status -- including the calls that fall back to CPython."""
    path = write("kernels.ppy", KERNELS)
    lowered = _lower(analyze, path)
    engine = JitEngine(opt_level=2).open()
    engine.add(lowered.ir)
    engine.finalize()
    checked = 0
    for name, cases in CASES.items():
        signature = lowered.functions[f"kernels.{name}"].signature
        for arguments in cases:
            expected_status, expected = _python(name, arguments)
            status, value = _call(engine, signature, arguments)
            assert status == expected_status, (
                f"{name}{arguments}: status {status}, python {expected_status}"
            )
            if status == STATUS_OK:
                if isinstance(expected, float):
                    assert math.isclose(value, expected, rel_tol=1e-12), f"{name}{arguments}"
                else:
                    assert value == expected, f"{name}{arguments}: {value}, python {expected}"
            checked += 1
    assert checked == sum(len(cases) for cases in CASES.values())


@requires_llvm
def test_fallbacks_happen_where_python_semantics_demand_them(write, analyze):
    path = write("kernels.ppy", KERNELS)
    lowered = _lower(analyze, path)
    engine = JitEngine(opt_level=2).open()
    engine.add(lowered.ir)
    engine.finalize()
    functions = lowered.functions
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


def test_the_direct_road_is_gone_and_asking_for_it_is_answered(write, tmp_path):
    """`pipeline = "ast"` and `PPY_LOWERING=ast` select the one road there is, with a word."""
    import os
    import subprocess
    import sys

    from ppy_compiler.driver.config import selected_pipeline

    assert selected_pipeline("ast") == "ir" and selected_pipeline(None) == "ir"
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ast"\n', encoding="utf-8"
    )
    (tmp_path / "prog.ppy").write_text(
        "def f(n: int) -> int:\n    return n + 1\n\n\nprint(f(2))\n", encoding="utf-8"
    )
    if not llvm_available():
        return
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "prog.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PPY_LOWERING": "ast"},
    )
    assert done.returncode == 0, done.stderr
    assert "W2004" in done.stderr and "direct AST road" in done.stderr, done.stderr
