"""Effect system v2, ownership, generics, and static dispatch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect, EffectSet
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.ir import decode

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")


def _codes(bundle) -> list[str]:  # type: ignore[no-untyped-def]
    return [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]


def _messages(bundle, code: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [d.message for d in bundle.diagnostics.sorted() if d.code == code]


# -- effects ------------------------------------------------------------------


def test_the_effect_vocabulary_spells_itself_and_reads_back():
    assert Effect.WRITE_MEMORY.spelled == "write_memory"
    assert Effect.IO.spelled == "io"
    assert Effect.GPU_LAUNCH.spelled == "gpu_launch"
    assert Effect.PYTHON_DYNAMIC.spelled == "python_dynamic"
    every = EffectSet(frozenset(Effect))
    assert EffectSet.parse(every.spelled()) == every
    with pytest.raises(ValueError, match="unknown effect 'teleport'"):
        EffectSet.parse(["teleport"])
    assert EffectSet.of(Effect.ALLOC, Effect.READ_MEMORY).is_removable
    assert not EffectSet.of(Effect.WRITE_MEMORY).is_removable
    assert not EffectSet.of(raises=("ValueError",)).is_removable
    assert Effect.NETWORK in EffectSet.of(Effect.NETWORK).violations()
    assert EffectSet.of(Effect.READ_MEMORY).is_pure
    assert not EffectSet.of(Effect.ATOMIC).is_pure
    assert EffectSet.of(Effect.IO).gpu_blockers == frozenset({Effect.IO})


def test_network_calls_carry_the_network_effect(write, analyze):
    path = write(
        "net.ppy",
        """
        import socket


        def open_it(host: str) -> str:
            return socket.gethostbyname(host)
        """,
    )
    bundle = analyze(path)
    effects = bundle.analysis.modules["net"].functions["net.open_it"].effects
    assert Effect.NETWORK in effects and Effect.IO in effects


@requires_llvm
def test_the_ir_carries_effects_and_dce_drops_an_unused_pure_call(write, analyze):
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules

    path = write(
        "fx.ppy",
        """
        from ppy import Buffer


        def square(n: int) -> int:
            return n * n


        def fill(xs: Buffer[int], n: int) -> int:
            for i in range(n):
                xs[i] = i
            return n


        def caller(n: int) -> int:
            square(n)
            return n + 1
        """,
    )
    bundle = analyze(path, backend="llvm")
    module = ir_modules(bundle)["fx"]
    square = module.functions["fx_square"]
    assert square.attributes["effects"] == ()
    fill = module.functions["fx_fill"]
    assert "write_memory" in fill.attributes["effects"]
    assert "read_memory" in fill.attributes["effects"]
    caller = module.functions["fx_caller"]
    assert "core.call" not in [op.name for op in caller.operations()], (
        "an unused call to a function with no effects is dead"
    )


# -- ownership ------------------------------------------------------------------


def test_a_borrow_may_not_be_returned_stored_or_written(write, analyze):
    path = write(
        "own.ppy",
        """
        import ppy
        from ppy import Borrowed, Buffer, Mut, Owned


        def keep(xs: Borrowed[list[int]]) -> list[int]:
            return xs


        def stash(xs: Borrowed[list[int]], into: list[list[int]]) -> None:
            into.append(xs)


        def write(xs: Borrowed[Buffer[int]], n: int) -> None:
            xs[0] = n


        def fine(xs: Mut[Buffer[int]], n: int) -> None:
            xs[0] = n


        def take(xs: Owned[list[int]]) -> list[int]:
            return xs
        """,
    )
    bundle = analyze(path)
    codes = _codes(bundle)
    assert (
        codes.count("E1611") == 1
        and "returns `xs`, which it only borrows" in _messages(bundle, "E1611")[0]
    )
    assert codes.count("E1612") == 1
    assert codes.count("E1613") == 1 and "borrows read-only" in _messages(bundle, "E1613")[0]
    assert "E1601" not in codes
    params = {p.name: p for p in bundle.symbols.modules["own"].functions["take"].params}
    assert params["xs"].facts.ownership == "owned"
    assert str(params["xs"].type) == "list[int]", "the marker leaves the type itself alone"


@requires_llvm
def test_ownership_reaches_the_ir_and_its_verifier(write, analyze):
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules
    from ppy_compiler.ir import verify
    from ppy_compiler.ir.dialects import core

    path = write(
        "own.ppy",
        """
        from ppy import Buffer, Mut


        def total(xs: Buffer[float], n: int) -> float:
            acc = 0.0
            for i in range(n):
                acc += xs[i]
            return acc


        def scale(xs: Mut[Buffer[float]], n: int, k: float) -> int:
            for i in range(n):
                xs[i] = xs[i] * k
            return n
        """,
    )
    bundle = analyze(path, backend="llvm")
    module = ir_modules(bundle)["own"]
    assert module.functions["own_total"].param_attributes[0]["ownership"] == "borrowed"
    assert module.functions["own_scale"].param_attributes[0]["ownership"] == "mut"
    # Returning a borrowed parameter is refused by the verifier itself.
    function = module.functions["own_total"]
    function.results = (function.params[0][1],)
    entry = function.entry
    assert entry is not None
    from ppy_compiler.ir import Builder

    for block in list(function.blocks()):
        for op in list(block.operations):
            if op.name == "core.ret":
                op.erase()
                core.ret(Builder(block), entry.arguments[0])
    errors = [e.message for e in verify(module)]
    assert any("borrowed and may not be returned" in e for e in errors)


def test_the_markers_are_importable_and_inert():
    import ppy

    assert ppy.Owned[int] is not None
    assert ppy.Borrowed.mode == "borrowed" and ppy.Mut.mode == "mut"


# -- generics ---------------------------------------------------------------------


GENERIC = """
    from typing import Protocol


    class Addable(Protocol):
        def __add__(self, other: "Addable") -> "Addable": ...


    def largest[T: int | float](a: T, b: T) -> T:
        return a if a > b else b


    def twice[T: Addable](a: T) -> T:
        return a + a


    def first[T](items: list[T]) -> T:
        return items[0]


    def use() -> float:
        x = largest(1, 2)
        y = largest(1.5, 2.5)
        z = first([1, 2, 3])
        return x + y + z
    """


def test_generics_declare_infer_and_substitute(write, analyze):
    path = write("gen.ppy", GENERIC)
    bundle = analyze(path)
    assert _codes(bundle) == []
    functions = bundle.symbols.modules["gen"].functions
    largest = functions["largest"]
    assert [str(v) for v in largest.type_params] == ["T"]
    assert str(largest.type_params[0].bound) == "int | float"
    assert str(largest.params[0].type) == "T"
    use = bundle.analysis.modules["gen"].functions["gen.use"]
    assert str(use.locals["x"]) == "int"
    assert str(use.locals["y"]) == "float"
    assert str(use.locals["z"]) == "int"
    assert bundle.symbols.specializations["gen.largest"] == {("int",), ("float",)}


def test_a_bound_is_checked_and_an_unbounded_parameter_has_no_operators(write, analyze):
    path = write(
        "gen.ppy",
        """
        def largest[T: int | float](a: T, b: T) -> T:
            return a if a > b else b


        def join[T](a: T, b: T) -> T:
            return a + b


        def use() -> str:
            return largest("a", "b")
        """,
    )
    bundle = analyze(path)
    codes = _codes(bundle)
    assert "E1721" in codes, "str does not satisfy int | float"
    assert "E1302" in codes and "has no bound" in _messages(bundle, "E1302")[0]


def test_specialization_explosions_are_refused(write, analyze):
    path = write(
        "gen.ppy",
        """
        def wrap[T](x: T) -> list[T]:
            return wrap([x])


        def ident[T](x: T) -> T:
            return x


        def use() -> int:
            ident(1)
            ident(1.5)
            ident("s")
            return 0
        """,
    )
    (path.parent / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\n\n[tool.ppy.generics]\nmax-specializations = 2\n",
        encoding="utf-8",
    )
    bundle = analyze(path)
    codes = _codes(bundle)
    assert "E1723" in codes, "wrap[T] calls wrap[list[T]]: the specializations never end"
    assert "E1722" in codes, "ident is called on three types; the project allows two"


@requires_llvm
def test_a_generic_is_monomorphized_where_native_code_calls_it(write, analyze):
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules

    path = write(
        "gen.ppy",
        """
        def largest[T: int | float](a: T, b: T) -> T:
            return a if a > b else b


        def ints(a: int, b: int) -> int:
            return largest(a, b) + 1


        def floats(a: float, b: float) -> float:
            return largest(a, b) * 2.0
        """,
    )
    bundle = analyze(path, backend="llvm")
    module = ir_modules(bundle)["gen"]
    names = sorted(module.functions)
    assert "gen_largest__int" in names and "gen_largest__float" in names
    assert "gen_largest" not in names, "the generic itself has no single native body"
    instance = module.functions["gen_largest__int"]
    assert instance.attributes["ppy.generic"] == "gen.largest"
    assert instance.attributes["ppy.type_arguments"] == ("int",)
    assert [str(t) for _n, t in instance.params] == ["i64", "i64"]
    assert [str(t) for _n, t in module.functions["gen_largest__float"].params] == ["f64", "f64"]
    calls = [
        op.attributes["callee"].name
        for op in module.functions["gen_ints"].operations()
        if op.name == "core.call"
    ]
    assert calls == ["gen_largest__int"]


@requires_llvm
def test_the_three_paths_agree_on_a_generic_program(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(
        """\
def largest[T: int | float](a: T, b: T) -> T:
    return a if a > b else b


def clamp[T: int | float](x: T, lo: T, hi: T) -> T:
    return largest(lo, x) if x < hi else hi


def run(n: int) -> int:
    total = 0
    for i in range(n):
        total += clamp(i * 3, 2, 20)
    return total


def runf(x: float) -> float:
    return clamp(x, 0.5, 1.5) + largest(x, 2.0)


print(run(10), runf(0.25), runf(3.0))
""",
        encoding="utf-8",
    )
    outputs = []
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = subprocess.run(
            [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout.strip())
    assert outputs[0] == outputs[1] == outputs[2] == "125 2.5 4.5"
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "ir", "prog.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    module = decode(emitted.stdout)
    assert "prog_clamp__int" in module.functions and "prog_clamp__float" in module.functions


@requires_llvm
def test_a_value_class_operator_dispatches_statically(write, analyze):
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules

    path = write(
        "vec.ppy",
        """
        from dataclasses import dataclass


        @dataclass
        class Money:
            cents: int

            def __add__(self, other: "Money") -> "Money":
                return Money(self.cents + other.cents)


        def total_cents(a: Money, b: Money) -> int:
            return a.cents + b.cents
        """,
    )
    bundle = analyze(path, backend="llvm")
    module = ir_modules(bundle)["vec"]
    assert "vec_total_cents" in module.functions
    assert T.INT is not None
