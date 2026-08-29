"""Type, effect, refinement, and validity analysis (spec 8, 9, 11, 12)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect
from ppy_compiler.analysis.refinements import IntRange, width_range


def test_numeric_promotion_follows_python_typing():
    assert T.is_assignable(T.INT, T.FLOAT)
    assert T.is_assignable(T.BOOL, T.INT)
    assert not T.is_assignable(T.FLOAT, T.INT)
    assert not T.is_assignable(T.STR, T.INT)


def test_unions_normalize_and_absorb_literals():
    union = T.union(T.Literal(3, T.INT), T.INT, T.NONE)
    assert isinstance(union, T.Union_)
    assert set(union.members) == {T.INT, T.NONE}
    assert T.is_optional(union)
    assert T.remove_none(union) == T.INT


def test_container_invariance_and_tuple_covariance():
    assert T.is_assignable(T.list_of(T.INT), T.list_of(T.INT))
    assert not T.is_assignable(T.list_of(T.INT), T.list_of(T.FLOAT))
    assert T.is_assignable(T.Tuple_((T.INT, T.INT)), T.Tuple_((T.FLOAT, T.FLOAT)))


def test_int_range_arithmetic():
    assert IntRange(1, 3) + IntRange(2, 4) == IntRange(3, 7)
    assert IntRange(1, 3) - IntRange(2, 4) == IntRange(-3, 1)
    assert IntRange(-2, 3) * IntRange(2, 4) == IntRange(-8, 12)
    assert width_range(8, True) == IntRange(-128, 127)
    assert width_range(8, False) == IntRange(0, 255)
    assert width_range(64, True).contains(IntRange(0, 10))


def test_purity_is_verified(write, analyze):
    path = write(
        "pure_ok.ppy",
        """
        import ppy

        @ppy.pure
        def square(x: int) -> int:
            return x * x
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()
    report = bundle.reports["pure_ok.square"]
    assert report.pure and report.pure_declared


def test_false_purity_claim_is_rejected(write, codes):
    path = write(
        "impure.ppy",
        """
        import ppy

        @ppy.pure
        def shout(x: int) -> int:
            print(x)
            return x
        """,
    )
    assert "E1601" in codes(path)


def test_pure_function_calling_unknown_effects_is_rejected(write, codes):
    path = write(
        "unknown.ppy",
        """
        import ppy
        import somewhere_unknown

        @ppy.pure
        def f(x: int) -> int:
            return somewhere_unknown.thing(x)
        """,
    )
    found = codes(path)
    assert "E1602" in found


def test_pure_function_may_allocate_and_raise(write, analyze):
    path = write(
        "alloc.ppy",
        """
        import ppy

        @ppy.pure
        def pairs(n: int) -> list[int]:
            if n < 0:
                raise ValueError("negative")
            return [n, n]
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()
    analysis = bundle.analysis.function("alloc.pairs")
    assert Effect.ALLOC in analysis.effects
    assert analysis.effects.may_raise
    assert analysis.verified_pure


def test_mutating_a_parameter_breaks_purity(write, codes):
    path = write(
        "mutate.ppy",
        """
        import ppy

        @ppy.pure
        def push(xs: list[int]) -> list[int]:
            xs.append(1)
            return xs
        """,
    )
    assert "E1601" in codes(path)


def test_unannotated_parameter_is_an_error_not_implicit_any(write, codes):
    path = write(
        "implicit.ppy",
        """
        def f(x):
            return x
        """,
    )
    assert "E1201" in codes(path)


def test_bare_generic_is_rejected(write, codes):
    path = write(
        "bare.ppy",
        """
        def f(xs: list) -> int:
            return len(xs)
        """,
    )
    assert "E1203" in codes(path)


@pytest.mark.parametrize(
    ("snippet", "code"),
    [
        ("eval('1')", "E1501"),
        ("exec('x = 1')", "E1501"),
        ("globals()['x'] = 1", "E1502"),
        ("getattr(object(), name)", "E1504"),
    ],
)
def test_dynamic_features_are_rejected_in_strict_mode(write, codes, snippet, code):
    path = write(
        "dyn.ppy",
        f"""
        name: str = "x"

        def f() -> None:
            {snippet}
        """,
    )
    assert code in codes(path)


def test_dynamic_boundary_permits_dynamic_features(write, analyze):
    path = write(
        "boundary.ppy",
        """
        import ppy

        def f() -> None:
            with ppy.dynamic:
                eval("1 + 1")
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()
    assert bundle.analysis.function("boundary.f").dynamic


def test_dynamic_boundary_can_be_denied_by_configuration(write, codes):
    path = write(
        "denied.ppy",
        """
        import ppy

        def f() -> None:
            with ppy.dynamic:
                eval("1 + 1")
        """,
    )
    assert "E1505" in codes(path, dynamic_boundaries="deny")


def test_a_computed_base_is_rejected(write, codes):
    path = write(
        "built.ppy",
        """
        def factory() -> object:
            return object


        class Node(factory()):
            pass
        """,
    )
    assert "E1507" in codes(path)


def test_an_unvouched_metaclass_is_rejected(write, codes):
    path = write(
        "meta.ppy",
        """
        class Meta(type):
            pass


        class Leaf(metaclass=Meta):
            pass
        """,
    )
    assert "E1507" in codes(path)


def test_a_vouched_metaclass_is_accepted(write, codes):
    path = write(
        "abstract.ppy",
        """
        import abc


        class Base(metaclass=abc.ABCMeta):
            pass
        """,
    )
    assert "E1507" not in codes(path)


def test_a_dynamic_boundary_permits_dynamic_class_construction(write, codes):
    path = write(
        "escape.ppy",
        """
        import ppy


        def factory() -> object:
            return object


        with ppy.dynamic:
            class Node(factory()):
                pass
        """,
    )
    assert "E1507" not in codes(path)


def test_ppy_dynamic_is_an_explicit_any_boundary(write, analyze):
    path = write(
        "boundary_type.ppy",
        """
        import ppy


        def normalize(payload: ppy.Dynamic) -> int:
            return int(payload)


        print(normalize("5"))
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()


def test_fixed_width_violation_is_rejected(write, codes):
    path = write(
        "width.ppy",
        """
        from ppy import i8

        def f() -> i8:
            value: i8 = 999
            return value
        """,
    )
    assert "E1401" in codes(path)


def test_fixed_width_within_range_is_accepted(write, analyze):
    path = write(
        "width_ok.ppy",
        """
        from ppy import u8

        def f() -> u8:
            value: u8 = 200
            return value
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_narrowing_through_isinstance_and_none(write, analyze):
    path = write(
        "narrow.ppy",
        """
        def f(x: int | None) -> int:
            if x is None:
                return 0
            return x

        def g(x: int | str) -> int:
            if isinstance(x, str):
                return len(x)
            return x
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_a_union_is_assignable_whatever_order_its_members_are_in():
    left = T.union(T.NONE, T.INT)
    right = T.union(T.INT, T.NONE)
    assert T.is_assignable(left, right)
    assert T.is_assignable(right, left)
    assert not T.is_assignable(right, T.INT)


def test_boolean_operators_narrow_their_later_operands(write, analyze):
    path = write(
        "boolop.ppy",
        """
        def guarded(x: str | None) -> bool:
            return x is not None and len(x) > 0

        def defaulted(x: str | None) -> int:
            if x is None or len(x) == 0:
                return 0
            return len(x)
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_a_match_case_sees_what_earlier_cases_ruled_out(write, analyze):
    path = write(
        "cases.ppy",
        """
        def describe(value: int | str | None) -> str:
            match value:
                case None:
                    return "none"
                case int():
                    return f"int:{value + 1}"
                case _:
                    return value.upper()
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_a_guarded_case_rules_nothing_out(write, codes):
    path = write(
        "guarded.ppy",
        """
        def describe(value: int | None) -> str:
            match value:
                case None if True:
                    return "none"
                case _:
                    return str(value + 1)
        """,
    )
    assert "E1302" in codes(path)


def test_an_empty_container_is_absorbed_by_a_populated_one():
    empty = T.list_of(T.NEVER)
    populated = T.list_of(T.INT)
    assert T.union(empty, populated) == populated
    assert T.union(populated, empty) == populated
    assert T.union(T.list_of(T.INT), T.list_of(T.STR)) != T.list_of(T.INT)


def test_appending_gives_an_empty_list_its_element_type(write, analyze):
    path = write(
        "grow.ppy",
        """
        def build(count: int) -> list[int]:
            out = []
            for i in range(count):
                out.append(i * 2)
            return out
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_missing_narrowing_is_reported(write, codes):
    path = write(
        "no_narrow.ppy",
        """
        def f(x: int | None) -> int:
            return x
        """,
    )
    assert "E1303" in codes(path)


def test_call_arity_and_argument_types_are_checked(write, codes):
    path = write(
        "arity.ppy",
        """
        def f(a: int, b: int) -> int:
            return a + b

        def g() -> int:
            return f(1)

        def h() -> int:
            return f(1, "two")
        """,
    )
    found = codes(path)
    assert "E1305" in found and "E1301" in found


def test_operator_type_errors_are_reported(write, codes):
    path = write(
        "ops.ppy",
        """
        def f(a: int, b: str) -> int:
            return a - b
        """,
    )
    assert "E1302" in codes(path)


def test_bool_arithmetic_warns_but_is_allowed(write, analyze):
    path = write(
        "boolarith.ppy",
        """
        def f(a: bool, b: bool) -> int:
            return a + b
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()
    assert "W2002" in [d.code for d in bundle.diagnostics]


def test_range_induction_variable_is_bounded(write, analyze):
    path = write(
        "induction.ppy",
        """
        from ppy import u8

        def f() -> int:
            total: int = 0
            for i in range(10):
                value: u8 = i
                total += value
            return total
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()


def test_classes_and_attributes_are_checked(write, codes):
    path = write(
        "cls.ppy",
        """
        class Point:
            x: int
            y: int

            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

            def norm(self) -> int:
                return self.x * self.x + self.y * self.y

        def use() -> int:
            p = Point(1, 2)
            return p.missing
        """,
    )
    assert "E1202" in codes(path)


def test_inheritance_resolves_through_the_mro(write, analyze):
    path = write(
        "inherit.ppy",
        """
        class Base:
            value: int

            def get(self) -> int:
                return self.value

        class Child(Base):
            extra: int

        def use(c: Child) -> int:
            return c.get() + c.value + c.extra
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_module_graph_tracks_cross_module_types(write, analyze):
    write(
        "lib.ppy",
        """
        def double(x: int) -> int:
            return x * 2
        """,
    )
    path = write(
        "app.ppy",
        """
        import lib

        def use() -> str:
            return lib.double(2)
        """,
    )
    codes = [d.code for d in analyze(path).diagnostics]
    assert "E1303" in codes


def test_star_import_is_rejected(write, codes):
    write("lib2.ppy", "VALUE: int = 1\n")
    path = write(
        "star.ppy",
        """
        from lib2 import *
        """,
    )
    assert "E1103" in codes(path)


def test_match_statement_narrows(write, analyze):
    path = write(
        "match.ppy",
        """
        def f(x: int | str) -> int:
            match x:
                case int():
                    return x
                case _:
                    return 0
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_a_module_level_type_alias_is_resolved(write, analyze):
    path = write(
        "alias.ppy",
        """
        from typing import Annotated

        import ppy

        Pixel = Annotated[int, ppy.Range(0, 255)]
        Row = list[Pixel]


        def brighten(value: Pixel, row: Row) -> Pixel:
            return value + len(row)
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    info = bundle.symbols.functions["alias.brighten"]
    assert str(info.params[0].type) == "int"
    assert info.params[0].facts.int_range.high == 255
    assert str(info.params[1].type) == "list[int]"


def test_the_312_type_statement_is_resolved(write, analyze):
    path = write(
        "typestmt.ppy",
        """
        type Weight = float
        type Weights = list[Weight]


        def total(values: Weights) -> Weight:
            result: Weight = 0.0
            for value in values:
                result += value
            return result
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    assert str(bundle.symbols.functions["typestmt.total"].ret) == "float"


def test_a_recursive_alias_does_not_hang(write, analyze):
    path = write(
        "cycle.ppy",
        """
        Loop = list[Loop]


        def f(x: Loop) -> int:
            return len(x)
        """,
    )
    # The alias cannot be resolved, but analysis must terminate and say so.
    codes = [d.code for d in analyze(path).diagnostics]
    assert "E1101" in codes


def test_a_value_assignment_is_not_mistaken_for_an_alias(write, analyze):
    path = write(
        "notalias.ppy",
        """
        LIMIT = 255
        NAMES = ["a", "b"]


        def f(x: int) -> int:
            return x + LIMIT + len(NAMES)
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()
    assert "LIMIT" not in bundle.symbols.modules["notalias"].type_aliases
    assert "NAMES" not in bundle.symbols.modules["notalias"].type_aliases


def test_a_parameter_shadows_a_module_global_of_the_same_name(write, analyze):
    """Reading a parameter is not a global dependency, whatever it is called."""
    path = write(
        "shadow.ppy",
        """
        import ppy

        total: int = 0


        @ppy.pure
        def add(total: int, extra: int) -> int:
            return total + extra
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]

    analysis = bundle.analysis.function("shadow.add")
    assert analysis.verified_pure
    assert Effect.READ_GLOBAL not in analysis.effects


def test_a_local_shadows_a_module_global_of_the_same_name(write, analyze):
    path = write(
        "localshadow.ppy",
        """
        import ppy

        count: int = 5


        @ppy.pure
        def compute(n: int) -> int:
            count: int = n * 2
            return count
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    assert bundle.analysis.function("localshadow.compute").verified_pure


def test_reading_a_real_mutable_global_is_still_an_effect(write, analyze):
    path = write(
        "realglobal.ppy",
        """
        import ppy

        counter: int = 0


        def bump() -> None:
            global counter
            counter = counter + 1


        @ppy.pure
        def read() -> int:
            return counter
        """,
    )
    codes = [d.code for d in analyze(path).diagnostics]
    assert "E1601" in codes


def test_a_pure_function_may_fill_a_container_it_allocated(write, analyze):
    """Spec 11.2 allows allocation whose identity is not shared before return."""
    path = write(
        "local_alloc.ppy",
        """
        import ppy

        @ppy.pure
        def squares(count: int) -> list[int]:
            out = []
            for i in range(count):
                out.append(i * i)
            return out

        @ppy.pure
        def distinct(values: list[str]) -> int:
            seen = set()
            for value in values:
                seen.add(value)
            return len(seen)
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_a_pure_function_may_not_mutate_a_parameter(write, codes):
    path = write(
        "taint.ppy",
        """
        import ppy

        @ppy.pure
        def push(xs: list[int]) -> int:
            xs.append(1)
            return len(xs)
        """,
    )
    assert "E1601" in codes(path)


def test_a_local_shared_before_it_is_mutated_is_not_pure(write, codes):
    path = write(
        "shared.ppy",
        """
        import ppy

        def keep(values: list[int]) -> int:
            return len(values)

        @ppy.pure
        def leaks() -> int:
            out = []
            total: int = keep(out)
            out.append(1)
            return total
        """,
    )
    assert "E1601" in codes(path)


def test_a_dynamic_boundary_accepts_the_called_marker(write, analyze):
    path = write(
        "called.ppy",
        """
        import ppy

        def reflect(name: str) -> str:
            with ppy.dynamic():
                return str(getattr("abc", name)())
        """,
    )
    assert not analyze(path).diagnostics.has_errors()
    assert analyze(path).analysis.function("called.reflect").dynamic


def test_a_callee_that_mutates_a_borrowed_argument_breaks_purity(write, codes):
    """The write lands on the caller's list, so the caller is not pure either."""
    path = write(
        "indirect.ppy",
        """
        import ppy

        def sneaky(a: list[int], b: list[int]) -> None:
            a.append(1)
            b.append(2)

        @ppy.pure
        def caller(theirs: list[int]) -> list[int]:
            out: list[int] = []
            sneaky(out, theirs)
            return out
        """,
    )
    assert "E1601" in codes(path)


def test_a_callee_handed_only_local_allocations_keeps_purity(write, analyze):
    path = write(
        "indirect_ok.ppy",
        """
        import ppy

        def fill(target: list[int], count: int) -> None:
            for i in range(count):
                target.append(i)

        @ppy.pure
        def caller(count: int) -> list[int]:
            out: list[int] = []
            fill(out, count)
            return out
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_reading_through_an_optional_names_the_real_problem(write, analyze):
    """The old message blamed a missing stub, which sent readers the wrong way."""
    path = write(
        "optional_attr.ppy",
        """
        class Box:
            def __init__(self, size: int) -> None:
                self.size: int = size

            def grow(self) -> "Box":
                return Box(self.size + 1)

        def find(flag: bool) -> Box | None:
            return Box(1) if flag else None

        def use(flag: bool) -> int:
            return find(flag).grow().size
        """,
    )
    diagnostics = analyze(path).diagnostics.sorted()
    codes = [d.code for d in diagnostics]
    assert "E1206" in codes
    assert "E1306" not in codes, "the unknown-signature cascade was not suppressed"
    reported = next(d for d in diagnostics if d.code == "E1206")
    assert "may be `None`" in reported.message


def test_a_narrowed_optional_reads_cleanly(write, analyze):
    path = write(
        "narrowed_attr.ppy",
        """
        class Box:
            def __init__(self, size: int) -> None:
                self.size: int = size

        def find(flag: bool) -> Box | None:
            return Box(1) if flag else None

        def use(flag: bool) -> int:
            found = find(flag)
            if found is None:
                return 0
            return found.size
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_a_dynamic_boundary_still_allows_an_optional_receiver(write, analyze):
    path = write(
        "dyn_attr.ppy",
        """
        import ppy

        class Box:
            def __init__(self, size: int) -> None:
                self.size: int = size

        def find(flag: bool) -> Box | None:
            return Box(1) if flag else None

        @ppy.dynamic
        def use(flag: bool) -> int:
            return find(flag).size
        """,
    )
    assert "E1206" not in [d.code for d in analyze(path).diagnostics.sorted()]


def test_zero_argument_super_resolves_to_the_base(write, analyze):
    path = write(
        "inherit.ppy",
        """
        class Shape:
            def __init__(self, name: str) -> None:
                self.name: str = name

            def area(self) -> float:
                return 0.0

        class Circle(Shape):
            def __init__(self, radius: float) -> None:
                super().__init__("circle")
                self.radius: float = radius

            def area(self) -> float:
                return 3.141592653589793 * self.radius * self.radius
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_super_outside_a_class_is_still_unresolved(write, codes):
    path = write(
        "loose.ppy",
        """
        def f() -> int:
            return super().thing()
        """,
    )
    assert "E1101" in codes(path)


def test_the_runtime_import_hook_api_is_typed(write, analyze):
    path = write(
        "hook.ppy",
        """
        import ppy

        ppy.install()

        def installed() -> bool:
            return ppy.is_installed()
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_threading_is_typed(write, analyze):
    path = write(
        "threads.ppy",
        """
        import threading

        def spawn(count: int) -> int:
            workers: list[threading.Thread] = []
            for _ in range(count):
                workers.append(threading.Thread())
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            return len(workers)

        def guarded(lock: threading.Lock) -> bool:
            acquired: bool = lock.acquire()
            lock.release()
            return acquired
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_threading_carries_its_effects(write, analyze):
    path = write(
        "threadfx.ppy",
        """
        import ppy
        import threading

        @ppy.pure
        def spawn() -> None:
            threading.Thread().start()
        """,
    )
    codes = [d.code for d in analyze(path).diagnostics.sorted()]
    assert "E1601" in codes


def test_required_native_rejects_a_signature_with_no_native_abi(write, codes):
    """A clean body is not enough; the signature has to reach the native ABI."""
    path = write(
        "require_native.ppy",
        """
        import ppy

        @ppy.native(require=True)
        def joined(values: list[str]) -> str:
            return ",".join(values)
        """,
    )
    assert "E1702" in codes(path)


def test_required_native_accepts_a_buffer_signature(write, analyze):
    path = write(
        "require_ok.ppy",
        """
        import ppy

        @ppy.native(require=True)
        def total(values: list[float]) -> float:
            out: float = 0.0
            for value in values:
                out += value
            return out
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_native_report_matches_what_the_backend_does(write, analyze):
    """The contract and the lowering must agree about the same function."""
    from ppy_compiler.analysis.contracts import native_report

    path = write(
        "agree.ppy",
        """
        def scalars(a: int, b: float) -> float:
            return a + b

        def strings(values: list[str]) -> int:
            return len(values)

        def buffers(values: list[int]) -> int:
            out: int = 0
            for value in values:
                out += value
            return out
        """,
    )
    functions = analyze(path, backend="llvm").analysis.modules["agree"].functions
    assert native_report(functions["agree.scalars"])[0]
    assert native_report(functions["agree.buffers"])[0]
    assert not native_report(functions["agree.strings"])[0]


def test_a_closed_set_of_literals_renders_as_literal():
    from ppy_compiler.analysis.render import render_annotation

    closed = T.union(T.Literal("a", T.STR), T.Literal("b", T.STR), T.Literal("c", T.STR))
    assert render_annotation(closed, closed_literals=True).text == "Literal['a', 'b', 'c']"
    assert render_annotation(closed, closed_literals=True).typing_imports == frozenset({"Literal"})
    # Off by default, because a parameter's call sites are only a sample.
    assert render_annotation(closed).text == "str"


def test_a_single_literal_is_widened_rather_than_pinned():
    from ppy_compiler.analysis.render import render_annotation

    assert render_annotation(T.Literal(3, T.INT), closed_literals=True).text == "int"


def test_a_float_literal_is_never_written_as_literal():
    """PEP 586 does not admit `float`, whatever the runtime accepts."""
    from ppy_compiler.analysis.render import render_annotation

    floats = T.union(T.Literal(1.5, T.FLOAT), T.Literal(2.5, T.FLOAT))
    assert render_annotation(floats, closed_literals=True).text == "float"


def test_an_optional_closed_set_keeps_none_last():
    from ppy_compiler.analysis.render import render_annotation

    optional = T.union(T.Literal("x", T.STR), T.Literal("y", T.STR), T.NONE)
    assert render_annotation(optional, closed_literals=True).text == "Literal['x', 'y'] | None"


def test_a_wide_literal_set_is_widened():
    from ppy_compiler.analysis.render import render_annotation

    many = T.union(*[T.Literal(i, T.INT) for i in range(20)])
    assert render_annotation(many, closed_literals=True).text == "int"


def test_widening_two_members_to_one_text_does_not_repeat_it():
    """`Literal['a'] | Literal['b']` widened is `str`, not `str | str`."""
    from ppy_compiler.analysis.render import render_annotation

    pair = T.union(T.Literal("a", T.STR), T.Literal(1, T.INT))
    assert render_annotation(pair).text in {"str | int", "int | str"}
    same = T.union(T.Literal("a", T.STR), T.Literal("b", T.STR))
    assert render_annotation(same).text == "str"


def test_the_recording_pass_is_part_of_the_fixpoint(write, analyze, monkeypatch):
    """Confirming convergence and recording are the same traversal."""
    from ppy_compiler.analysis import checker as checker_module

    passes = 0
    original = checker_module._Checker.check_module

    def counted(self):
        nonlocal passes
        passes += 1
        return original(self)

    monkeypatch.setattr(checker_module._Checker, "check_module", counted)
    path = write(
        "converged.ppy",
        """
        def a(x: int) -> int:
            return b(x) + 1

        def b(x: int) -> int:
            return x * 2
        """,
    )
    analyze(path)
    # One silent seeding pass and one recording pass that changed nothing.
    assert passes == 2, f"{passes} traversals for one module"


def test_diagnostics_survive_a_pass_that_had_to_be_repeated(write, codes):
    """A pass whose summaries moved is discarded, so nothing is reported twice."""
    path = write(
        "chain.ppy",
        """
        def top(x: int) -> int:
            return middle(x)

        def middle(x: int) -> int:
            return bottom(x)

        def bottom(x):
            return x
        """,
    )
    reported = codes(path)
    assert reported.count("E1201") == 1


def test_widening_offers_only_what_the_protocol_has(tmp_path: Path):
    """`reversed()` is a `Sequence` operation, not a `Mapping` one.

    `dict` happens to be reversible; `Mapping` declares no `__reversed__`, so
    a body that reverses its argument may not have it widened -- one allowlist
    shared across protocols would let this through.
    """
    from ppy_compiler.driver.convert import build_plan, convert_source
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")

    def convert(body: str, call: str) -> str:
        path = tmp_path / "rev.py"
        path.write_text(
            f"def newest(events):\n{body}\n\n\nprint(newest({call}))\n", encoding="utf-8"
        )
        bundle = analyze_paths(open_project(path), [path], backend="python")
        plan, _ = build_plan(bundle, "rev")
        return convert_source(path.read_text(encoding="utf-8"), plan)

    loop = "    for item in reversed(events):\n        return item\n    return 0"
    assert "events: Sequence[int]" in convert(loop, "[1, 2, 3]")
    assert "events: dict[str, int]" in convert(loop.replace("return 0", 'return ""'), '{"a": 1}')


def test_the_call_binder_follows_python_argument_rules():
    import ast

    from ppy_compiler.analysis.binding import bind_call, positional_values
    from ppy_compiler.analysis.symbols import ParamInfo

    def param(name: str, kind: str = "positional_or_keyword") -> ParamInfo:
        return ParamInfo(name, T.INT, kind=kind)

    params = [
        param("self"),
        param("a"),
        param("b"),
        param("rest", "var_positional"),
        param("flag", "keyword_only"),
    ]

    def names(bound):
        return [(b.param.name, b.value) for b in bound]

    # A bound receiver shifts every written position by one.
    assert names(bind_call(params, ["x", "y"], offset=1)) == [("a", "x"), ("b", "y")]
    # A keyword reaches its parameter wherever it sits.
    assert names(bind_call(params, [], [("flag", "f")], offset=1)) == [("flag", "f")]
    # Positional arguments stop at `*rest` rather than spilling into it.
    assert names(bind_call(params, ["x", "y", "z", "w"], offset=1)) == [("a", "x"), ("b", "y")]
    # `**splat` names no parameter, so it binds nothing.
    assert not bind_call(params, [], [(None, "s")], offset=1)
    # The receiver is not addressable from the call site.
    assert not bind_call(params, [], [("self", "s")], offset=1)

    call = ast.parse("f(a, b, *rest, c)").body[0].value
    assert len(positional_values(call.args)) == 2


def test_the_alias_map_follows_python_binding():
    import ast

    from ppy_compiler.analysis.aliasing import EXTERNAL, analyze_aliases

    source = textwrap.dedent(
        """
        def f(xs, ys):
            a = xs
            b = a
            b.append(1)          # chain: mutates xs
            a = []
            a.append(2)          # killed: local now
            c = xs if ys else ys
            c.append(3)          # join: may be either parameter
            pair = (xs, [0])
            left, right = pair
            left.append(4)       # through the tuple: may be xs
            for row in [xs]:
                row.append(5)    # iteration over a literal: may be xs
            fresh = list(xs)
            fresh.append(6)      # a copy shares nothing
            return xs
        """
    )
    fn = ast.parse(source).body[0]
    info = analyze_aliases(fn)

    def roots_of(receiver: str, ordinal: int = 0) -> frozenset:
        seen = 0
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == receiver
            ):
                if seen == ordinal:
                    return info.roots_at(node.value, receiver)
                seen += 1
        raise AssertionError(receiver)

    assert info.param_roots(roots_of("b")) == {"xs"}
    assert info.only_local(roots_of("a"))
    assert info.param_roots(roots_of("c")) == {"xs", "ys"}
    assert "xs" in info.param_roots(roots_of("left"))
    assert "xs" in info.param_roots(roots_of("row"))
    assert info.only_local(roots_of("fresh"))
    # A name never seen answers conservatively.
    assert EXTERNAL in info.roots_at(ast.Name(id="ghost"), "ghost")


def test_purity_survives_mutating_a_local_through_an_alias(write, analyze):
    path = write(
        "alias_pure.ppy",
        """
        import ppy

        @ppy.pure
        def build(count: int) -> int:
            out = []
            tail = out
            for i in range(count):
                tail.append(i)
            return len(out)
        """,
    )
    assert not analyze(path).diagnostics.has_errors()


def test_purity_rejects_mutating_a_parameter_through_an_alias(write, codes):
    path = write(
        "alias_impure.ppy",
        """
        import ppy

        @ppy.pure
        def push(xs: list[int]) -> int:
            ys = xs
            ys.append(1)
            return len(xs)
        """,
    )
    assert "E1601" in codes(path)


def test_inference_settles_a_deep_call_chain(tmp_path: Path):
    """Ten levels of indirection, evidence entering only at the bottom.

    A fixed round count would type however many levels the count allows and
    freeze the rest; a fixpoint types all of them or is a bug.
    """
    from ppy_compiler.analysis.inference import refine_with_call_sites
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    depth = 10
    lines = ["def level0(x):", "    return x * 2.0", ""]
    for level in range(1, depth):
        lines += [f"def level{level}(x):", f"    return level{level - 1}(x)", ""]
    lines.append(f"print(level{depth - 1}(1.5))")
    path = tmp_path / "chain.py"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    project = open_project(path)
    project.config.strict = False
    bundle = analyze_paths(project, [path], backend="python")
    refine_with_call_sites(bundle, bundle.diagnostics)
    assert not any(d.code == "E9001" for d in bundle.diagnostics)
    for level in range(depth):
        info = bundle.symbols.functions[f"chain.level{level}"]
        assert str(info.params[0].type) == "float", f"level{level} was left untyped"
        assert str(info.ret) == "float"


def test_inference_survives_mutual_recursion(tmp_path: Path):
    from ppy_compiler.analysis.inference import refine_with_call_sites
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    path = tmp_path / "mutual.py"
    path.write_text(
        textwrap.dedent(
            """
            def is_even(n):
                if n == 0:
                    return True
                return is_odd(n - 1)


            def is_odd(n):
                if n == 0:
                    return False
                return is_even(n - 1)


            print(is_even(10), is_odd(7))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    project = open_project(path)
    project.config.strict = False
    bundle = analyze_paths(project, [path], backend="python")
    refine_with_call_sites(bundle, bundle.diagnostics)
    assert not any(d.code == "E9001" for d in bundle.diagnostics)
    for name in ("mutual.is_even", "mutual.is_odd"):
        info = bundle.symbols.functions[name]
        assert str(info.params[0].type) == "int"
        assert T.is_assignable(info.ret, T.BOOL)


def test_late_keyword_evidence_still_reaches_the_callee(tmp_path: Path):
    """The dict arrives by keyword, from a caller typed on a later round."""
    from ppy_compiler.analysis.inference import refine_with_call_sites
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    path = tmp_path / "late.py"
    path.write_text(
        textwrap.dedent(
            """
            def sink(data):
                return len(data)


            def direct():
                return sink([1])


            def relay(payload):
                return sink(data=payload)


            print(direct())
            print(relay({"a": 1}))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    project = open_project(path)
    project.config.strict = False
    bundle = analyze_paths(project, [path], backend="python")
    refine_with_call_sites(bundle, bundle.diagnostics)
    sink = bundle.symbols.functions["late.sink"]
    assert "dict[str, int]" in str(sink.params[0].type)
    assert "list[int]" in str(sink.params[0].type)


def test_augmented_assignment_and_del_count_as_mutations(tmp_path: Path):
    """`ys += [x]` is `ys.extend`, `del ys[0]` shrinks -- through any alias.

    Scalar `+=` must stay a rebinding: an accumulator loop is the single most
    common pure function there is.
    """
    from ppy_compiler.analysis.inference import refine_with_call_sites
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    path = tmp_path / "aug.py"
    path.write_text(
        textwrap.dedent(
            """
            def extend_it(xs):
                ys = xs
                ys += [4.0]
                return len(xs)


            def shrink_it(xs):
                ys = xs
                del ys[0]
                return len(xs)


            def accumulate(xs):
                total = 0.0
                for x in xs:
                    total += x
                return total


            print(extend_it([1.0]), shrink_it([1.0, 2.0]), accumulate([3.0]))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    project = open_project(path)
    project.config.strict = False
    bundle = analyze_paths(project, [path], backend="python")
    refine_with_call_sites(bundle, bundle.diagnostics)
    functions = bundle.analysis.modules["aug"].functions
    assert functions["aug.extend_it"].mutated_params == {"xs"}
    assert functions["aug.shrink_it"].mutated_params == {"xs"}
    assert functions["aug.accumulate"].mutated_params == set()
    assert functions["aug.accumulate"].verified_pure
    # The widening saw the mutation too: both stay concrete.
    for name in ("extend_it", "shrink_it"):
        assert str(bundle.symbols.functions[f"aug.{name}"].params[0].type) == "list[float]"


def test_augmenting_an_immutable_alias_breaks_it():
    """`y = xs; y += (1,)` on a tuple builds a new tuple: `y` is on its own.

    Only a provably immutable target may break the alias -- for a list the
    same syntax is an in-place extend and the roots must survive.
    """
    import ast

    from ppy_compiler.analysis.aliasing import analyze_aliases

    source = textwrap.dedent(
        """
        def f(xs):
            y = xs
            y += (1,)
            return y
        """
    )
    fn = ast.parse(source).body[0]
    use = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))

    frozen = analyze_aliases(fn, immutable_params=frozenset({"xs"}))
    assert not frozen.param_roots(frozen.roots_at(use.value, "y"))

    thawed = analyze_aliases(fn)
    assert thawed.param_roots(thawed.roots_at(use.value, "y")) == {"xs"}


def test_builtin_descriptor_decorators_are_recognized(tmp_path: Path):
    """`@staticmethod` must canonicalize, not vanish into an unknown name."""
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    path = tmp_path / "desc.ppy"
    path.write_text(
        textwrap.dedent(
            """
            class C:
                @staticmethod
                def f(x: int) -> int:
                    return x

                @classmethod
                def g(cls) -> int:
                    return 1

                @property
                def p(self) -> int:
                    return 2
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    bundle = analyze_paths(open_project(path), [path], backend="python")
    assert bundle.symbols.functions["desc.C.f"].is_static
    assert bundle.symbols.functions["desc.C.g"].is_classmethod
    assert bundle.symbols.functions["desc.C.p"].is_property


def test_lexical_bindings_are_point_sensitive():
    import ast

    from ppy_compiler.analysis.lexical import scan_module

    source = textwrap.dedent(
        """
        def cache(fn):
            return fn


        @cache
        def early(x):
            return x


        from functools import cache


        @cache
        def late(x):
            return x
        """
    )
    tree = ast.parse(source)
    bindings = scan_module(tree, "mod")
    decorators = {
        node.name: bindings.targets_at(node.decorator_list[0])
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    }
    assert decorators["early"] == {"mod.cache"}
    assert decorators["late"] == {"functools.cache"}


def test_global_rebinding_reaches_other_functions(tmp_path: Path):
    from ppy_compiler.analysis.global_writes import build_write_index

    (tmp_path / "store.py").write_text("LIMIT = 5\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("LIMIT = 7\nSAFE = 1\n", encoding="utf-8")
    (tmp_path / "rebind.py").write_text(
        textwrap.dedent(
            """
            import store as s

            import other


            def rebind():
                global s
                s = other


            def write():
                s.LIMIT = 9
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    index = build_write_index(tmp_path)
    assert not index.can_emit_final("store", "LIMIT")
    assert not index.can_emit_final("other", "LIMIT")
    assert index.can_emit_final("other", "SAFE")


def test_class_creation_effects_block_reordering():
    """A base's `__init_subclass__` and a field's `__set_name__` run at
    class creation; a simply *spelled* base or field proves nothing."""
    import ast

    from ppy_compiler.analysis.decorators import definition_time_reorder_safe

    def check(src: str) -> bool:
        return definition_time_reorder_safe(ast.parse(textwrap.dedent(src)).body[0])

    assert check("class C:\n    X = 1")
    assert check("class C(object):\n    pass")
    assert not check("class C(Base):\n    pass")
    assert not check("class C:\n    field = descriptor")
    assert check("class C:\n    field = (1, 2)")
