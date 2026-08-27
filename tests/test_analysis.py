"""Type, effect, refinement, and validity analysis (spec 8, 9, 11, 12)."""

from __future__ import annotations

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
