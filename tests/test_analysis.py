"""Type, effect, refinement, and validity analysis (spec 8, 9, 11, 12)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect
from ppy_compiler.analysis.refinements import IntRange, width_range
from ppy_compiler.diagnostics import Severity


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


def test_an_unvouched_decorator_is_rejected_in_strict_mode(write, codes):
    """The decorator may replace the object; believing the `def` is unsound."""
    path = write(
        "swapped.ppy",
        """
        import ppy


        def deco(fn: ppy.Dynamic) -> ppy.Dynamic:
            def wrapped(s: str) -> str:
                return s.upper()

            return wrapped


        @deco
        def f(x: int) -> int:
            return x + 1


        y: int = f(1)
        print(y)
        """,
    )
    assert "E1204" in codes(path)


def test_an_unvouched_class_decorator_is_rejected(write, codes):
    path = write(
        "classy.ppy",
        """
        import ppy


        def register(cls: ppy.Dynamic) -> ppy.Dynamic:
            return cls


        @register
        class Node:
            pass


        print(Node)
        """,
    )
    assert "E1204" in codes(path)


def test_an_unvouched_method_decorator_is_rejected(write, codes):
    path = write(
        "method.ppy",
        """
        import ppy


        def guard(fn: ppy.Dynamic) -> ppy.Dynamic:
            return fn


        class Node:
            @guard
            def touch(self) -> int:
                return 1


        print(Node().touch())
        """,
    )
    assert "E1204" in codes(path)


def test_vouched_decorators_pass_strict_mode(write, codes):
    path = write(
        "vouched.ppy",
        """
        import functools


        @functools.cache
        def known(x: int) -> int:
            return x * 3


        print(known(2))
        """,
    )
    assert "E1204" not in codes(path)


def test_a_dynamic_boundary_permits_an_unvouched_decorator(write, codes):
    path = write(
        "escaped.ppy",
        """
        import ppy


        def registry(fn: ppy.Dynamic) -> ppy.Dynamic:
            return fn


        @ppy.dynamic
        @registry
        def f(x: int) -> int:
            return x


        print(f(1))
        """,
    )
    assert "E1204" not in codes(path)


def test_dynamic_import_detection_resolves_lexically(write, codes):
    """`imp(name)` is importlib's importer; `x.import_module(...)` is not."""
    path = write(
        "imports.ppy",
        """
        from importlib import import_module as imp


        class X:
            def import_module(self, name: str) -> str:
                return name


        def load(name: str) -> None:
            imp(name)


        x = X()
        print(x.import_module("fine"))
        """,
    )
    found = codes(path)
    assert found.count("E1503") == 1


def test_a_constant_import_module_is_typed_as_its_module(write, analyze, codes):
    path = write(
        "modtype.ppy",
        """
        import importlib

        m = importlib.import_module("math")
        print(m.floor(2.5))
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()

    patched = write(
        "patch.ppy",
        """
        import importlib

        s = importlib.import_module("string")
        s.digits = "nope"
        """,
    )
    assert "E1506" in codes(patched)


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


def test_an_effectful_init_subclass_base_is_rejected(write, codes):
    path = write(
        "hooked.ppy",
        """
        class Base:
            def __init_subclass__(cls) -> None:
                print("side effect")


        class Child(Base):
            pass


        print(Child)
        """,
    )
    assert "E1507" in codes(path)


def test_a_trivial_init_subclass_is_tolerated(write, codes):
    path = write(
        "inert.ppy",
        """
        class Base:
            def __init_subclass__(cls) -> None:
                pass


        class Child(Base):
            pass


        print(Child)
        """,
    )
    assert "E1507" not in codes(path)


def test_an_executable_class_body_is_rejected(write, codes):
    path = write(
        "loud.ppy",
        """
        class Loud:
            print("executed at class definition")


        print(Loud)
        """,
    )
    assert "E1507" in codes(path)


def test_a_project_descriptor_in_a_class_body_is_rejected(write, codes):
    path = write(
        "descriptor.ppy",
        """
        class Descriptor:
            def __set_name__(self, owner: object, name: str) -> None:
                print("side effect")


        class Holder:
            field = Descriptor()


        print(Holder)
        """,
    )
    assert "E1507" in codes(path)


def test_a_declarative_class_body_passes(write, codes):
    path = write(
        "plain.ppy",
        """
        class Plain:
            X = 3

            def method(self) -> int:
                return 4


        print(Plain().method())
        """,
    )
    assert "E1507" not in codes(path)


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


def test_a_dynamic_value_may_not_escape_its_boundary(write, codes):
    """`Dynamic -> int` is a missing runtime check, not a spelling choice."""
    path = write(
        "escape.ppy",
        """
        import ppy


        class C:
            def __init__(self) -> None:
                self.x: int = 0


        def need_int(x: int) -> int:
            return x + 1


        def f(c: C, source: str) -> int:
            with ppy.dynamic:
                value = eval(source)

            c.x = value
            need_int(value)
            return value
        """,
    )
    found = codes(path)
    assert found.count("E1508") == 3


def test_ppy_check_sanctions_the_crossing(write, analyze):
    path = write(
        "sanctioned.ppy",
        """
        import ppy


        def f(source: str) -> int:
            with ppy.dynamic:
                value = eval(source)

            return ppy.check[int](value)


        print(f("1 + 1"))
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()


def test_dynamic_stays_dynamic_through_operations(write, codes):
    """Attribute hops and arithmetic keep the taint until a check clears it."""
    path = write(
        "taint.ppy",
        """
        import ppy


        def f(source: str) -> int:
            with ppy.dynamic:
                value = eval(source)

            shifted = value + 1
            return shifted
        """,
    )
    assert "E1508" in codes(path)


def test_any_remains_the_permissive_legacy_spelling(write, codes):
    """`typing.Any` absorbs a dynamic value; `Dynamic` is the policed one."""
    path = write(
        "legacy.ppy",
        """
        from typing import Any

        import ppy


        def sink(x: Any) -> None:
            print(x)


        def f(source: str) -> None:
            with ppy.dynamic:
                value = eval(source)

            sink(value)
        """,
    )
    assert "E1508" not in codes(path)


def test_the_widened_protocols_answer_their_promised_methods(write, analyze):
    """What the widener writes, the checker must accept back."""
    path = write(
        "protocols.ppy",
        """
        from collections.abc import Mapping, Sequence


        def hits(xs: Sequence[float]) -> int:
            return xs.count(1.0)


        def label(values: Mapping[str, float]) -> float:
            return values.get("a", 0.0) or sum(values.values())


        print(hits([1.0, 2.0]), label({"a": 1.5}))
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()


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


def test_a_try_that_returns_on_every_path_falls_off_no_end(write, analyze):
    """`try: return a / except E: return b` cannot reach what follows."""
    path = write(
        "exhaustive.ppy",
        """
        def read() -> int:
            try:
                return int(input())
            except EOFError:
                return 12
        """,
    )
    bundle = analyze(path)
    assert not [d for d in bundle.diagnostics if d.severity is Severity.ERROR]
    analysis = bundle.analysis.modules["exhaustive"].functions["exhaustive.read"]
    # An inferred `int | None` would mean the checker thought it fell through.
    assert analysis.inferred_ret == T.INT


def test_a_recursive_return_settles_through_a_local(write, analyze):
    """`total += f(...)` must not leave `f` unknown forever."""
    path = write(
        "recursive.ppy",
        """
        def count_down(n):
            if n <= 0:
                return 1
            total = 0
            total += count_down(n - 1)
            return total
        """,
    )
    analysis = analyze(path).analysis.modules["recursive"].functions["recursive.count_down"]
    assert analysis.inferred_ret == T.INT


def test_every_diagnostic_code_is_documented_exactly_once():
    """The registry and the reference are one claim, not two."""
    import re

    from ppy_compiler.diagnostics.codes import CODES

    reference = Path(__file__).parent.parent / "docs" / "diagnostics.md"
    rows = re.findall(r"\| `([EWR][0-9]{4})` \|", reference.read_text(encoding="utf-8"))

    assert sorted(rows) == sorted(set(rows)), "a code is documented twice"
    assert set(rows) == set(CODES), (
        f"undocumented: {sorted(set(CODES) - set(rows))}; "
        f"documented but unregistered: {sorted(set(rows) - set(CODES))}"
    )


def test_every_emitted_code_is_registered():
    """A code a pass can emit has to be one the reference explains."""
    import re

    from ppy_compiler.diagnostics.codes import CODES

    source_root = Path(__file__).parent.parent / "src" / "ppy_compiler"
    emitted: set[str] = set()
    for path in source_root.rglob("*.py"):
        if path.name == "codes.py":
            continue
        emitted.update(re.findall(r'"([EWR][0-9]{4})"', path.read_text(encoding="utf-8")))
        emitted.update(re.findall(r"\[([EWR][0-9]{4})\]", path.read_text(encoding="utf-8")))
    assert emitted <= set(CODES), f"unregistered: {sorted(emitted - set(CODES))}"


def test_a_buffer_is_checked_for_its_element_and_its_size(write, analyze):
    """`ppy.buffer` allocates memory, so both operands are settled statically.

    A standalone binary has no CPython to raise for it, and a byte count is
    the one argument where a wrong answer is a segmentation fault rather than
    an exception.
    """
    cases = [
        ("element.ppy", "ppy.buffer[str](4)", "E1306", "not `str`"),
        ("negative.ppy", "ppy.buffer[int](-1)", "E1401", "fewer than no elements"),
        ("textual.ppy", "ppy.buffer[int]('4')", "E1301", "how many elements"),
    ]
    for name, expression, code, wording in cases:
        path = write(
            name,
            f"""
            import ppy

            def main() -> int:
                room = {expression}
                return len(room)
            """,
        )
        diagnostics = analyze(path).diagnostics.sorted()
        reported = [d for d in diagnostics if d.code == code]
        assert reported, f"{expression}: expected {code}, got {[d.code for d in diagnostics]}"
        assert wording in reported[0].message


def test_a_byte_buffer_is_allocated_by_its_signedness(write, analyze):
    """`ppy.i8` and `ppy.u8` are both elements, and both are one byte."""
    path = write(
        "bytes_alloc.ppy",
        """
        import ppy
        from ppy import Buffer

        def main() -> int:
            signed: Buffer[ppy.i8] = ppy.buffer[ppy.i8](2)
            unsigned: Buffer[ppy.u8] = ppy.buffer[ppy.u8](2)
            signed[0] = -128
            unsigned[0] = 255
            return signed[0] + unsigned[0]
        """,
    )
    assert [d.code for d in analyze(path).diagnostics.sorted()] == []


def test_purity_still_counts_a_write_the_function_handed_on(write, analyze):
    """Two questions about one write, and only one of them is purity's.

    A standalone build may lower a function that fills memory it allocated
    and passes it along -- nothing there needs CPython. `@ppy.pure` may not
    permit it: something else saw the object before the function returned,
    which is exactly what spec 11.2 rules out.
    """
    path = write(
        "shared_write.ppy",
        """
        import ppy
        from ppy import Buffer

        def observe(b: Buffer[int]) -> int:
            return b[0]

        @ppy.pure
        def shares_what_it_wrote(n: int) -> int:
            room: Buffer[int] = ppy.buffer[int](n)
            room[0] = 7
            return observe(room)

        @ppy.pure
        def keeps_it_to_itself(n: int) -> int:
            room: Buffer[int] = ppy.buffer[int](n)
            room[0] = 7
            return room[0]
        """,
    )
    bundle = analyze(path)
    reported = [d for d in bundle.diagnostics.sorted() if d.code == "E1601"]
    assert len(reported) == 1, [d.message for d in reported]
    assert "shares_what_it_wrote" in reported[0].message

    shared = bundle.analysis.function("shared_write.shares_what_it_wrote")
    kept = bundle.analysis.function("shared_write.keeps_it_to_itself")
    assert not shared.writes_only_locals, "the callee saw it"
    assert shared.writes_only_allocations, "and yet it is memory this function made"
    assert kept.writes_only_locals and kept.writes_only_allocations


def test_an_unresolved_type_is_reported_once_not_everywhere_it_flowed(write, analyze):
    """Ten call sites of a function with an unknown argument were ten errors.

    They all said the same thing -- `<unknown>` -- and none of them said why.
    The origin is the finding; the rest is a count.
    """
    path = write(
        "cascade.ppy",
        """
        import somewhere_unmodeled

        def charge(amount: float) -> float:
            return amount * 1.1

        def bill() -> float:
            total: float = 0.0
            for _ in range(3):
                raw = somewhere_unmodeled.price()
                total += charge(raw)
                total += charge(raw + 1)
            return total
        """,
    )
    diagnostics = analyze(path, strict=False).diagnostics.sorted()
    messages = [d.message for d in diagnostics]
    assert not any("<unknown>" in m for m in messages), messages
    summaries = [d for d in diagnostics if d.code == "W2006"]
    assert len(summaries) == 1, [d.code for d in diagnostics]
    assert "somewhere_unmodeled.price" in summaries[0].message
    assert "not shown" in summaries[0].message


def test_a_field_is_every_type_its_class_assigns_to_it(write, analyze):
    """`self.buffer = None` in `__init__` and a real value later is `T | None`.

    Fixing the field as `None` from the first line alone made every later
    assignment an error -- the shape of most stateful classes.
    """
    path = write(
        "fields.ppy",
        """
        class Reader:
            def __init__(self) -> None:
                self.count = None
                self.name = "pending"

            def start(self, n: int) -> None:
                self.count = n

            def total(self) -> int:
                return 0 if self.count is None else self.count
        """,
    )
    from ppy_compiler.analysis.inference import refine_with_call_sites

    bundle = analyze(path, strict=False)
    refine_with_call_sites(bundle)
    errors = [d for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]
    cls = bundle.symbols.classes["fields.Reader"]
    assert str(cls.fields["count"]) in {
        "int | None",
        "None | int",
        "int | NoneType",
        "NoneType | int",
    }
    assert str(cls.fields["name"]) == "str"


def test_a_declared_field_is_not_widened_by_what_a_method_assigns(write, analyze):
    path = write(
        "declared.ppy",
        """
        class Box:
            size: int

            def __init__(self) -> None:
                self.size = 0

            def reset(self) -> None:
                self.size = None
        """,
    )
    from ppy_compiler.analysis.inference import refine_with_call_sites

    bundle = analyze(path, strict=False)
    refine_with_call_sites(bundle)
    cls = bundle.symbols.classes["declared.Box"]
    assert str(cls.fields["size"]) == "int", "the author said int"
    assert "size" in cls.declared_fields


def test_every_builtin_exception_is_a_known_name(write, analyze):
    """`raise RuntimeError(...)` was "not defined at this point".

    Thirteen exceptions were listed by hand and the other fifty-six were
    not, so most programs that raised or caught one got an error for it.
    """
    import builtins

    for name in dir(builtins):
        value = getattr(builtins, name)
        if isinstance(value, type) and issubclass(value, BaseException):
            assert name in T.BUILTIN_MRO, name
            assert T.BUILTIN_MRO[name][-1] == "object"
    assert T.BUILTIN_MRO["FileNotFoundError"][1] == "OSError", "the real hierarchy, not a flat list"

    path = write(
        "raises.ppy",
        """
        def check(x: int) -> int:
            if x < 0:
                raise AssertionError("negative")
            try:
                return 10 // x
            except ZeroDivisionError:
                raise RuntimeError("zero") from None
            except (OSError, KeyboardInterrupt):
                raise
        """,
    )
    codes = [d.code for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert "E1101" not in codes, codes


def test_the_math_module_is_modeled(write, analyze):
    """`math.tanh` in a reward function was an unknown signature.

    Every numeric kernel imports `math`, and none of it was in the stdlib
    table, so everything computed from a `math` call followed an unknown.
    """
    import math

    from ppy_compiler.analysis import stdlib

    public = {n for n in dir(math) if not n.startswith("_") and callable(getattr(math, n))}
    modeled = {k.removeprefix("math.") for k in stdlib._FUNCTIONS if k.startswith("math.")}
    assert public <= modeled, sorted(public - modeled)

    path = write(
        "kernel.ppy",
        """
        import math

        def squash(x: float) -> float:
            return math.tanh(x) * math.pi

        def whole(x: float) -> int:
            return math.floor(x) + math.isqrt(9)

        def finite(x: float) -> bool:
            return math.isfinite(x) and not math.isnan(x)
        """,
    )
    bundle = analyze(path)
    assert [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"] == []
    assert str(bundle.analysis.function("kernel.squash").inferred_ret) == "float"
    assert str(bundle.analysis.function("kernel.whole").inferred_ret) == "int"
    assert str(bundle.analysis.function("kernel.finite").inferred_ret) == "bool"


def test_a_reanalysis_confirms_without_reseeding(write, analyze):
    """Inference analyzes a project once per round; only the first needs a seed.

    The silent seed pass gives every function a first summary. After that,
    every function already carries one, and a confirming pass finds what
    moved -- so the seed is skipped, and the answers must not change.
    """
    from ppy_compiler.analysis.checker import analyze as run_checker
    from ppy_compiler.diagnostics import DiagnosticBag

    path = write(
        "rounds.ppy",
        """
        def scale(x: int) -> int:
            return x * 2

        def twice(x: int) -> int:
            return scale(scale(x))

        def report(xs: list[int]) -> int:
            total: int = 0
            for x in xs:
                total += twice(x)
            return total
        """,
    )
    bundle = analyze(path)
    assert bundle.symbols.seeded, "the first analysis seeded the summaries"
    first = {
        name: (str(f.effects), str(f.inferred_ret), sorted(f.calls))
        for name, f in bundle.analysis.modules["rounds"].functions.items()
    }
    first_types = dict(bundle.analysis.modules["rounds"].node_types)

    again = run_checker(bundle.symbols, DiagnosticBag(), strict=True)
    second = {
        name: (str(f.effects), str(f.inferred_ret), sorted(f.calls))
        for name, f in again.modules["rounds"].functions.items()
    }
    assert second == first
    assert {k: str(v) for k, v in again.modules["rounds"].node_types.items()} == {
        k: str(v) for k, v in first_types.items()
    }


def test_a_module_nothing_moved_in_is_not_checked_again(write, analyze):
    """An inference round hands the previous analysis over; unchanged modules keep it.

    A module's checking reads its own signatures and fields and those of
    what it imports. When none of those moved since it was last checked,
    checking it again could only say the same thing -- so it is not checked,
    and the earlier `ModuleAnalysis` object is the answer. When a callee's
    signature does move, the importer is checked again and agrees with a
    fresh analysis.
    """
    from ppy_compiler.analysis.checker import analyze as run_checker
    from ppy_compiler.diagnostics import DiagnosticBag

    write(
        "lib.ppy",
        """
        def scale(x):
            return x * 2
        """,
    )
    path = write(
        "app.ppy",
        """
        from lib import scale

        def run(n: int) -> int:
            return scale(n)
        """,
    )
    bundle = analyze(path, strict=False)
    first = bundle.analysis

    again = run_checker(bundle.symbols, DiagnosticBag(), strict=False, previous=first)
    assert again.modules["lib"] is first.modules["lib"], "nothing moved: kept"
    assert again.modules["app"] is first.modules["app"]

    # Inference learns `scale`'s parameter: the module that calls it must be
    # looked at again, and the one that defines it too.
    scale = bundle.symbols.functions["lib.scale"]
    scale.params[0].type = T.INT
    scale.params[0].inferred = True
    moved = run_checker(bundle.symbols, DiagnosticBag(), strict=False, previous=again)
    assert moved.modules["lib"] is not again.modules["lib"]
    assert moved.modules["app"] is not again.modules["app"]
    assert str(moved.function("lib.scale").inferred_ret) == "int"

    fresh = run_checker(bundle.symbols, DiagnosticBag(), strict=False)
    for name in ("lib", "app"):
        assert {k: str(v) for k, v in moved.modules[name].node_types.items()} == {
            k: str(v) for k, v in fresh.modules[name].node_types.items()
        }


def test_set_algebra_is_typed(write, analyze):
    """`a | b` on sets was "not defined": the numeric rules were the only rules."""
    path = write(
        "sets.ppy",
        """
        def union(a: set[str], b: set[str]) -> set[str]:
            return a | b

        def keep(a: frozenset[str], b: set[str]) -> frozenset[str]:
            return (a & b) - frozenset({"x"})

        def either(a: set[int], b: frozenset[int]) -> set[int]:
            return a ^ b
        """,
    )
    bundle = analyze(path)
    assert [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"] == []
    assert str(bundle.analysis.function("sets.union").inferred_ret) == "set[str]"
    assert str(bundle.analysis.function("sets.keep").inferred_ret) == "frozenset[str]"
    assert str(bundle.analysis.function("sets.either").inferred_ret) == "set[int]"


def test_frozenset_is_not_a_set(write, analyze):
    """`frozenset(...)` was typed as `set[...]`, so `-> frozenset[str]` was a wrong return."""
    path = write(
        "frozen.ppy",
        """
        def names(xs: list[str]) -> frozenset[str]:
            return frozenset(xs)

        def empty() -> frozenset[str]:
            return frozenset()
        """,
    )
    bundle = analyze(path)
    assert [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"] == []
    assert str(bundle.analysis.function("frozen.names").inferred_ret) == "frozenset[str]"


def test_a_class_can_overload_an_operator(write, analyze):
    """`EffectSet | EffectSet` is whatever `EffectSet.__or__` returns."""
    path = write(
        "ops.ppy",
        """
        class Mask:
            def __init__(self, bits: int) -> None:
                self.bits: int = bits

            def __or__(self, other: "Mask") -> "Mask":
                return Mask(self.bits | other.bits)

            def __rsub__(self, other: int) -> int:
                return other - self.bits

        def both(a: Mask, b: Mask) -> Mask:
            return a | b

        def left(n: int, m: Mask) -> int:
            return n - m
        """,
    )
    bundle = analyze(path)
    assert [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"] == []
    assert str(bundle.analysis.function("ops.both").inferred_ret) == "ops.Mask"
    assert str(bundle.analysis.function("ops.left").inferred_ret) == "int"
    assert "ops.Mask.__or__" in bundle.analysis.function("ops.both").calls


def test_common_stdlib_classes_are_types_an_annotation_may_name(write, analyze):
    """`p: Path` was "not a type the project can analyze" in every file that took a path.

    The analyzer need not know what a `Path` does to accept the name; a
    call on one is still unknown and still says so.
    """
    from ppy_compiler.analysis import stdlib

    assert "ast.FunctionDef" in stdlib.EXTERNAL_TYPES and "pathlib.Path" in stdlib.EXTERNAL_TYPES

    path = write(
        "paths.ppy",
        """
        import ast
        from pathlib import Path

        def where(p: Path, node: ast.AST) -> ast.expr:
            return node.body[0].value

        def name(p: Path) -> str:
            return str(p)
        """,
    )
    diagnostics = analyze(path, strict=False).diagnostics.sorted()
    assert not [d for d in diagnostics if d.code == "E1101"], [d.message for d in diagnostics]
    assert (
        str(analyze(path, strict=False).symbols.functions["paths.where"].params[0].type)
        == "pathlib.Path"
    )


def test_the_builtin_singletons_are_names(write, analyze):
    """`return NotImplemented` is how an operator method declines."""
    path = write(
        "singletons.ppy",
        """
        class Box:
            def __eq__(self, other: object) -> bool:
                if not isinstance(other, Box):
                    return NotImplemented
                return True

        def gap() -> object:
            return Ellipsis
        """,
    )
    codes = [d.code for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert "E1101" not in codes, codes


def test_super_past_a_base_the_project_does_not_define(write, analyze):
    """`super().__init__(...)` in an `Exception` subclass was an undefined name."""
    path = write(
        "errors.ppy",
        """
        class Failure(Exception):
            def __init__(self, count: int) -> None:
                super().__init__(f"{count} failure(s)")
                self.count: int = count
        """,
    )
    codes = [d.code for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert "E1101" not in codes, codes


def test_a_container_display_takes_its_declared_type(write, analyze):
    """`stack: list[ast.AST] = [stmt]` holds AST nodes; it is not a narrower list."""
    path = write(
        "displays.ppy",
        """
        class Node:
            pass

        class Leaf(Node):
            pass

        def walk(leaf: Leaf) -> int:
            stack: list[Node] = [leaf]
            seen: set[Node] = {leaf}
            by_name: dict[str, Node] = {"leaf": leaf}
            empty: list[Node] = []
            return len(stack) + len(seen) + len(by_name) + len(empty)

        def wrong() -> int:
            names: list[str] = [1, 2]
            return len(names)
        """,
    )
    diagnostics = analyze(path).diagnostics.sorted()
    errors = [d for d in diagnostics if d.severity.name == "ERROR"]
    assert len(errors) == 1, [d.message for d in errors]
    assert "names" in errors[0].message or "list[int]" in errors[0].message
    assert errors[0].span is not None and errors[0].span.line == 15, (
        "the wrong display, not the right ones"
    )


def test_dictionary_views_take_the_set_algebra(write, analyze):
    """`left.keys() | right.keys()` was "not defined for Iterable and Iterable"."""
    path = write(
        "views.ppy",
        """
        def merged(left: dict[str, int], right: dict[str, int]) -> int:
            names = left.keys() | right.keys()
            shared = left.keys() & right.keys()
            return len(names) + len(shared)

        def pairs(d: dict[str, int]) -> int:
            total = 0
            for key, value in d.items():
                total += len(key) + value
            return total
        """,
    )
    bundle = analyze(path)
    assert [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"] == []
    assert str(bundle.analysis.function("views.merged").locals["names"]) == "frozenset[str]"


def test_a_name_a_package_reexports_resolves_to_where_it_is_defined(write, analyze):
    """`from pkg import Thing` where `pkg/__init__.py` did `from .model import Thing`."""
    write("pkg/__init__.py", "from .model import Thing\n")
    write(
        "pkg/model.py",
        """
        class Thing:
            def __init__(self, size: int) -> None:
                self.size: int = size
        """,
    )
    path = write(
        "app.py",
        """
        from pkg import Thing

        def make(n: int) -> Thing:
            return Thing(n)

        def grow(t: Thing) -> int:
            return t.size + 1
        """,
    )
    bundle = analyze(path, strict=True)
    errors = [d for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]
    assert str(bundle.symbols.functions["app.make"].ret) == "pkg.model.Thing"
    assert str(bundle.analysis.function("app.grow").inferred_ret) == "int"


def test_a_bound_method_passed_along_has_no_self(write, analyze):
    """`notify=reporter.note` is a callable of one argument, not two."""
    path = write(
        "callbacks.ppy",
        """
        from collections.abc import Callable

        class Reporter:
            def note(self, message: str) -> None:
                print(message)

        def build(notify: Callable[[str], None]) -> None:
            notify("compiling")

        def run(reporter: Reporter) -> None:
            build(reporter.note)
            reporter.note("done")
            saved = reporter.note
            saved("later")
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_a_check_on_an_attribute_narrows_that_attribute(write, analyze):
    """`if self.end is not None: self.end - 1` was `-` on `int | None`."""
    path = write(
        "spans.ppy",
        """
        class Span:
            def __init__(self, start: int) -> None:
                self.start: int = start
                self.end: int | None = None
                self.label: str | None = None

            def width(self) -> int:
                if self.end is not None:
                    return self.end - self.start
                return 1

            def wide(self) -> int:
                if self.end is None:
                    return 0
                return self.end - self.start

            def named(self) -> int:
                if self.label:
                    return len(self.label)
                return 0

            def both(self) -> int:
                if self.end is not None and self.label is not None:
                    return self.end + len(self.label)
                return 0

            def forgotten(self) -> int:
                if self.end is not None:
                    self.end = None
                    return self.end - 1
                return 0
        """,
    )
    diagnostics = analyze(path).diagnostics.sorted()
    errors = [d for d in diagnostics if d.severity.name == "ERROR"]
    assert len(errors) == 1, [d.message for d in errors]
    assert "forgotten" in path.read_text().splitlines()[errors[0].span.line - 4], errors[0].message


def test_isinstance_on_an_attribute_narrows_it(write, analyze):
    path = write(
        "shapes.ppy",
        """
        class Node:
            pass

        class Call(Node):
            def __init__(self) -> None:
                self.name: str = "f"

        class Holder:
            def __init__(self, node: Node) -> None:
                self.node: Node = node

            def label(self) -> str:
                if isinstance(self.node, Call):
                    return self.node.name
                return ""
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_a_walrus_inside_an_and_binds_for_the_body(write, analyze):
    """`if (d := find()) is not None and d.ok:` used `d` in the body -- undefined."""
    path = write(
        "walrus.ppy",
        """
        class Directive:
            def __init__(self) -> None:
                self.require: bool = True

        def find(name: str) -> Directive | None:
            return Directive() if name else None

        def check(name: str) -> bool:
            if (directive := find(name)) is not None and directive.require:
                return directive.require
            return False
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_dir_is_a_builtin(write, analyze):
    path = write("names.ppy", "def names(x: object) -> list[str]:\n    return dir(x)\n")
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_a_name_imported_from_a_library_is_what_the_library_says(write, analyze):
    """`from pathlib import Path` then `Path(p)` was an unknown signature; `pathlib.Path(p)` was not."""
    path = write(
        "paths.ppy",
        """
        from pathlib import Path
        from math import sqrt
        import pathlib

        def size(root: str) -> int:
            here = Path(root)
            there = pathlib.Path(root) / "sub"
            total = 0
            for entry in here.glob("*.txt"):
                if entry.is_file():
                    total += len(entry.read_text())
            return total + len(there.name) + int(sqrt(4.0))
        """,
    )
    bundle = analyze(path)
    errors = [d for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]
    assert str(bundle.analysis.function("paths.size").locals["here"]) == "pathlib.Path"
    assert str(bundle.analysis.function("paths.size").locals["there"]) == "pathlib.Path"
    assert "IO" in str(bundle.analysis.function("paths.size").effects)


def test_the_ast_functions_are_modeled(write, analyze):
    path = write(
        "walker.ppy",
        """
        import ast

        def names(source: str) -> list[str]:
            tree = ast.parse(source)
            found: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    found.append(ast.unparse(node))
            return found
        """,
    )
    bundle = analyze(path)
    errors = [d for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]
    assert str(bundle.analysis.function("walker.names").locals["tree"]) == "ast.Module"


def test_type_is_an_annotation(write, analyze):
    path = write(
        "kinds.ppy",
        """
        def name_of(kind: type) -> str:
            return kind.__name__
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_a_starred_argument_is_as_many_arguments_as_it_holds(write, analyze):
    path = write(
        "spread.ppy",
        """
        def describe(a: int, b: int, c: int) -> int:
            return a + b + c


        def total(pair: tuple[int, int]) -> int:
            return describe(1, *pair)
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_a_starred_display_element_holds_the_elements(write, analyze):
    path = write(
        "spread_display.ppy",
        """
        def widen(first: int, rest: list[int]) -> list[int]:
            return [first, *rest]


        def pair_up(first: str, rest: tuple[str, ...]) -> tuple[str, ...]:
            return (first, *rest)
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_type_of_a_class_is_the_class_object(write, analyze):
    path = write(
        "factory.ppy",
        """
        from collections.abc import Callable


        class Base:
            def __init__(self) -> None:
                self.n = 1


        class Derived(Base):
            pass


        def make(kind: type[Base]) -> Base:
            return kind()


        def derived() -> Base:
            return make(Derived)


        def caster() -> Callable[[object], object]:
            return int
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_any_is_anything_at_any_depth(write, analyze):
    path = write(
        "nested.ppy",
        """
        from typing import Any


        def payload() -> dict[str, Any]:
            return {"a": {"b": 1}}


        def rows() -> list[dict[str, Any]]:
            return [{"n": 1, "name": "x"}]


        def wrong() -> dict[str, int]:
            return {"a": {"b": 1}}
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert [d.span.line for d in errors if d.span] == [13], [d.message for d in errors]


def test_the_partitions_are_modeled(write, analyze):
    path = write(
        "parts.ppy",
        """
        def last(qualname: str) -> str:
            return qualname.rpartition(".")[2]


        def head(qualname: str) -> str:
            return qualname.partition(".")[0]
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_an_external_class_names_its_bases_the_public_way():
    pytest.importorskip("libcst")
    from ppy_compiler.analysis import stdlib

    assert "libcst.BaseExpression" in stdlib.EXTERNAL_MRO["libcst.Call"]
    assert "libcst.CSTNode" in stdlib.EXTERNAL_MRO["libcst.Call"]


def test_sqlite_types_are_annotations(write, analyze):
    path = write(
        "store.ppy",
        """
        import sqlite3


        def open_one(connection: sqlite3.Connection) -> None:
            pass
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_a_class_that_satisfies_a_protocol_is_one(write, analyze):
    path = write(
        "binder.ppy",
        """
        from typing import Protocol


        class Binder(Protocol):
            def bind(self, name: str) -> int: ...


        class Prebuilt:
            def __init__(self) -> None:
                self.count = 0

            def bind(self, name: str) -> int:
                return len(name)


        class Unrelated:
            def __init__(self) -> None:
                self.count = 0


        def run(binder: Binder | None) -> int:
            return 0 if binder is None else binder.bind("x")


        def go() -> int:
            return run(Prebuilt())


        def wrong() -> int:
            return run(Unrelated())
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert [d.span.line for d in errors if d.span] == [30], [d.message for d in errors]


def test_a_class_with_iter_is_an_iterable(write, analyze):
    path = write(
        "bag.ppy",
        """
        from collections.abc import Iterable, Iterator


        class Bag:
            def __init__(self) -> None:
                self.items: list[int] = []

            def __iter__(self) -> Iterator[int]:
                return iter(self.items)


        def consume(items: Iterable[int]) -> int:
            total = 0
            for item in items:
                total += item
            return total


        def drain(bag: Bag) -> int:
            return consume(bag)
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_an_unresolved_class_object_is_some_class():
    some_class = T.instance("type")
    assert T.is_assignable(some_class, T.ClassObject("int", T.INT))
    assert T.is_assignable(some_class, T.union(T.ClassObject("int", T.INT), T.NONE))
    assert not T.is_assignable(some_class, T.INT)
    assert T.is_assignable(T.ClassObject("int", T.INT), T.OBJECT)
    assert T.is_assignable(T.Callable_((), T.INT, "f"), T.OBJECT)


def test_a_union_of_classes_is_a_type_and_everything_is_an_object(write, analyze):
    path = write(
        "objects.ppy",
        """
        import ast


        def is_container(value: object) -> bool:
            return isinstance(value, list | dict)


        def hold(value: object) -> object:
            return value


        def a_function() -> object:
            return hold(len)


        def a_module() -> object:
            return hold(ast)
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_a_library_exception_is_a_base_exception(write, analyze):
    path = write(
        "damage.ppy",
        """
        import sqlite3


        def damaged(error: BaseException) -> bool:
            return True


        def probe(error: sqlite3.DatabaseError) -> bool:
            return damaged(error)
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_any_inside_a_tuple_argument(write, analyze):
    path = write(
        "pairs.ppy",
        """
        from typing import Any


        def rows() -> dict[str, tuple[str, Any]]:
            return {"a": ("x", 1)}
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_or_drops_none_from_every_operand_but_the_last(write, analyze):
    path = write(
        "fallback.ppy",
        """
        def modules(program: dict[str, int] | None) -> int:
            return (program or {}).get("modules", 0)


        def either(first: int | None, second: int | None) -> int | None:
            return first or second


        def wrong(first: int | None, second: int | None) -> int:
            return first or second
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert [d.span.line for d in errors if d.span] == [10], [d.message for d in errors]


def test_a_constant_subscript_is_a_place_a_check_can_be_about(write, analyze):
    path = write(
        "places.ppy",
        """
        class Facts:
            def __init__(self, low: int | None) -> None:
                self.low: int | None = low


        class Arg:
            def __init__(self, facts: Facts | None) -> None:
                self.facts: Facts | None = facts


        def lowest(args: list[Arg]) -> int:
            if args[0].facts is not None:
                r = args[0].facts
                return r.low or 0
            return 0


        def reset(args: list[Arg]) -> int:
            if args[0].facts is not None:
                args[0] = Arg(None)
                return args[0].facts.low or 0
            return 0
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert [d.span.line for d in errors if d.span] == [21], [d.message for d in errors]


def test_a_union_of_tuples_unpacks_position_by_position(write, analyze):
    path = write(
        "counts.ppy",
        """
        def count_up(seen: dict[str, tuple[int, str]], key: str) -> int:
            count, label = seen.get(key, (0, 1.5))
            return count + 1
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_being_one_of_the_constants_is_being_of_their_type(write, analyze):
    path = write(
        "members.ppy",
        """
        def short(canonical: str | None) -> str:
            if canonical not in {"a.b", "c.d"}:
                return ""
            return canonical.rpartition(".")[2]


        def loud(name: str | None) -> str:
            if name in ("x", "y"):
                return name.upper()
            return ""


        def wrong(name: str | None) -> str:
            if name not in ("x", "y"):
                return name.upper()
            return ""
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert [d.span.line for d in errors if d.span] == [15], [d.message for d in errors]


def test_a_field_init_assigns_is_known_to_a_single_pass(write, analyze):
    path = write(
        "fields.ppy",
        """
        class Facts:
            def __init__(self, low: int | None) -> None:
                self.low = low


        class Arg:
            def __init__(self, facts: Facts) -> None:
                self.facts = facts
                self.low = facts.low


        def lowest(arg: Arg) -> int:
            return arg.low or arg.facts.low or 0


        def wrong(arg: Arg) -> int:
            return arg.low
        """,
    )
    bundle = analyze(path)
    errors = [d for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]
    assert [d.span.line for d in errors if d.span] == [17], [d.message for d in errors]
    assert str(bundle.symbols.classes["fields.Arg"].fields["low"]) == "int | NoneType"


def test_a_module_and_a_function_hand_out_their_nodes_walked_once(write, analyze):
    import ast

    path = write(
        "walked.ppy",
        """
        def total(values: list[int]) -> int:
            result = 0
            for value in values:
                result += value
            return result
        """,
    )
    bundle = analyze(path)
    module = bundle.symbols.modules["walked"].module
    assert module.nodes == tuple(ast.walk(module.tree))
    assert module.nodes is module.nodes
    info = bundle.symbols.functions["walked.total"]
    assert info.nodes == tuple(ast.walk(info.node))
    assert info.nodes is info.nodes
    assert info.nodes[0] is info.node


def test_a_revisited_node_keeps_each_state_once():
    from ppy_compiler.analysis.aliasing import _Reached

    first, second, third = {"a": frozenset({"x"})}, {"b": frozenset()}, {"c": frozenset()}
    reached = _Reached(first, second)
    reached.add(first)
    reached.add(third)
    reached.add(third)
    assert list(reached) == [first, second, third]
    assert isinstance(reached, list)


def test_a_subclass_of_a_builtin_has_the_builtin_methods(write, analyze):
    path = write(
        "reached.ppy",
        """
        class Reached(list):
            def __init__(self, first: int) -> None:
                super().__init__((first,))
                self.ids = {first}

            def add(self, value: int) -> None:
                if value not in self.ids:
                    self.ids.add(value)
                    self.append(value)


        def size(reached: Reached) -> int:
            reached.add(3)
            return len(reached) + reached.count(3)
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert errors == [], [d.message for d in errors]


def test_normalizing_a_module_in_hand_is_normalizing_its_source():
    import libcst as cst

    from ppy_compiler.driver.formatting import normalize_module, normalize_source

    source = "import sys\nimport os\ndef f(a,b):\n    return a+b\nclass C:\n    pass\n"
    assert normalize_module(cst.parse_module(source)) == normalize_source(source)


def test_an_annotation_spelling_is_parsed_once():
    from ppy_compiler.driver.rewrite import _expression

    assert _expression("list[str]") is _expression("list[str]")
    assert _expression("list[str]").deep_equals(_expression("list[str]"))


def test_signatures_are_wrapped_only_past_the_limit():
    from ppy_compiler.driver.formatting import normalize_source

    short = "def f(a: int, b: int) -> int:\n    return a + b\n"
    assert normalize_source(short) == short
    names = ", ".join(f"parameter_{i}: int" for i in range(8))
    long = f"def g({names}) -> int:\n    return 0\n"
    wrapped = normalize_source(long)
    assert wrapped != long
    assert "def g(\n    parameter_0: int,\n" in wrapped
    assert all(len(line) <= 100 for line in wrapped.splitlines())
