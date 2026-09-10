"""`ppy.check[T]` validates all the way down; `ppy.assume[T]` validates nothing and says so."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Annotated, Any, Literal, Optional

import pytest

import ppy


@dataclasses.dataclass
class Point:
    x: int
    tags: list[str]


@pytest.mark.parametrize(
    ("target", "value"),
    [
        (int, 3),
        (float, 2.5),
        (float, 3),
        (str, "s"),
        (bool, True),
        (None, None),
        (list[int], [1, 2, 3]),
        (list[int], []),
        (list, ["a", 1]),
        (list[list[int]], [[1], [2, 3]]),
        (dict[str, float], {"a": 1.0, "b": 2}),
        (dict, {1: "x"}),
        (set[str], {"a"}),
        (tuple[int, str], (1, "x")),
        (tuple[int, ...], (1, 2, 3)),
        (tuple[()], ()),
        (int | None, None),
        (int | None, 4),
        (Optional[str], "a"),  # noqa: UP045 -- the spelling must be accepted too
        (Literal["a", "b"], "a"),
        (Annotated[int, "meta"], 5),
        (Any, object()),
        (Point, Point(1, ["a"])),
        (list[Point], [Point(1, []), Point(2, ["b"])]),
    ],
)
def test_check_hands_back_what_fits(target, value):
    assert ppy.check[target](value) is value


@pytest.mark.parametrize(
    ("target", "value", "where"),
    [
        (int, "x", "value: expected int, got str"),
        (str, 1, "value: expected str, got int"),
        (list[int], [1, "x"], "value[1]: expected int, got str"),
        (list[int], "not a list", "value: expected list, got str"),
        (list[list[int]], [[1], ["a"]], "value[1][0]: expected int, got str"),
        (dict[str, float], {1: 1.0}, "value key 1: expected str, got int"),
        (dict[str, float], {"a": "b"}, "value['a']: expected float, got str"),
        (tuple[int, str], (1, 2), "value[1]: expected str, got int"),
        (tuple[int, int], (1,), "value: expected a tuple of 2, got 1"),
        (tuple[int, ...], (1, "x"), "value[1]: expected int, got str"),
        (int | None, "x", "value: expected int | None, got str"),
        (Literal["a", "b"], "c", "value: expected one of ('a', 'b'), got 'c'"),
        (None, 0, "value: expected None, got int"),
        (Point, Point(1, ["a", 2]), "value.tags[1]: expected str, got int"),
        (Point, {"x": 1}, "value: expected Point, got dict"),
    ],
)
def test_check_refuses_what_does_not_fit_all_the_way_down(target, value, where):
    with pytest.raises(
        TypeError,
        match=where.replace("(", r"\(")
        .replace(")", r"\)")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("|", r"\|"),
    ):
        ppy.check[target](value)


def test_a_shallow_check_can_no_longer_inject_a_false_type():
    """The documented contract: what comes out of `check[list[int]]` is a list of ints."""
    with pytest.raises(TypeError):
        ppy.check[list[int]](["not an int"])


@pytest.mark.parametrize(
    "target",
    [Callable[[int], int], "a string", 3, list[int, int], dict[str]],
)
def test_what_cannot_be_validated_soundly_is_refused_outright(target):
    """Rather than checking the outer type and calling it validated."""
    with pytest.raises(TypeError, match="cannot validate"):
        ppy.check[target](None)


def test_assume_validates_nothing_and_reads_as_the_escape_hatch():
    value = ["not an int"]
    assert ppy.assume[list[int]](value) is value
    assert ppy.assume[int]("x") == "x"
    assert repr(ppy.assume) == "ppy.assume" and repr(ppy.assume[int]).startswith("ppy.assume[")
    assert repr(ppy.check) == "ppy.check" and repr(ppy.check[list[int]]).startswith("ppy.check[")
    assert "assume" in ppy.__all__ and "check" in ppy.__all__


def test_the_checker_types_check_and_assume_and_only_check_may_raise(write, analyze):
    path = write(
        "crossing.ppy",
        """
        import ppy


        @ppy.dynamic
        def loose(source: str) -> ppy.Dynamic:
            with ppy.dynamic:
                return eval(source)


        def checked(source: str) -> list[int]:
            return ppy.check[list[int]](loose(source))


        def assumed(source: str) -> list[int]:
            return ppy.assume[list[int]](loose(source))


        def wrong(source: str) -> int:
            return ppy.assume[int](loose(source), 2)
        """,
    )
    bundle = analyze(path)
    codes = [d.code for d in bundle.diagnostics.sorted()]
    assert codes == ["E1305"], [d.message for d in bundle.diagnostics.sorted()]
    functions = bundle.analysis.modules["crossing"].functions
    assert "TypeError" in str(functions["crossing.checked"].effects)
    assert "TypeError" not in str(functions["crossing.assumed"].effects)
    assert str(functions["crossing.assumed"].info.ret) == "list[int]"


def test_a_dynamic_value_crossing_is_told_about_both(write, analyze):
    path = write(
        "cross.ppy",
        """
        import ppy


        @ppy.dynamic
        def loose() -> ppy.Dynamic:
            with ppy.dynamic:
                return eval("1")


        def typed() -> int:
            return loose()
        """,
    )
    diagnostics = analyze(path).diagnostics.sorted()
    assert [d.code for d in diagnostics] == ["E1508"]
    assert "ppy.check[T](value)" in (diagnostics[0].help or "")
    assert "ppy.assume[T](value)" in (diagnostics[0].help or "")
