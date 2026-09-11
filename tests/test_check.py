"""`ppy.check[T]` validates all the way down; `ppy.assume[T]` validates nothing and says so."""

from __future__ import annotations

import dataclasses
import re
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


@pytest.mark.parametrize(
    ("target", "value"),
    [
        (ppy.i8, 127),
        (ppy.i8, -128),
        (ppy.u8, 0),
        (ppy.u8, 255),
        (ppy.i32, -(1 << 31)),
        (ppy.u64, (1 << 64) - 1),
        (ppy.f32, 1.5),
        (ppy.f32, 3),
        (ppy.f64, 1e300),
        (ppy.Array[int, 3], (1, 2, 3)),
        (ppy.Array[float, 0], ()),
        (ppy.Vector[int], [1, 2]),
        (ppy.Vector[str], []),
        (Annotated[int, ppy.Range(0, 10)], 10),
        (Annotated[float, ppy.Range(0.0, 1.0)], 0.5),
        (Annotated[list[int], ppy.Length(2)], [1, 2]),
        (Annotated[str, ppy.Length(3)], "abc"),
        (ppy.i8 | None, None),
        (ppy.i8 | None, -5),
        (list[ppy.u8], [0, 255]),
        (tuple[ppy.i8, ppy.u8], (-1, 200)),
    ],
)
def test_a_refinement_that_holds_passes(target, value):
    assert ppy.check[target](value) is value or ppy.check[target](value) == value


@pytest.mark.parametrize(
    ("target", "value", "where"),
    [
        (ppy.i8, 128, "value: 128 does not fit i8 (-128..127)"),
        (ppy.i8, -129, "value: -129 does not fit i8 (-128..127)"),
        (ppy.u8, -1, "value: -1 does not fit u8 (0..255)"),
        (ppy.u8, 256, "value: 256 does not fit u8 (0..255)"),
        (ppy.i8, 300, "value: 300 does not fit i8"),
        (ppy.i8, 1.5, "value: expected int, got float"),
        (ppy.u64, 1 << 64, "does not fit u64"),
        (ppy.f32, 1e39, "value: 1e+39 is wider than f32 holds"),
        (ppy.f16, 70000.0, "is wider than f16 holds"),
        (ppy.f32, "x", "value: expected float, got str"),
        (ppy.Array[int, 3], (1, 2), "value: expected a length of 3, got 2"),
        (ppy.Array[int, 3], (1, 2, 3, 4), "value: expected a length of 3, got 4"),
        (ppy.Array[int, 3], (1, "x", 3), "value[1]: expected int, got str"),
        (ppy.Array[int, 3], [1, 2, 3], "value: expected tuple, got list"),
        (ppy.Vector[int], [1, "x"], "value[1]: expected int, got str"),
        (ppy.Vector[int], (1, 2), "value: expected list, got tuple"),
        (Annotated[int, ppy.Range(0, 10)], 11, "value: 11 is outside ppy.Range(low=0, high=10)"),
        (Annotated[int, ppy.Range(0, 10)], -1, "is outside"),
        (Annotated[int, ppy.Range(0, 10)], "5", "value: expected int, got str"),
        (Annotated[list[int], ppy.Length(2)], [1], "value: expected a length of 2, got 1"),
        (list[ppy.u8], [0, 256], "value[1]: 256 does not fit u8"),
        (tuple[ppy.i8, ppy.u8], (-1, -1), "value[1]: -1 does not fit u8"),
        (ppy.i8 | None, 300, "expected"),
    ],
)
def test_a_refinement_that_fails_is_refused(target, value, where):
    with pytest.raises(TypeError, match=re.escape(where)):
        ppy.check[target](value)


def test_a_buffer_is_the_buffer_protocol_with_the_elements_format():
    import array

    ints = array.array("q", [1, 2, 3])
    assert ppy.check[ppy.Buffer[int]](ints) is ints
    assert ppy.check[ppy.Buffer[int]](memoryview(ints)) is not None
    doubles = array.array("d", [1.0])
    assert ppy.check[ppy.Buffer[float]](doubles) is doubles
    assert ppy.check[ppy.Buffer[ppy.f64]](doubles) is doubles
    assert ppy.check[ppy.Buffer[ppy.i8]](array.array("b", [1])) is not None
    assert ppy.check[ppy.Buffer[ppy.u8]](bytearray(b"ab")) is not None
    assert ppy.check[ppy.Buffer[ppy.i32]](array.array("i", [1])) is not None
    with pytest.raises(TypeError, match=r"a Buffer\[int\] holds 'q' elements, this holds 'd'"):
        ppy.check[ppy.Buffer[int]](doubles)
    with pytest.raises(TypeError, match="expected a buffer of int, got list"):
        ppy.check[ppy.Buffer[int]]([1, 2, 3])
    with pytest.raises(TypeError, match="holds 'q' elements, this holds 'B'"):
        ppy.check[ppy.Buffer[int]](b"bytes are not 64-bit integers")
    with pytest.raises(TypeError, match="one contiguous dimension"):
        ppy.check[ppy.Buffer[int]](memoryview(ints)[::2])
    with pytest.raises(TypeError, match=r"cannot validate against Buffer\[str\]"):
        ppy.check[ppy.Buffer[str]](ints)


def test_array_refinements_read_the_arrays_own_metadata():
    numpy = pytest.importorskip("numpy")
    matrix = numpy.zeros((2, 3))
    target = Annotated[numpy.ndarray, ppy.Shape(2, "n"), ppy.DType("float64"), ppy.Contiguous()]
    assert ppy.check[target](matrix) is matrix
    assert ppy.check[Annotated[numpy.ndarray, ppy.Shape("n", "n")]](numpy.eye(3)) is not None
    with pytest.raises(TypeError, match="dimension 'n' is 2 and 3"):
        ppy.check[Annotated[numpy.ndarray, ppy.Shape("n", "n")]](matrix)
    with pytest.raises(TypeError, match=r"expected 2 dimension\(s\), got shape \(6,\)"):
        ppy.check[Annotated[numpy.ndarray, ppy.Shape(2, 3)]](matrix.reshape(6))
    with pytest.raises(TypeError, match=r"expected shape \(3, 2\), got \(2, 3\)"):
        ppy.check[Annotated[numpy.ndarray, ppy.Shape(3, 2)]](matrix)
    with pytest.raises(TypeError, match="expected dtype 'float32', got float64"):
        ppy.check[Annotated[numpy.ndarray, ppy.DType("float32")]](matrix)
    with pytest.raises(TypeError, match="not C-contiguous"):
        ppy.check[Annotated[numpy.ndarray, ppy.Contiguous()]](matrix.T)
    with pytest.raises(TypeError, match="needs a value with a shape"):
        ppy.check[Annotated[object, ppy.Shape(2)]]([1, 2])
    with pytest.raises(TypeError, match="cannot tell whether a int is contiguous"):
        ppy.check[Annotated[object, ppy.Contiguous()]](3)


@pytest.mark.parametrize(
    "target",
    [
        ppy.Owned[int],
        ppy.Borrowed[list[int]],
        ppy.Mut[ppy.Buffer[float]],
        Annotated[int, ppy.NoAlias()],
        Annotated[list[int], ppy.Owned(), ppy.Length(2)],
        ppy.Array[int, "n"],
    ],
)
def test_a_contract_no_single_value_can_witness_is_refused_not_stripped(target):
    """`Owned`, `Borrowed`, `Mut`, `NoAlias`, a symbolic length: `check` refuses the target."""
    with pytest.raises(TypeError, match=r"ppy\.check cannot validate"):
        ppy.check[target]([1, 2])
    assert ppy.assume[target]([1, 2]) == [1, 2], "the unchecked crossing still takes it"


def test_metadata_that_is_not_ppys_is_not_a_contract():
    assert ppy.check[Annotated[int, "meta", 3]](5) == 5


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
