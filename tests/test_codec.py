"""The type codec must be an exact inverse; anything less corrupts the cache."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.codec import (
    CodecError,
    decode_effects,
    decode_facts,
    decode_type,
    encode_effects,
    encode_facts,
    encode_type,
)
from ppy_compiler.analysis.effects import Effect, EffectSet
from ppy_compiler.analysis.refinements import Facts, IntRange

SAMPLES: list[T.Type] = [
    T.ANY,
    T.UNKNOWN,
    T.UnknownType("because"),
    T.NEVER,
    T.INT,
    T.FLOAT,
    T.STR,
    T.BYTES,
    T.BOOL,
    T.NONE,
    T.list_of(T.INT),
    T.dict_of(T.STR, T.list_of(T.FLOAT)),
    T.Tuple_((T.INT, T.STR)),
    T.Tuple_((T.FLOAT,), homogeneous=True),
    T.Tuple_(()),
    T.union(T.INT, T.NONE),
    T.union(T.Literal("a", T.STR), T.Literal("b", T.STR)),
    T.Literal(3, T.INT),
    T.Literal(True, T.BOOL),
    T.Literal(-1.5, T.FLOAT),
    T.Literal(b"bytes", T.BYTES),
    T.Literal("text", T.STR),
    T.Module_("numpy"),
    T.ClassObject("m.Box", T.Instance("m.Box", (), ("m.Box", "object"))),
    T.TypeVar_("Tv"),
    T.TypeVar_("Tv", T.INT),
    T.Callable_((T.Param("x", T.INT), T.Param("y", T.STR, True, "keyword_only")), T.FLOAT, "m.f"),
    T.Callable_((), T.NONE, "m.g", is_async=True, is_generator=True),
    T.Instance("Sequence", (T.FLOAT,), ("Sequence", "Iterable", "object")),
]


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s)[:40])
def test_a_type_survives_a_round_trip(sample: T.Type):
    restored = decode_type(encode_type(sample))
    assert restored == sample
    assert type(restored) is type(sample)
    assert str(restored) == str(sample)


def test_a_bool_literal_does_not_come_back_as_an_int():
    """`bool` is a subclass of `int`, so the order of the checks matters."""
    restored = decode_type(encode_type(T.Literal(True, T.BOOL)))
    assert isinstance(restored, T.Literal)
    assert restored.value is True
    assert not isinstance(restored.value, int) or restored.value.__class__ is bool


FACTS: list[Facts] = [
    Facts(),
    Facts(int_range=IntRange(-3, 9)),
    Facts(length=4),
    Facts(constant=7, has_constant=True),
    Facts(constant="s", has_constant=True),
    Facts(constant=None, has_constant=True),
    Facts(exact_class="numpy.ndarray", non_null=True),
    Facts(contiguous=True, no_alias=True),
    Facts(shape=(2, "B"), dtype="float32"),
    Facts(width=(64, True), float_bits=32),
]


@pytest.mark.parametrize("facts", FACTS, ids=lambda f: str(f)[:40])
def test_facts_survive_a_round_trip(facts: Facts):
    assert decode_facts(encode_facts(facts)) == facts


def test_effects_survive_a_round_trip():
    for effects in (
        EffectSet(),
        EffectSet.of(Effect.IO),
        EffectSet.of(Effect.ALLOC, Effect.WRITE_OBJECT, raises=("ValueError", "KeyError")),
    ):
        assert decode_effects(encode_effects(effects)) == effects


def test_an_unknown_shape_is_refused_rather_than_guessed():
    class _Alien(T.Type):
        pass

    with pytest.raises(CodecError):
        encode_type(_Alien())
    with pytest.raises(CodecError):
        decode_type(["nonsense"])
    with pytest.raises(CodecError):
        decode_type("not a list")


def test_every_type_the_checker_produced_survives(tmp_path: Path):
    """Round-trip whatever a real analysis actually put in its summaries."""
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    path = tmp_path / "wide.ppy"
    path.write_text(
        textwrap.dedent(
            """
            from collections.abc import Sequence
            from dataclasses import dataclass

            import ppy


            @dataclass
            class Box:
                width: float
                tags: list[str]


            @ppy.pure
            def measure(boxes: Sequence[Box], scale: float) -> tuple[float, int]:
                total: float = 0.0
                names: list[str] = []
                for box in boxes:
                    total += box.width * scale
                    names.extend(box.tags)
                return (total, len(names))


            def maybe(flag: bool) -> Box | None:
                return Box(1.0, ["a"]) if flag else None


            def kind(value: int | str | None) -> str:
                if value is None:
                    return "none"
                if isinstance(value, str):
                    return "text"
                return "number"
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    bundle = analyze_paths(open_project(path), [path], backend="python")
    module = bundle.analysis.modules["wide"]

    seen = 0
    for function in module.functions.values():
        for value in (function.inferred_ret, *function.locals.values()):
            assert decode_type(encode_type(value)) == value, value
            seen += 1
        assert decode_facts(encode_facts(function.ret_facts)) == function.ret_facts
        assert decode_effects(encode_effects(function.effects)) == function.effects
    for value in module.node_types.values():
        assert decode_type(encode_type(value)) == value, value
        seen += 1
    for facts in module.node_facts.values():
        assert decode_facts(encode_facts(facts)) == facts
    assert seen > 40, f"the sample was too small to be evidence ({seen})"


def test_dynamic_and_any_encode_distinctly():
    from ppy_compiler.analysis import types as T
    from ppy_compiler.analysis.codec import decode_type, encode_type

    assert encode_type(T.DYNAMIC) != encode_type(T.ANY)
    assert decode_type(encode_type(T.DYNAMIC)) is T.DYNAMIC
    assert decode_type(encode_type(T.ANY)) is T.ANY
