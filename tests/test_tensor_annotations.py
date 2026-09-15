"""The dependency-free common tensor annotation and its analysis meaning."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

import ppy
from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.refinements import Facts

N = "N"


def test_tensor_annotations_import_and_resolve_without_frameworks() -> None:
    def kernel(
        x: ppy.Tensor[ppy.bf16, (3840,)],
        out: ppy.Mut[ppy.Tensor[ppy.f32, (N, 4)]],
    ) -> ppy.Tensor[ppy.i8, ()]:
        raise NotImplementedError

    hints = get_type_hints(kernel, include_extras=True)
    assert get_origin(hints["x"]) is Annotated
    x_base, x_spec = get_args(hints["x"])
    assert x_base is ppy.Tensor
    assert (x_spec.dtype, x_spec.shape) == ("bfloat16", (3840,))
    out_base, out_spec, ownership = get_args(hints["out"])
    assert out_base is ppy.Tensor
    assert (out_spec.dtype, out_spec.shape, ownership.mode) == ("float32", ("N", 4), "mut")
    _ret_base, ret_spec = get_args(hints["return"])
    assert (ret_spec.dtype, ret_spec.shape) == ("int8", ())

    command = (
        "import sys, ppy; "
        "print(any(name == root or name.startswith(root + '.') "
        "for root in ('numpy', 'torch', 'jax') for name in sys.modules))"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "False"


def test_tensor_runtime_validation_is_structural_and_keeps_bfloat_distinct() -> None:
    class TensorLike:
        def __init__(self, shape, dtype) -> None:
            self.shape = shape
            self.dtype = dtype

    value = TensorLike((2, 2), "bfloat16")
    assert ppy.check[ppy.Tensor](value) is value
    assert ppy.check[ppy.Tensor[ppy.bf16, ("N", "N")]](value) is value
    scalar = TensorLike((), "bf16")
    assert ppy.check[ppy.Tensor[ppy.bf16, ()]](scalar) is scalar
    with pytest.raises(TypeError, match="bfloat16"):
        ppy.check[ppy.Tensor[ppy.bf16, (2, 2)]](TensorLike((2, 2), "float16"))
    with pytest.raises(TypeError, match="dimension 'N'"):
        ppy.check[ppy.Tensor[ppy.bf16, ("N", "N")]](TensorLike((2, 3), "bfloat16"))
    assert ppy.check[ppy.bf16](1.0e20) == 1.0e20
    with pytest.raises(TypeError, match="f16"):
        ppy.check[ppy.f16](1.0e20)


def test_annotated_tensor_always_checks_the_structural_base() -> None:
    with pytest.raises(TypeError, match="tensor-like"):
        ppy.check[Annotated[ppy.Tensor, "description"]](5)
    with pytest.raises(TypeError, match="tensor-like"):
        ppy.check[Annotated[ppy.Tensor, ppy.Shape(2)]](SimpleNamespace(shape=(2,)))
    with pytest.raises(TypeError, match="tensor-like"):
        ppy.check[Annotated[ppy.Tensor, ppy.DType("bf16")]](SimpleNamespace(dtype="bf16"))


def test_legacy_tensor_metadata_rejects_invalid_contracts_at_runtime() -> None:
    value = SimpleNamespace(shape=(2,), dtype="garbage")
    invalid = [
        Annotated[ppy.Tensor, ppy.Shape(""), ppy.DType("garbage")],
        Annotated[ppy.Tensor, ppy.Shape(-1), ppy.DType("bf16")],
        Annotated[ppy.Tensor, ppy.Shape(True), ppy.DType("bf16")],
    ]
    for annotation in invalid:
        with pytest.raises(TypeError):
            ppy.check[annotation](value)


def test_tensor_dtype_metadata_requires_a_supported_matching_scalar() -> None:
    invalid = [
        Annotated[float, ppy.FloatWidth(7)],
        Annotated[float, ppy.FloatFormat("garbage")],
        Annotated[str, ppy.IntWidth(8, True)],
    ]
    for dtype in invalid:
        with pytest.raises(TypeError, match="dtype"):
            _ = ppy.Tensor[dtype, (2,)]


def test_bfloat16_runtime_range_uses_its_own_finite_maximum() -> None:
    maximum = (2 - 2**-7) * 2**127
    assert ppy.check[ppy.bf16](maximum) == maximum
    with pytest.raises(TypeError, match="bfloat16"):
        ppy.check[ppy.bf16](3.4e38)


@pytest.mark.parametrize(
    "declaration",
    [
        pytest.param(lambda: ppy.Tensor[ppy.bf16], id="wrong-arity"),
        pytest.param(lambda: ppy.Tensor[ppy.bf16, 3], id="shape-not-tuple"),
        pytest.param(lambda: ppy.Tensor[object, (3,)], id="unknown-dtype"),
        pytest.param(lambda: ppy.Tensor[ppy.bf16, (-1,)], id="negative-dimension"),
        pytest.param(lambda: ppy.Tensor[ppy.bf16, ("",)], id="empty-symbol"),
        pytest.param(lambda: ppy.Tensor[ppy.bf16, (True,)], id="boolean-dimension"),
    ],
)
def test_runtime_tensor_declarations_reject_invalid_dtype_or_shape(
    declaration: Callable[[], object],
) -> None:
    with pytest.raises(TypeError):
        declaration()


def test_analysis_resolves_tensor_forms_to_one_nominal_type(write, analyze) -> None:
    path = write(
        "tensor_types.ppy",
        """
        from typing import Annotated
        import ppy

        def kernel(
            bare: ppy.Tensor,
            x: ppy.Tensor[ppy.bf16, (3840,)],
            out: ppy.Mut[ppy.Tensor[ppy.f32, ("N", 4)]],
            legacy: Annotated[ppy.Tensor, ppy.Shape("N", 4), ppy.DType("bf16")],
            result: ppy.Tensor[ppy.i8, ()],
        ) -> ppy.Tensor[ppy.i8, ()]:
            return result
        """,
    )
    bundle = analyze(path)
    errors = [d for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]
    assert not errors, [d.message for d in errors]
    function = bundle.symbols.modules["tensor_types"].functions["kernel"]
    params = {param.name: param for param in function.params}
    assert all(T.is_tensor(param.type) for param in params.values())
    assert params["bare"].facts == Facts()
    assert params["x"].type.name == "ppy.Tensor"
    assert params["x"].facts == Facts(dtype="bfloat16", shape=(3840,))
    assert params["out"].facts == Facts(dtype="float32", shape=("N", 4), ownership="mut")
    assert params["legacy"].facts == Facts(dtype="bfloat16", shape=("N", 4))
    assert T.is_tensor(function.ret)
    assert function.ret_facts == Facts(dtype="int8", shape=())


def test_analysis_rejects_invalid_tensor_declarations(write, analyze) -> None:
    path = write(
        "bad_tensors.ppy",
        """
        import ppy

        def wrong_arity(x: ppy.Tensor[ppy.bf16]) -> None: ...
        def shape_is_not_a_tuple(x: ppy.Tensor[ppy.bf16, 3]) -> None: ...
        def unknown_dtype(x: ppy.Tensor[object, (3,)]) -> None: ...
        def negative_dimension(x: ppy.Tensor[ppy.bf16, (-1,)]) -> None: ...
        def empty_symbol(x: ppy.Tensor[ppy.bf16, ("",)]) -> None: ...
        def boolean_dimension(x: ppy.Tensor[ppy.bf16, (True,)]) -> None: ...
        """,
    )
    errors = [d for d in analyze(path).diagnostics.sorted() if d.severity.name == "ERROR"]
    assert [d.code for d in errors].count("E1301") == 6
    messages = "\n".join(d.message for d in errors)
    assert "dtype" in messages and "shape" in messages


def test_tensor_facts_merge_retains_matching_shape_dtype_and_ownership() -> None:
    facts = Facts(dtype="bfloat16", shape=("N", 4), ownership="mut")
    assert facts.merge(facts) == facts
    assert facts.merge(facts.with_(ownership="borrowed")).ownership is None
