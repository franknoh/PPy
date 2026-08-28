"""NumPy plugin: the reference native-library integration (spec 19)."""

from __future__ import annotations

import importlib.util
from typing import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, Lowering

__all__ = [
    "NumPyPlugin",
    "ELEMENTWISE",
    "REDUCTIONS",
    "LINALG",
    "FUSIBLE_UNARY",
    "FUSIBLE_BINARY",
    "FUSIBLE_REDUCTIONS",
]

PLUGIN_VERSION = 1

#: Curated ufunc-like operations that lower to fused native loops (spec 19.4).
ELEMENTWISE = frozenset({
    "add", "subtract", "multiply", "true_divide", "divide", "floor_divide",
    "remainder", "mod", "fmod", "power", "negative", "absolute", "abs", "positive",
    "less", "less_equal", "equal", "not_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "invert",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "exp", "expm1", "log", "log2", "log10", "log1p",
    "sqrt", "cbrt", "square", "reciprocal", "sign",
    "minimum", "maximum", "clip", "round", "around", "rint", "floor", "ceil", "trunc",
})

#: Reductions supported for the v1 fast path (spec 19.8).
REDUCTIONS = frozenset({"sum", "prod", "product", "min", "max", "mean", "any", "all"})

#: Operations the LLVM backend has a generated kernel for. Only these are
#: reported as `Intrinsic`; the rest are typed but left to NumPy's own dispatch.
FUSIBLE_UNARY = frozenset({
    "sin", "cos", "exp", "log", "log2", "log10", "sqrt", "absolute", "abs", "negative",
})
FUSIBLE_BINARY = frozenset({
    "add", "subtract", "multiply", "true_divide", "divide", "minimum", "maximum", "power",
})
FUSIBLE_REDUCTIONS = frozenset({"sum", "prod", "product", "max", "min", "mean"})
FUSIBLE = FUSIBLE_UNARY | FUSIBLE_BINARY | FUSIBLE_REDUCTIONS

#: Linear algebra lowered to an available BLAS routine (spec 19.9).
LINALG = frozenset({"dot", "matmul", "inner", "vdot", "tensordot"})

#: Creation routines that allocate a new array.
CREATION = frozenset({
    "array", "asarray", "zeros", "ones", "empty", "full", "arange", "linspace",
    "zeros_like", "ones_like", "empty_like", "full_like", "eye", "identity",
})

_BOOL_RESULTS = frozenset({
    "less", "less_equal", "equal", "not_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "any", "all",
})

_SCALAR_DTYPES = (
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "bool_", "complex64", "complex128",
)

#: dtypes the fused fast path accepts (spec 19.3).
_SUPPORTED_DTYPES = frozenset(_SCALAR_DTYPES) - {"complex64", "complex128"}

_NDARRAY = T.Instance("numpy.ndarray", (), ("numpy.ndarray", "object"))

#: Array methods that return another array of the same dtype.
_ARRAY_SHAPING = frozenset({
    "reshape", "ravel", "flatten", "copy", "transpose", "squeeze", "astype",
    "conj", "conjugate", "view", "clip", "round", "repeat", "take",
})

#: Python operators mapped to the ufunc that implements them.
_OPERATORS = {
    "+": "add", "-": "subtract", "*": "multiply", "/": "true_divide",
    "//": "floor_divide", "%": "remainder", "**": "power", "@": "matmul",
    "<": "less", "<=": "less_equal", "==": "equal", "!=": "not_equal",
    ">": "greater", ">=": "greater_equal",
    "u-": "negative", "u+": "positive", "u~": "invert",
}


class NumPyPlugin:
    """Types, effects, and lowering decisions for exact `numpy.ndarray` values."""

    name = "numpy"
    modules = ("numpy", "numpy.linalg")

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.fusion = bool(self.options.get("fusion", True))
        self.internal_api = bool(self.options.get("internal-api", False))

    def fingerprint(self) -> str:
        """Plugin version plus the installed NumPy build (spec 18.3)."""
        version = "absent"
        try:
            if importlib.util.find_spec("numpy") is not None:
                import numpy

                version = f"{numpy.__version__}"
        except Exception:  # noqa: BLE001 - a broken install must not break analysis
            version = "unknown"
        return f"v{PLUGIN_VERSION}:numpy={version}:internal={int(self.internal_api)}"

    def external_types(self) -> dict[str, str]:
        types = {
            "numpy.ndarray": "numpy.ndarray",
            "numpy.dtype": "numpy.dtype",
            "numpy.generic": "numpy.generic",
        }
        for dtype in _SCALAR_DTYPES:
            types[f"numpy.{dtype}"] = f"numpy.{dtype}"
        return types

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        attribute = qualname.rpartition(".")[2]
        if attribute in {"pi", "e", "euler_gamma", "inf", "nan"}:
            return T.FLOAT, Facts()
        if attribute == "ndarray":
            return T.ClassObject("numpy.ndarray", _NDARRAY), Facts()
        return None

    def instance_attribute(
        self, type_name: str, attribute: str, facts: Facts | None = None
    ) -> tuple[T.Type, Facts] | None:
        """Methods and attributes of an exact `numpy.ndarray` value."""
        if type_name != "numpy.ndarray":
            return None
        if attribute in _ARRAY_SHAPING:
            return T.Callable_((), _NDARRAY, f"numpy.ndarray.{attribute}"), Facts()
        if attribute == "tolist":
            return T.Callable_((), T.list_of(T.ANY), "numpy.ndarray.tolist"), Facts()
        if attribute == "item":
            return T.Callable_((), T.FLOAT, "numpy.ndarray.item"), Facts()
        if attribute == "fill":
            return T.Callable_((), T.NONE, "numpy.ndarray.fill"), Facts()
        return None


    def subscript(
        self, type_name: str, *, is_slice: bool, tupled: bool
    ) -> tuple[T.Type, Facts] | None:
        """Indexing an array: a slice stays an array, a full index is a scalar."""
        if type_name != "numpy.ndarray":
            return None
        if is_slice or tupled:
            return _NDARRAY, Facts()
        # A single index into an array of unknown rank may still be an array,
        # so the result is the element type only when the rank is known to be
        # one, which v1 does not track. `_ARRAY` is the safe answer.
        return _NDARRAY, Facts()

    def operator(self, symbol: str) -> str | None:
        """The ufunc implementing a Python operator, if any."""
        return _OPERATORS.get(symbol)

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        operation = qualname.rpartition(".")[2]
        if operation in CREATION:
            return CallResult(
                type=_NDARRAY,
                facts=self._creation_facts(operation, args, keywords),
                effects=EffectSet.of(Effect.ALLOC, raises=("ValueError", "TypeError", "MemoryError")),
                lowering=Lowering.DIRECT_NATIVE_CALL,
                reason="allocated through the public NumPy C API",
            )
        if operation in ELEMENTWISE:
            return self._elementwise(operation, args, keywords)
        if operation in REDUCTIONS:
            return self._reduction(operation, args, keywords)
        if operation in LINALG:
            return self._linalg(operation, args, keywords)
        return None

    def _elementwise(
        self,
        operation: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult:
        result_type: T.Type = _NDARRAY if self._any_array(args) else T.FLOAT
        if operation in _BOOL_RESULTS and not self._any_array(args):
            result_type = T.BOOL
        effects = EffectSet.of(Effect.ALLOC, raises=("ValueError", "TypeError"))
        lowering, reason, guards = self._fast_path(operation, args, keywords)
        if "out" in keywords:
            # `out=` aliases a caller-visible buffer (spec 19.10).
            effects = effects.add(Effect.WRITE_OBJECT)
            if lowering is Lowering.INTRINSIC:
                lowering = Lowering.DIRECT_NATIVE_CALL
                reason = "`out=` requires overlap-safe NumPy semantics"
        return CallResult(result_type, Facts(), effects, lowering, reason, guards)

    def _reduction(
        self,
        operation: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult:
        has_axis = "axis" in keywords or len(args) > 1
        result_type: T.Type = _NDARRAY if has_axis else self._scalar_result(operation, args)
        lowering, reason, guards = self._fast_path(operation, args, keywords)
        if lowering is Lowering.INTRINSIC and operation in {"sum", "prod", "product", "mean"}:
            # Strict float reduction order is retained unless reassociation is
            # explicitly permitted (spec 19.8).
            reason = "fused strided reduction with strict floating-point order"
        return CallResult(
            result_type,
            Facts(),
            EffectSet.of(Effect.ALLOC, raises=("ValueError", "TypeError")),
            lowering,
            reason,
            guards,
        )

    def _linalg(
        self,
        operation: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult:
        return CallResult(
            _NDARRAY,
            Facts(),
            EffectSet.of(Effect.ALLOC, raises=("ValueError",)),
            Lowering.DIRECT_NATIVE_CALL,
            "lowered to the installed BLAS routine without a layout-forcing copy",
            ("exact ndarray", "supported dtype", "shape compatibility"),
        )

    def _fast_path(
        self,
        operation: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> tuple[Lowering, str, tuple[str, ...]]:
        """Decide the lowering for one operation (spec 19.3)."""
        if not self.fusion:
            return Lowering.PYTHON_FALLBACK, "fusion is disabled by project configuration", ()
        for name, (_type, facts) in keywords.items():
            if name not in {"axis", "out", "dtype", "keepdims"}:
                return Lowering.PYTHON_FALLBACK, f"unsupported keyword `{name}`", ()
        dtype = keywords.get("dtype")
        if dtype is not None and dtype[1].has_constant:
            named = str(dtype[1].constant)
            if named not in _SUPPORTED_DTYPES:
                return Lowering.PYTHON_FALLBACK, f"dtype `{named}` is outside the fast-path domain", ()
        for arg_type, _facts in args:
            base = T.strip_literal(arg_type)
            if isinstance(base, (T.AnyType, T.UnknownType)):
                return Lowering.PYTHON_FALLBACK, "an operand has no static type", ()
            if isinstance(base, T.Instance) and base.name not in {
                "numpy.ndarray", "int", "float", "bool", *[f"numpy.{d}" for d in _SCALAR_DTYPES]
            }:
                return Lowering.PYTHON_FALLBACK, f"operand `{base}` is outside the fast-path domain", ()
        guards = (
            "exact numpy.ndarray (no __array_ufunc__ override)",
            "float64 dtype in native byte order",
            "C-contiguous layout",
            "identical shapes across array operands",
            "a floating-point error state the generated loop can honor",
        )
        if operation not in FUSIBLE:
            return (
                Lowering.DIRECT_NATIVE_CALL,
                f"`{operation}` has no generated kernel, so NumPy's own loop runs",
                (),
            )
        if not self._any_array(args):
            return (
                Lowering.DIRECT_NATIVE_CALL,
                f"`{operation}` has no array operand, so there is no loop to fuse",
                (),
            )
        return (
            Lowering.INTRINSIC,
            f"`{operation}` fused into one strided loop",
            guards,
        )

    def _any_array(self, args: Sequence[tuple[T.Type, Facts]]) -> bool:
        return any(
            isinstance(base := T.strip_literal(t), T.Instance) and base.name == "numpy.ndarray"
            for t, _ in args
        )

    def _scalar_result(self, operation: str, args: Sequence[tuple[T.Type, Facts]]) -> T.Type:
        if operation in _BOOL_RESULTS:
            return T.BOOL
        if operation == "mean":
            return T.FLOAT
        return T.FLOAT

    def _creation_facts(
        self,
        operation: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> Facts:
        facts = Facts(exact_class="numpy.ndarray", contiguous=True)
        if operation in {"zeros", "ones", "empty", "full", "arange"} and args:
            length = args[0][1]
            if length.has_constant and isinstance(length.constant, int):
                facts = facts.with_(shape=(length.constant,))
        return facts
