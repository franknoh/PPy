"""JAX plugin: staged export plus PJRT execution (spec 21)."""

from __future__ import annotations

import importlib.util
from typing import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from ..analysis.symbols import FunctionInfo
from .base import CallResult, Lowering

__all__ = ["JaxPlugin", "STAGING_MARKERS", "StagedFunction", "staged_functions"]

PLUGIN_VERSION = 1

#: Decorators that mark a staged region PPY may export (spec 21.2).
STAGING_MARKERS = ("jax.jit", "jit", "ppy.jax")

_ARRAY = T.Instance("jax.Array", (), ("jax.Array", "object"))

#: Python operators mapped to the `jax.numpy` function implementing them.
_OPERATORS = {
    "+": "add", "-": "subtract", "*": "multiply", "/": "true_divide",
    "//": "floor_divide", "%": "remainder", "**": "power", "@": "matmul",
    "<": "less", "<=": "less_equal", "==": "equal", "!=": "not_equal",
    ">": "greater", ">=": "greater_equal",
    "u-": "negative", "u+": "positive",
}

#: `jax.numpy` operations whose result is an array.
_ARRAY_RESULTS = frozenset({
    "add", "subtract", "multiply", "true_divide", "divide", "floor_divide",
    "remainder", "power", "negative", "positive", "abs", "absolute", "sign",
    "sin", "cos", "tan", "tanh", "sinh", "cosh", "exp", "expm1", "log", "log1p",
    "sqrt", "rsqrt", "square", "reciprocal", "clip", "round", "floor", "ceil",
    "minimum", "maximum", "matmul", "dot", "where", "concatenate", "stack",
    "reshape", "transpose", "ravel", "squeeze", "expand_dims", "arange",
    "zeros", "ones", "full", "zeros_like", "ones_like", "eye", "linspace",
    "array", "asarray", "astype", "einsum", "take", "split",
    "less", "less_equal", "equal", "not_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not",
    "softmax", "log_softmax", "relu", "sigmoid", "erf",
})

#: Reductions: an array when an axis is given, a scalar array otherwise.
_REDUCTIONS = frozenset({"sum", "prod", "mean", "std", "var", "min", "max", "argmin", "argmax", "any", "all"})

#: Attributes of a `jax.Array` value.
_ARRAY_MEMBERS: dict[str, str] = {
    "shape": "shape", "ndim": "int", "size": "int", "dtype": "dtype",
    "T": "array", "at": "indexer",
    "reshape": "array_method", "astype": "array_method", "sum": "array_method",
    "mean": "array_method", "max": "array_method", "min": "array_method",
    "ravel": "array_method", "squeeze": "array_method", "block_until_ready": "array_method",
    "item": "float_method", "tolist": "list_method",
}


class StagedFunction:
    """A function PPY may export, with the input specification it declared."""

    __slots__ = ("info", "shapes", "dtypes", "reason")

    def __init__(
        self,
        info: FunctionInfo,
        shapes: tuple[tuple[int | str, ...], ...] = (),
        dtypes: tuple[str, ...] = (),
        reason: str = "",
    ) -> None:
        self.info = info
        self.shapes = shapes
        self.dtypes = dtypes
        self.reason = reason

    @property
    def exportable(self) -> bool:
        return bool(self.shapes) and len(self.shapes) == len(self.dtypes)

    def __repr__(self) -> str:
        return f"StagedFunction({self.info.qualname!r}, shapes={self.shapes!r})"


def staged_functions(symbols) -> list[StagedFunction]:  # type: ignore[no-untyped-def]
    """Find `@jax.jit` functions whose inputs are fully specified (spec 21.2)."""
    found: list[StagedFunction] = []
    for info in symbols.functions.values():
        if not any(marker in decorator for decorator in info.decorators for marker in STAGING_MARKERS):
            continue
        shapes: list[tuple[int | str, ...]] = []
        dtypes: list[str] = []
        reason = ""
        for param in info.params:
            if param.facts.shape is None:
                reason = f"parameter `{param.name}` declares no `ppy.Shape`"
                break
            shapes.append(param.facts.shape)
            dtypes.append(param.facts.dtype or _default_dtype(param.type))
        else:
            found.append(StagedFunction(info, tuple(shapes), tuple(dtypes)))
            continue
        found.append(StagedFunction(info, reason=reason))
    return found


def _default_dtype(t: T.Type) -> str:
    base = T.strip_literal(t)
    if isinstance(base, T.Instance) and base.name == "int":
        return "int32"
    return "float32"


class JaxPlugin:
    """Recognizes staged JAX functions; eager calls stay on the Python path."""

    name = "jax"
    modules = ("jax", "jax.numpy", "jax.lax", "jaxlib")

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.allow_build_export = bool(self.options.get("allow-build-export", False))

    def fingerprint(self) -> str:
        """JAX, jaxlib, and XLA identities all enter the cache key (spec 21.8)."""
        jax_version = jaxlib_version = "absent"
        try:
            if importlib.util.find_spec("jax") is not None:
                import jax

                jax_version = jax.__version__
                try:
                    import jaxlib

                    jaxlib_version = jaxlib.__version__
                except Exception:  # noqa: BLE001
                    jaxlib_version = "unknown"
        except Exception:  # noqa: BLE001
            jax_version = "unknown"
        return (
            f"v{PLUGIN_VERSION}:jax={jax_version}:jaxlib={jaxlib_version}"
            f":export={int(self.allow_build_export)}"
        )

    def external_types(self) -> dict[str, str]:
        return {
            "jax.Array": "jax.Array",
            "jax.numpy.ndarray": "jax.Array",
            "jax.Device": "jax.Device",
            "jax.numpy.dtype": "jax.numpy.dtype",
        }

    def operator(self, symbol: str) -> str | None:
        return _OPERATORS.get(symbol)

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        attribute = qualname.rpartition(".")[2]
        if attribute in {"pi", "e", "inf", "nan", "euler_gamma"}:
            return T.FLOAT, Facts()
        if attribute in {"float32", "float16", "bfloat16", "int32", "int64", "bool_"}:
            return T.Instance("jax.numpy.dtype", (), ("jax.numpy.dtype", "object")), Facts()
        if attribute in {"numpy", "lax", "nn", "random", "export", "stages"}:
            return T.Module_(qualname), Facts()
        if attribute == "Array":
            return T.ClassObject("jax.Array", _ARRAY), Facts()
        if qualname == "jax.__version__":
            return T.STR, Facts()
        return None

    def instance_attribute(self, type_name: str, attribute: str) -> tuple[T.Type, Facts] | None:
        """Attributes of a `jax.Array` value."""
        if type_name != "jax.Array":
            return None
        kind = _ARRAY_MEMBERS.get(attribute)
        if kind is None:
            return None
        if kind == "shape":
            return T.Tuple_((T.INT,), homogeneous=True), Facts()
        if kind == "int":
            return T.INT, Facts()
        if kind == "array":
            return _ARRAY, Facts()
        if kind == "dtype":
            return T.Instance("jax.numpy.dtype", (), ("jax.numpy.dtype", "object")), Facts()
        if kind == "array_method":
            return T.Callable_((), _ARRAY, f"jax.Array.{attribute}"), Facts()
        if kind == "float_method":
            return T.Callable_((), T.FLOAT, f"jax.Array.{attribute}"), Facts()
        if kind == "list_method":
            return T.Callable_((), T.list_of(T.ANY), f"jax.Array.{attribute}"), Facts()
        return T.UNKNOWN, Facts()

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        if not qualname.startswith(("jax.numpy.", "jax.lax.", "jax.nn.", "jax.")):
            return None
        operation = qualname.rpartition(".")[2]
        effects = EffectSet.of(Effect.ALLOC, raises=("ValueError", "TypeError"))
        if operation == "devices":
            return CallResult(
                T.list_of(T.Instance("jax.Device", (), ("jax.Device", "object"))),
                Facts(), effects, Lowering.PYTHON_FALLBACK, "a runtime query",
            )
        if operation not in _ARRAY_RESULTS and operation not in _REDUCTIONS:
            return None
        # Full eager lowering of jax.numpy is explicitly out of v1 scope (21.7).
        return CallResult(
            _ARRAY,
            Facts(),
            effects,
            Lowering.PYTHON_FALLBACK,
            "eager JAX calls outside an exported region use the Python path",
        )

    def export_permitted(self, build_execution: str) -> tuple[bool, str]:
        """Both the project policy and the plugin must allow build-time export."""
        if build_execution != "allow":
            return False, (
                "`build-execution` is not `allow`; JAX export runs project code "
                "and needs explicit permission (spec 21.8, 31.2)"
            )
        if not self.allow_build_export:
            return False, "`allow-build-export = false` disables JAX export for this project"
        return True, ""

    def staged_region(self, decorators: Sequence[str]) -> CallResult | None:
        """A `@jax.jit` function PPY may export to StableHLO at build time."""
        if not any(marker in d for d in decorators for marker in STAGING_MARKERS):
            return None
        if not self.allow_build_export:
            return CallResult(
                _ARRAY,
                Facts(),
                EffectSet.of(Effect.EXTERNAL_UNKNOWN),
                Lowering.PYTHON_FALLBACK,
                "build-time export is disabled (`allow-build-export = false`); "
                "JAX export executes project code and needs explicit permission",
            )
        return CallResult(
            _ARRAY,
            Facts(),
            EffectSet.of(Effect.ALLOC),
            Lowering.GRAPH_REGION,
            "staged and exported to StableHLO, then executed through PJRT without per-call Python dispatch",
            ("input shape and dtype match the exported signature", "PJRT client available for the target device"),
        )
