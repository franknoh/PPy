"""SciPy plugin: a curated surface, typed and lowered by family (spec 50).

Not a reimplementation of SciPy. Special functions, transforms, dense
linear algebra, and sparse matrices are typed here and named as
operations of the `special`, `fft`, `linalg`, and `sparse` dialects, so a
backend that has them lowers them and every other runs SciPy; the
callback-driven families -- optimize, integrate, stats -- stay Python
calls that carry the callback effect they have.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, DialectOperationSpec, FallbackSpec, Plugin
from .convergence import SPECIAL as _DIALECT_NAMES
from .convergence import special as _converged_special
from .numpy_plugin import _NDARRAY

__all__ = ["SPECIAL", "SciPyPlugin"]

PLUGIN_VERSION = 1

#: `scipy.special` functions with one float in and one float out.
SPECIAL = frozenset(
    {
        "erf",
        "erfc",
        "erfinv",
        "gamma",
        "gammaln",
        "digamma",
        "expit",
        "logit",
        "ndtr",
        "ndtri",
        "log_ndtr",
        "i0",
        "i1",
        "j0",
        "j1",
        "y0",
        "y1",
        "sinc",
    }
)
#: Two-argument special functions.
SPECIAL_BINARY = frozenset(
    {"beta", "betaln", "gammainc", "gammaincc", "xlogy", "jv", "yv", "iv", "kv"}
)
FFT = frozenset(
    {"fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn", "fftfreq", "rfftfreq"}
)
LINALG_ARRAY = frozenset(
    {"solve", "inv", "pinv", "cholesky", "solve_triangular", "expm", "sqrtm", "lstsq"}
)
LINALG_SCALAR = frozenset({"det", "norm"})
LINALG_TUPLE = {"qr": 2, "eig": 2, "eigh": 2, "svd": 3, "lu": 3}
SPARSE_FORMATS = ("csr_matrix", "csc_matrix", "coo_matrix", "csr_array", "csc_array", "coo_array")
SPARSE_ARRAY_METHODS = frozenset(
    {"toarray", "todense", "transpose", "tocsr", "tocsc", "tocoo", "copy"}
)
CALLBACK_DRIVEN = frozenset(
    {
        "minimize",
        "minimize_scalar",
        "root",
        "root_scalar",
        "curve_fit",
        "quad",
        "dblquad",
        "solve_ivp",
        "odeint",
    }
)

_SPARSE_TYPES = {
    f"scipy.sparse.{name}": T.Instance(
        f"scipy.sparse.{name}", (), (f"scipy.sparse.{name}", "scipy.sparse.spmatrix", "object")
    )
    for name in SPARSE_FORMATS
}
_PURE = EffectSet()
_ALLOC = EffectSet.of(Effect.ALLOC, raises=("ValueError",))
_LINALG = EffectSet.of(Effect.ALLOC, raises=("ValueError", "scipy.linalg.LinAlgError"))


def _is_float_like(t: T.Type) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name in {"int", "float", "bool"}


def _is_array(t: T.Type) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name == "numpy.ndarray"


class SciPyPlugin(Plugin):
    """Types the curated SciPy surface and names its dialect operations."""

    name = "scipy"
    modules = (
        "scipy",
        "scipy.special",
        "scipy.fft",
        "scipy.linalg",
        "scipy.sparse",
        "scipy.optimize",
        "scipy.integrate",
        "scipy.stats",
    )

    def fingerprint(self) -> str:
        version = "absent"
        try:
            if importlib.util.find_spec("scipy") is not None:
                import scipy

                version = str(scipy.__version__)
        except Exception:  # noqa: BLE001 - a broken install must not break analysis
            version = "unknown"
        return f"v{PLUGIN_VERSION}:api{self.api_version}:scipy={version}"

    def external_types(self) -> dict[str, str]:
        return {name: name for name in _SPARSE_TYPES} | {
            "scipy.sparse.spmatrix": "scipy.sparse.spmatrix"
        }

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        if qualname in _SPARSE_TYPES:
            return T.ClassObject(qualname, _SPARSE_TYPES[qualname]), Facts()
        return None

    def instance_attribute(
        self, type_name: str, attribute: str, facts: Facts | None = None
    ) -> tuple[T.Type, Facts] | None:
        if type_name not in _SPARSE_TYPES:
            return None
        if attribute == "shape":
            return T.Tuple_((T.INT, T.INT)), Facts()
        if attribute in {"nnz", "ndim"}:
            return T.INT, Facts()
        if attribute == "T":
            return _SPARSE_TYPES[type_name], Facts()
        if attribute in SPARSE_ARRAY_METHODS:
            result = _NDARRAY if attribute in {"toarray", "todense"} else _SPARSE_TYPES[type_name]
            return T.Callable_((), result, f"{type_name}.{attribute}"), Facts()
        if attribute in {"dot", "multiply"}:
            return T.Callable_((), _NDARRAY, f"{type_name}.{attribute}"), Facts()
        if attribute in {"sum", "mean", "max", "min"}:
            return T.Callable_((), T.FLOAT, f"{type_name}.{attribute}"), Facts()
        return None

    def operator(self, symbol: str) -> str | None:
        return {
            "@": "scipy.sparse.matmul",
            "*": "scipy.sparse.multiply",
            "+": "scipy.sparse.add",
        }.get(symbol)

    def tensor_operation(self, qualname: str) -> DialectOperationSpec | None:
        """A one-argument special function over arrays is `tensor.unary` of
        the `special` dialect's function, so it fuses with the arithmetic
        around it."""
        if not qualname.startswith("scipy.special."):
            return None
        return _converged_special(qualname.rpartition(".")[2])

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        family = qualname.split(".")[1] if qualname.count(".") >= 2 else ""
        operation = qualname.rpartition(".")[2]
        if family == "special":
            return self._special(operation, args)
        if family == "fft":
            return self._fft(operation, args)
        if family == "linalg":
            return self._linalg(operation, args)
        if family == "sparse":
            return self._sparse(qualname, operation, args)
        if operation in CALLBACK_DRIVEN:
            return CallResult(
                T.ANY,
                Facts(),
                EffectSet.of(Effect.PYTHON_CALLBACK, Effect.ALLOC, raises=("ValueError",)),
                FallbackSpec("drives a Python callback, so SciPy runs it"),
                "drives a Python callback",
            )
        return None

    def _special(self, operation: str, args: Sequence[tuple[T.Type, Facts]]) -> CallResult | None:
        if (operation in SPECIAL and len(args) == 1) or (
            operation in SPECIAL_BINARY and len(args) == 2
        ):
            # The dialect's name for it, where SciPy's differs (`j0` is `bessel_j0`).
            named = _DIALECT_NAMES.get(operation, operation)
            if all(_is_float_like(t) for t, _f in args):
                return CallResult(
                    T.FLOAT,
                    Facts(),
                    _PURE,
                    DialectOperationSpec("special", named),
                    f"`special.{named}` over scalars",
                )
            if all(_is_float_like(t) or _is_array(t) for t, _f in args):
                return CallResult(
                    _NDARRAY,
                    Facts(),
                    _ALLOC,
                    DialectOperationSpec("special", named, (("elementwise", True),)),
                    f"`special.{named}` elementwise over arrays",
                )
            return CallResult(
                T.ANY, Facts(), _ALLOC, FallbackSpec("operands of unknown type"), "unknown operands"
            )
        return None

    def _fft(self, operation: str, args: Sequence[tuple[T.Type, Facts]]) -> CallResult | None:
        if operation not in FFT:
            return None
        if operation in {"fftfreq", "rfftfreq"}:
            return CallResult(
                _NDARRAY,
                Facts(),
                _ALLOC,
                DialectOperationSpec("fft", operation),
                "sample frequencies",
            )
        if args and _is_array(args[0][0]):
            return CallResult(
                _NDARRAY,
                Facts(),
                _ALLOC,
                DialectOperationSpec("fft", operation),
                f"`fft.{operation}` over an array",
            )
        return CallResult(
            _NDARRAY, Facts(), _ALLOC, FallbackSpec("input is not a known array"), "unknown input"
        )

    def _linalg(self, operation: str, args: Sequence[tuple[T.Type, Facts]]) -> CallResult | None:
        arrays = all(_is_array(t) for t, _f in args) and bool(args)
        spec = DialectOperationSpec("linalg", operation)
        if operation in LINALG_ARRAY:
            return CallResult(
                _NDARRAY,
                Facts(),
                _LINALG,
                spec if arrays else FallbackSpec("not arrays"),
                operation,
            )
        if operation in LINALG_SCALAR:
            return CallResult(
                T.FLOAT, Facts(), _LINALG, spec if arrays else FallbackSpec("not arrays"), operation
            )
        if operation in LINALG_TUPLE:
            width = LINALG_TUPLE[operation]
            return CallResult(
                T.Tuple_(tuple([_NDARRAY] * width)),
                Facts(),
                _LINALG,
                spec if arrays else FallbackSpec("not arrays"),
                operation,
            )
        return None

    def _sparse(
        self, qualname: str, operation: str, args: Sequence[tuple[T.Type, Facts]]
    ) -> CallResult | None:
        if qualname in _SPARSE_TYPES:
            return CallResult(
                _SPARSE_TYPES[qualname],
                Facts(),
                _ALLOC,
                DialectOperationSpec(
                    "sparse", "construct", (("format", operation.partition("_")[0]),)
                ),
                f"a {operation.partition('_')[0].upper()} matrix",
            )
        if operation in {"eye", "identity", "random", "diags"}:
            return CallResult(
                _SPARSE_TYPES["scipy.sparse.csr_matrix"],
                Facts(),
                _ALLOC,
                DialectOperationSpec("sparse", operation),
                operation,
            )
        if operation in {"issparse", "isspmatrix"}:
            return CallResult(T.BOOL, Facts(), _PURE, FallbackSpec("a type test"), operation)
        return None
