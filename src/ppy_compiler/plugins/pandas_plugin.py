"""pandas plugin: DataFrame, Series, and Index typed by what they are (spec 57-59).

The curated surface -- selection, boolean filters, assignment, arithmetic,
comparison, null handling, casting, sorting, aggregation, grouped
aggregation, merges, concatenation -- is typed and named as `columnar`
operations for a backend that has that dialect. Everything the model does
not capture exactly -- the index, nullable dtypes, `NA` against `NaN`,
categoricals, time zones, extension dtypes, duplicate column names, the
copy-or-view question -- keeps the Python implementation: a DataFrame is
never "roughly a 2-D tensor".
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, DialectOperationSpec, FallbackSpec, Plugin
from .numpy_plugin import _NDARRAY

__all__ = ["PandasPlugin"]

PLUGIN_VERSION = 1

_FRAME = T.Instance("pandas.DataFrame", (), ("pandas.DataFrame", "object"))
_SERIES = T.Instance("pandas.Series", (), ("pandas.Series", "object"))
_INDEX = T.Instance("pandas.Index", (), ("pandas.Index", "object"))
_GROUPED = T.Instance("pandas.DataFrameGroupBy", (), ("pandas.DataFrameGroupBy", "object"))
_ALLOC = EffectSet.of(Effect.ALLOC, raises=("ValueError", "KeyError", "TypeError"))
_READ = EffectSet.of(Effect.IO, Effect.ALLOC, raises=("OSError", "ValueError"))
_WRITE = EffectSet.of(Effect.IO, raises=("OSError", "ValueError"))
_CALLBACK = EffectSet.of(Effect.PYTHON_CALLBACK, Effect.ALLOC, raises=("ValueError", "TypeError"))

#: Aggregations that reduce a Series to a scalar and a DataFrame to a Series.
AGGREGATIONS = frozenset({"sum", "mean", "min", "max", "count", "std", "var", "median", "prod"})
#: Methods that keep the shape of what they are called on, and name a columnar operation.
COLUMNAR_METHODS: dict[str, str] = {
    "fillna": "fill_null",
    "isna": "is_null",
    "isnull": "is_null",
    "notna": "is_valid",
    "notnull": "is_valid",
    "astype": "cast",
    "sort_values": "sort",
    "dropna": "filter",
    "abs": "abs",
    "round": "round",
    "clip": "clip",
    "copy": "copy",
    "head": "slice",
    "tail": "slice",
    "reset_index": "reset_index",
}
FRAME_ONLY: dict[str, str] = {"assign": "project", "drop": "project", "rename": "project"}
#: pandas' operator names, as the columnar dialect spells the operations.
_COLUMNAR_OPERATORS: dict[str, str] = {
    "truediv": "div",
    "eq": "equal",
    "ne": "not_equal",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
    "and_": "and",
    "or_": "or",
}
CALLBACK_METHODS = frozenset({"apply", "map", "applymap", "transform", "agg", "aggregate", "pipe"})
READERS = frozenset({"read_csv", "read_parquet", "read_json", "read_feather", "read_excel"})
WRITERS = frozenset({"to_csv", "to_parquet", "to_json", "to_feather", "to_excel"})


def _is(t: T.Type, name: str) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name == name


class PandasPlugin(Plugin):
    """Types the curated pandas surface and names its columnar operations."""

    name = "pandas"
    modules = ("pandas",)

    def fingerprint(self) -> str:
        version = "absent"
        try:
            if importlib.util.find_spec("pandas") is not None:
                import pandas

                version = str(pandas.__version__)
        except Exception:  # noqa: BLE001 - a broken install must not break analysis
            version = "unknown"
        return f"v{PLUGIN_VERSION}:api{self.api_version}:pandas={version}"

    def external_types(self) -> dict[str, str]:
        return {
            "pandas.DataFrame": "pandas.DataFrame",
            "pandas.Series": "pandas.Series",
            "pandas.Index": "pandas.Index",
            "pandas.DataFrameGroupBy": "pandas.DataFrameGroupBy",
        }

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        classes = {"pandas.DataFrame": _FRAME, "pandas.Series": _SERIES, "pandas.Index": _INDEX}
        if qualname in classes:
            return T.ClassObject(qualname, classes[qualname]), Facts()
        if qualname in {"pandas.NA", "pandas.NaT"}:
            return T.ANY, Facts()
        return None

    def instance_attribute(
        self, type_name: str, attribute: str, facts: Facts | None = None
    ) -> tuple[T.Type, Facts] | None:
        if type_name == "pandas.DataFrame":
            return self._frame_attribute(attribute)
        if type_name == "pandas.Series":
            return self._series_attribute(attribute)
        if type_name == "pandas.Index":
            if attribute in {"values", "to_numpy"}:
                return _NDARRAY, Facts()
            if attribute == "tolist":
                return T.Callable_((), T.list_of(T.ANY), "pandas.Index.tolist"), Facts()
            return None
        if type_name == "pandas.DataFrameGroupBy":
            if attribute in AGGREGATIONS or attribute in {"size", "first", "last", "nunique"}:
                return T.Callable_((), _FRAME, f"pandas.DataFrameGroupBy.{attribute}"), Facts()
            if attribute in CALLBACK_METHODS:
                return T.Callable_((), _FRAME, f"pandas.DataFrameGroupBy.{attribute}"), Facts()
        return None

    def _frame_attribute(self, attribute: str) -> tuple[T.Type, Facts] | None:
        if attribute == "shape":
            return T.Tuple_((T.INT, T.INT)), Facts()
        if attribute in {"columns", "index"}:
            return _INDEX, Facts()
        if attribute == "values":
            return _NDARRAY, Facts()
        if attribute == "dtypes":
            return _SERIES, Facts()
        if attribute == "empty":
            return T.BOOL, Facts()
        if attribute in {"size", "ndim"}:
            return T.INT, Facts()
        if attribute in {"loc", "iloc", "at", "iat"}:
            return T.ANY, Facts()
        if attribute == "to_numpy":
            return T.Callable_((), _NDARRAY, "pandas.DataFrame.to_numpy"), Facts()
        if attribute in AGGREGATIONS or attribute in {"describe", "nunique"}:
            result = _FRAME if attribute == "describe" else _SERIES
            return T.Callable_((), result, f"pandas.DataFrame.{attribute}"), Facts()
        if attribute in COLUMNAR_METHODS or attribute in FRAME_ONLY:
            return T.Callable_((), _FRAME, f"pandas.DataFrame.{attribute}"), Facts()
        if attribute in {"merge", "join", "sort_index", "set_index", "query", "sample"}:
            return T.Callable_((), _FRAME, f"pandas.DataFrame.{attribute}"), Facts()
        if attribute == "groupby":
            return T.Callable_((), _GROUPED, "pandas.DataFrame.groupby"), Facts()
        if attribute in CALLBACK_METHODS:
            return T.Callable_((), T.ANY, f"pandas.DataFrame.{attribute}"), Facts()
        if attribute in WRITERS:
            return T.Callable_((), T.NONE, f"pandas.DataFrame.{attribute}"), Facts()
        if attribute in {"to_dict", "to_records"}:
            return T.Callable_((), T.ANY, f"pandas.DataFrame.{attribute}"), Facts()
        return None

    def _series_attribute(self, attribute: str) -> tuple[T.Type, Facts] | None:
        if attribute == "shape":
            return T.Tuple_((T.INT,)), Facts()
        if attribute == "values":
            return _NDARRAY, Facts()
        if attribute == "index":
            return _INDEX, Facts()
        if attribute in {"dtype", "name"}:
            return T.ANY, Facts()
        if attribute in {"size", "ndim"}:
            return T.INT, Facts()
        if attribute == "empty":
            return T.BOOL, Facts()
        if attribute in {"loc", "iloc", "at", "iat", "str", "dt"}:
            return T.ANY, Facts()
        if attribute == "to_numpy":
            return T.Callable_((), _NDARRAY, "pandas.Series.to_numpy"), Facts()
        if attribute == "tolist":
            return T.Callable_((), T.list_of(T.ANY), "pandas.Series.tolist"), Facts()
        if attribute in AGGREGATIONS or attribute in {"nunique", "idxmax", "idxmin"}:
            result = T.INT if attribute in {"count", "nunique"} else T.FLOAT
            if attribute in {"idxmax", "idxmin"}:
                result = T.ANY
            return T.Callable_((), result, f"pandas.Series.{attribute}"), Facts()
        if attribute in COLUMNAR_METHODS or attribute in {
            "unique",
            "value_counts",
            "between",
            "isin",
        }:
            result = _NDARRAY if attribute == "unique" else _SERIES
            return T.Callable_((), result, f"pandas.Series.{attribute}"), Facts()
        if attribute in CALLBACK_METHODS:
            return T.Callable_((), _SERIES, f"pandas.Series.{attribute}"), Facts()
        if attribute in WRITERS:
            return T.Callable_((), T.NONE, f"pandas.Series.{attribute}"), Facts()
        return None

    def subscript(
        self, type_name: str, *, is_slice: bool, tupled: bool
    ) -> tuple[T.Type, Facts] | None:
        if type_name == "pandas.DataFrame":
            # `frame["col"]` is a Series; `frame[["a", "b"]]`, `frame[mask]`,
            # and a slice are frames. The key's type is not known here, so
            # the answer a caller can always narrow from is the frame.
            return (_FRAME if is_slice or tupled else T.ANY), Facts()
        if type_name == "pandas.Series":
            return (_SERIES if is_slice else T.ANY), Facts()
        if type_name == "pandas.Index":
            return (_INDEX if is_slice else T.ANY), Facts()
        return None

    def operator(self, symbol: str) -> str | None:
        return {
            "+": "pandas.add",
            "-": "pandas.sub",
            "*": "pandas.mul",
            "/": "pandas.truediv",
            "//": "pandas.floordiv",
            "%": "pandas.mod",
            "**": "pandas.pow",
            "==": "pandas.eq",
            "!=": "pandas.ne",
            "<": "pandas.lt",
            "<=": "pandas.le",
            ">": "pandas.gt",
            ">=": "pandas.ge",
            "&": "pandas.and_",
            "|": "pandas.or_",
            "~": "pandas.invert",
        }.get(symbol)

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        operation = qualname.rpartition(".")[2]
        owner = qualname.rpartition(".")[0]
        if qualname in {"pandas.DataFrame", "pandas.Series", "pandas.Index"}:
            result = {"pandas.DataFrame": _FRAME, "pandas.Series": _SERIES, "pandas.Index": _INDEX}[
                qualname
            ]
            return CallResult(
                result, Facts(), _ALLOC, FallbackSpec("constructed by pandas"), "constructor"
            )
        if operation in READERS:
            return CallResult(_FRAME, Facts(), _READ, FallbackSpec("reads a file"), "reads a file")
        if operation == "concat":
            return CallResult(
                _FRAME, Facts(), _ALLOC, DialectOperationSpec("columnar", "concat"), "concatenation"
            )
        if operation == "merge":
            return CallResult(
                _FRAME, Facts(), _ALLOC, DialectOperationSpec("columnar", "join"), "join"
            )
        if operation in {"isna", "isnull", "notna", "notnull"} and owner == "pandas":
            return CallResult(T.ANY, Facts(), EffectSet(), FallbackSpec("null test"), "null test")
        if owner in {"pandas.DataFrame", "pandas.Series", "pandas.DataFrameGroupBy"}:
            return self._method(owner, operation, args)
        # Operators between frames and series, from `operator`.
        if owner == "pandas" and operation in {
            "add",
            "sub",
            "mul",
            "truediv",
            "floordiv",
            "mod",
            "pow",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "and_",
            "or_",
            "invert",
        }:
            frame = any(_is(t, "pandas.DataFrame") for t, _f in args)
            result = _FRAME if frame else _SERIES
            return CallResult(
                result,
                Facts(),
                _ALLOC,
                DialectOperationSpec("columnar", _COLUMNAR_OPERATORS.get(operation, operation)),
                "elementwise over columns",
            )
        return None

    def _method(
        self, owner: str, operation: str, args: Sequence[tuple[T.Type, Facts]]
    ) -> CallResult | None:
        is_frame = owner == "pandas.DataFrame"
        if operation in CALLBACK_METHODS:
            return CallResult(
                _FRAME if owner != "pandas.Series" else _SERIES,
                Facts(),
                _CALLBACK,
                FallbackSpec("drives a Python callback"),
                "drives a Python callback",
            )
        if operation in WRITERS:
            return CallResult(
                T.NONE, Facts(), _WRITE, FallbackSpec("writes a file"), "writes a file"
            )
        if operation in AGGREGATIONS:
            if owner == "pandas.DataFrameGroupBy":
                return CallResult(
                    _FRAME, Facts(), _ALLOC, DialectOperationSpec("columnar", "group_by"), "grouped"
                )
            if is_frame:
                return CallResult(
                    _SERIES,
                    Facts(),
                    _ALLOC,
                    DialectOperationSpec("columnar", "aggregate"),
                    operation,
                )
            scalar = T.INT if operation == "count" else T.FLOAT
            return CallResult(
                scalar,
                Facts(),
                EffectSet(),
                DialectOperationSpec("columnar", "aggregate"),
                operation,
            )
        if operation in COLUMNAR_METHODS:
            result = _FRAME if is_frame else _SERIES
            return CallResult(
                result,
                Facts(),
                _ALLOC,
                DialectOperationSpec("columnar", COLUMNAR_METHODS[operation]),
                operation,
            )
        if operation in FRAME_ONLY and is_frame:
            return CallResult(
                _FRAME,
                Facts(),
                _ALLOC,
                DialectOperationSpec("columnar", FRAME_ONLY[operation]),
                operation,
            )
        if operation == "groupby" and is_frame:
            return CallResult(_GROUPED, Facts(), _ALLOC, FallbackSpec("grouping keys"), "grouping")
        if operation in {"merge", "join"} and is_frame:
            return CallResult(
                _FRAME, Facts(), _ALLOC, DialectOperationSpec("columnar", "join"), "join"
            )
        if operation == "to_numpy":
            return CallResult(
                _NDARRAY, Facts(), _ALLOC, FallbackSpec("a view or a copy"), "to_numpy"
            )
        return None
