"""PyArrow plugin: Arrow's representation, typed as Arrow (spec 52-56).

Arrays, chunked arrays, record batches, tables, schemas, and buffers are
typed by what they are, and the curated compute surface -- cast, filter,
take, sort, arithmetic, comparison, boolean, aggregation, null handling
-- is named as `columnar` operations shared with pandas, so one backend
serves both. Dataset and Parquet calls carry the IO effect they have. What
the plugin does not model runs PyArrow, which is never wrong.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, DialectOperationSpec, FallbackSpec, Plugin
from .numpy_plugin import _NDARRAY

__all__ = ["COMPUTE", "PyArrowPlugin"]

PLUGIN_VERSION = 1


def _instance(name: str) -> T.Instance:
    return T.Instance(f"pyarrow.{name}", (), (f"pyarrow.{name}", "object"))


_ARRAY = _instance("Array")
_CHUNKED = _instance("ChunkedArray")
_TABLE = _instance("Table")
_BATCH = _instance("RecordBatch")
_SCHEMA = _instance("Schema")
_FIELD = _instance("Field")
_DATATYPE = _instance("DataType")
_BUFFER = _instance("Buffer")
_SCALAR = _instance("Scalar")
_DATASET = _instance("Dataset")
_TYPES = {
    "pyarrow.Array": _ARRAY,
    "pyarrow.ChunkedArray": _CHUNKED,
    "pyarrow.Table": _TABLE,
    "pyarrow.RecordBatch": _BATCH,
    "pyarrow.Schema": _SCHEMA,
    "pyarrow.Field": _FIELD,
    "pyarrow.DataType": _DATATYPE,
    "pyarrow.Buffer": _BUFFER,
    "pyarrow.Scalar": _SCALAR,
    "pyarrow.Dataset": _DATASET,
}
_ALLOC = EffectSet.of(Effect.ALLOC, raises=("pyarrow.ArrowInvalid", "TypeError"))
_READ = EffectSet.of(Effect.IO, Effect.ALLOC, raises=("OSError", "pyarrow.ArrowInvalid"))
_WRITE = EffectSet.of(Effect.IO, raises=("OSError", "pyarrow.ArrowInvalid"))

#: `pyarrow.compute` functions, and the columnar operation each is.
COMPUTE: dict[str, str] = {
    "add": "add",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide",
    "negate": "negate",
    "abs": "abs",
    "equal": "equal",
    "not_equal": "not_equal",
    "less": "less",
    "less_equal": "less_equal",
    "greater": "greater",
    "greater_equal": "greater_equal",
    "and_": "and",
    "or_": "or",
    "invert": "invert",
    "xor": "xor",
    "cast": "cast",
    "fill_null": "fill_null",
    "is_null": "is_null",
    "is_valid": "is_valid",
    "drop_null": "filter",
    "filter": "filter",
    "take": "take",
    "sort_indices": "sort",
    "unique": "unique",
    "if_else": "select",
}
#: Reductions, and what they return.
REDUCTIONS: dict[str, T.Type] = {
    "sum": _SCALAR,
    "mean": _SCALAR,
    "min": _SCALAR,
    "max": _SCALAR,
    "min_max": _SCALAR,
    "count": _SCALAR,
    "count_distinct": _SCALAR,
    "any": _SCALAR,
    "all": _SCALAR,
    "stddev": _SCALAR,
    "variance": _SCALAR,
}
TYPE_CONSTRUCTORS = frozenset(
    {
        "null",
        "bool_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "string",
        "utf8",
        "large_string",
        "binary",
        "date32",
        "date64",
        "timestamp",
        "list_",
        "struct",
        "dictionary",
        "decimal128",
    }
)


def _is(t: T.Type, *names: str) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name in names


class PyArrowPlugin(Plugin):
    """Types Arrow's containers and names the curated compute as columnar operations."""

    name = "pyarrow"
    modules = ("pyarrow", "pyarrow.compute", "pyarrow.parquet", "pyarrow.dataset", "pyarrow.csv")

    def fingerprint(self) -> str:
        version = "absent"
        try:
            if importlib.util.find_spec("pyarrow") is not None:
                import pyarrow

                version = str(pyarrow.__version__)
        except Exception:  # noqa: BLE001 - a broken install must not break analysis
            version = "unknown"
        return f"v{PLUGIN_VERSION}:api{self.api_version}:pyarrow={version}"

    def external_types(self) -> dict[str, str]:
        return {name: name for name in _TYPES}

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        if qualname in _TYPES:
            return T.ClassObject(qualname, _TYPES[qualname]), Facts()
        return None

    def instance_attribute(
        self, type_name: str, attribute: str, facts: Facts | None = None
    ) -> tuple[T.Type, Facts] | None:
        if type_name in {"pyarrow.Array", "pyarrow.ChunkedArray"}:
            return self._array_attribute(type_name, attribute)
        if type_name in {"pyarrow.Table", "pyarrow.RecordBatch"}:
            return self._table_attribute(type_name, attribute)
        if type_name == "pyarrow.Schema":
            if attribute == "names":
                return T.list_of(T.STR), Facts()
            if attribute == "types":
                return T.list_of(_DATATYPE), Facts()
            if attribute == "field":
                return T.Callable_((), _FIELD, "pyarrow.Schema.field"), Facts()
        if type_name == "pyarrow.Scalar":
            if attribute == "as_py":
                return T.Callable_((), T.ANY, "pyarrow.Scalar.as_py"), Facts()
            if attribute == "is_valid":
                return T.BOOL, Facts()
        if type_name == "pyarrow.Buffer":
            if attribute in {"size", "address"}:
                return T.INT, Facts()
            if attribute == "to_pybytes":
                return T.Callable_((), T.BYTES, "pyarrow.Buffer.to_pybytes"), Facts()
        if type_name == "pyarrow.Dataset" and attribute in {"to_table", "head"}:
            return T.Callable_((), _TABLE, f"pyarrow.Dataset.{attribute}"), Facts()
        return None

    def _array_attribute(self, type_name: str, attribute: str) -> tuple[T.Type, Facts] | None:
        own = _TYPES[type_name]
        if attribute in {"null_count", "nbytes", "offset", "num_chunks"}:
            return T.INT, Facts()
        if attribute == "type":
            return _DATATYPE, Facts()
        if attribute == "buffers":
            return T.Callable_((), T.list_of(_BUFFER), f"{type_name}.buffers"), Facts()
        if attribute == "to_pylist":
            return T.Callable_((), T.list_of(T.ANY), f"{type_name}.to_pylist"), Facts()
        if attribute == "to_numpy":
            return T.Callable_((), _NDARRAY, f"{type_name}.to_numpy"), Facts()
        if attribute in {
            "slice",
            "filter",
            "take",
            "cast",
            "fill_null",
            "drop_null",
            "unique",
            "combine_chunks",
        }:
            result = _ARRAY if attribute in {"unique", "combine_chunks"} else own
            return T.Callable_((), result, f"{type_name}.{attribute}"), Facts()
        if attribute in {"is_null", "is_valid"}:
            return T.Callable_((), own, f"{type_name}.{attribute}"), Facts()
        if attribute == "equals":
            return T.Callable_((), T.BOOL, f"{type_name}.equals"), Facts()
        if attribute == "chunks":
            return T.list_of(_ARRAY), Facts()
        return None

    def _table_attribute(self, type_name: str, attribute: str) -> tuple[T.Type, Facts] | None:
        own = _TYPES[type_name]
        if attribute in {"num_rows", "num_columns", "nbytes"}:
            return T.INT, Facts()
        if attribute == "schema":
            return _SCHEMA, Facts()
        if attribute == "column_names":
            return T.list_of(T.STR), Facts()
        if attribute == "columns":
            return T.list_of(_CHUNKED if type_name == "pyarrow.Table" else _ARRAY), Facts()
        if attribute == "column":
            result = _CHUNKED if type_name == "pyarrow.Table" else _ARRAY
            return T.Callable_((), result, f"{type_name}.column"), Facts()
        if attribute in {
            "select",
            "filter",
            "take",
            "slice",
            "sort_by",
            "drop",
            "append_column",
            "rename_columns",
            "combine_chunks",
            "drop_null",
        }:
            return T.Callable_((), own, f"{type_name}.{attribute}"), Facts()
        if attribute in {"join", "group_by"}:
            return T.Callable_((), _TABLE, f"{type_name}.{attribute}"), Facts()
        if attribute == "to_pylist":
            return T.Callable_(
                (), T.list_of(T.dict_of(T.STR, T.ANY)), f"{type_name}.to_pylist"
            ), Facts()
        if attribute == "to_pandas":
            return T.Callable_((), T.ANY, f"{type_name}.to_pandas"), Facts()
        if attribute == "to_batches":
            return T.Callable_((), T.list_of(_BATCH), f"{type_name}.to_batches"), Facts()
        if attribute == "equals":
            return T.Callable_((), T.BOOL, f"{type_name}.equals"), Facts()
        return None

    def subscript(
        self, type_name: str, *, is_slice: bool, tupled: bool
    ) -> tuple[T.Type, Facts] | None:
        if type_name in {"pyarrow.Array", "pyarrow.ChunkedArray"}:
            return (_TYPES[type_name] if is_slice else _SCALAR), Facts()
        if type_name in {"pyarrow.Table", "pyarrow.RecordBatch"}:
            # A column by name or position; a slice of rows is a table.
            column = _CHUNKED if type_name == "pyarrow.Table" else _ARRAY
            return (_TYPES[type_name] if is_slice else column), Facts()
        return None

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        owner, _, operation = qualname.rpartition(".")
        if owner == "pyarrow":
            return self._constructor(operation, args)
        if owner == "pyarrow.compute":
            return self._compute(operation, args)
        if owner == "pyarrow.parquet":
            if operation == "read_table":
                return CallResult(
                    _TABLE, Facts(), _READ, FallbackSpec("reads a file"), "reads a file"
                )
            if operation == "write_table":
                return CallResult(
                    T.NONE, Facts(), _WRITE, FallbackSpec("writes a file"), "writes a file"
                )
        if owner == "pyarrow.dataset" and operation == "dataset":
            return CallResult(_DATASET, Facts(), _READ, FallbackSpec("opens files"), "opens files")
        if owner == "pyarrow.csv" and operation == "read_csv":
            return CallResult(_TABLE, Facts(), _READ, FallbackSpec("reads a file"), "reads a file")
        if owner in _TYPES:
            return self._method(owner, operation, args)
        return None

    def _constructor(
        self, operation: str, args: Sequence[tuple[T.Type, Facts]]
    ) -> CallResult | None:
        if operation == "array":
            return CallResult(_ARRAY, Facts(), _ALLOC, FallbackSpec("built by Arrow"), "an array")
        if operation == "chunked_array":
            return CallResult(
                _CHUNKED, Facts(), _ALLOC, FallbackSpec("built by Arrow"), "a chunked array"
            )
        if operation in {"table", "Table"}:
            return CallResult(_TABLE, Facts(), _ALLOC, FallbackSpec("built by Arrow"), "a table")
        if operation in {"record_batch", "RecordBatch"}:
            return CallResult(
                _BATCH, Facts(), _ALLOC, FallbackSpec("built by Arrow"), "a record batch"
            )
        if operation == "schema":
            return CallResult(_SCHEMA, Facts(), _ALLOC, FallbackSpec("built by Arrow"), "a schema")
        if operation == "field":
            return CallResult(_FIELD, Facts(), _ALLOC, FallbackSpec("built by Arrow"), "a field")
        if operation == "scalar":
            return CallResult(_SCALAR, Facts(), _ALLOC, FallbackSpec("built by Arrow"), "a scalar")
        if operation == "py_buffer":
            return CallResult(
                _BUFFER, Facts(), EffectSet(), FallbackSpec("wraps memory"), "a buffer"
            )
        if operation in TYPE_CONSTRUCTORS:
            return CallResult(
                _DATATYPE, Facts(), EffectSet(), FallbackSpec("a data type"), "a type"
            )
        if operation == "concat_tables":
            return CallResult(
                _TABLE, Facts(), _ALLOC, DialectOperationSpec("columnar", "concat"), "concat"
            )
        if operation == "concat_arrays":
            return CallResult(
                _ARRAY, Facts(), _ALLOC, DialectOperationSpec("columnar", "concat"), "concat"
            )
        return None

    def _compute(self, operation: str, args: Sequence[tuple[T.Type, Facts]]) -> CallResult | None:
        if operation in REDUCTIONS:
            return CallResult(
                REDUCTIONS[operation],
                Facts(),
                _ALLOC,
                DialectOperationSpec("columnar", "aggregate", (("function", operation),)),
                f"`{operation}` over an array",
            )
        if operation not in COMPUTE:
            return None
        chunked = any(_is(t, "pyarrow.ChunkedArray") for t, _f in args)
        table = any(_is(t, "pyarrow.Table", "pyarrow.RecordBatch") for t, _f in args)
        if operation in {"filter", "take", "drop_null"} and table:
            result: T.Type = _TABLE
        elif operation == "sort_indices":
            result = _ARRAY
        else:
            result = _CHUNKED if chunked else _ARRAY
        return CallResult(
            result, Facts(), _ALLOC, DialectOperationSpec("columnar", COMPUTE[operation]), operation
        )

    def _method(
        self, owner: str, operation: str, args: Sequence[tuple[T.Type, Facts]]
    ) -> CallResult | None:
        described = self.instance_attribute(owner, operation)
        if described is None:
            return None
        result_type, _facts = described
        if isinstance(result_type, T.Callable_):
            result_type = result_type.ret
        if operation in COMPUTE or operation in {
            "slice",
            "sort_by",
            "select",
            "drop",
            "join",
            "group_by",
            "combine_chunks",
        }:
            columnar = COMPUTE.get(operation, operation)
            return CallResult(
                result_type, Facts(), _ALLOC, DialectOperationSpec("columnar", columnar), operation
            )
        if operation == "to_numpy":
            return CallResult(
                _NDARRAY,
                Facts(),
                _ALLOC,
                FallbackSpec("a zero-copy view only for a fixed-width array without nulls"),
                "to_numpy",
            )
        return CallResult(result_type, Facts(), _ALLOC, FallbackSpec("Arrow runs it"), operation)
