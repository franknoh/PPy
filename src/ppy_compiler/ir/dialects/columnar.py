"""The columnar dialect: columns with nulls, and tables of them, as values.

`columnar.column<f64>` is a column of `f64` with no nulls; `columnar.column<f64,
nullable>` has a validity bitmap beside its values. A column's length is a
run-time fact, not part of its type: a `filter` keeps as many rows as its
mask allows. `columnar.table<a, column<i64>, b, column<f64, nullable>>` is a
table of named columns. The operations are the ones pandas and PyArrow
share (spec 51): arithmetic, comparison, and boolean logic over columns,
with a null where any input is null; `cast`, `is_null`, `is_valid`,
`fill_null`, `select`, `fill` (one scalar, `n` rows); `filter`, `take`,
`sort_indices`, `concat`;
`aggregate` to a one-row column; and over tables `make`, `column_of`,
`project`, `filter`, `take`, `concat`, `group_by`, and an inner `join`.
`lower-columnar` makes memory and loops of them, in Arrow's layout: a
values buffer and a bit-packed validity bitmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import BOOL, I64, U8, BoolType, BufferType, DialectType, FloatType, IntType, IRType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "AGGREGATES",
    "ARITHMETIC",
    "BOOLEAN",
    "COMPARISON",
    "ColumnInfo",
    "ColumnarDialect",
    "TableInfo",
    "column_type",
    "describe",
    "describe_table",
    "table_type",
    "values_element",
]

#: Two numeric columns in, one of the same element type out.
ARITHMETIC = ("add", "sub", "mul", "div")
#: Two columns in, a bool column out.
COMPARISON = ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal")
#: Two bool columns in, one out.
BOOLEAN = ("and", "or", "xor")
#: What `aggregate` folds a column to.
AGGREGATES = ("sum", "mean", "min", "max", "count", "any", "all")
#: What `group_by` folds each group's values with.
GROUP_AGGREGATES = ("sum", "count", "min", "max")


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    dtype: IRType
    nullable: bool


@dataclass(frozen=True, slots=True)
class TableInfo:
    names: tuple[str, ...]
    columns: tuple[ColumnInfo, ...]

    def index(self, name: str) -> int | None:
        return self.names.index(name) if name in self.names else None


def column_type(dtype: IRType, nullable: bool = False) -> DialectType:
    return DialectType("columnar", "column", (dtype, "nullable") if nullable else (dtype,))


def table_type(names: tuple[str, ...], columns: tuple[DialectType, ...]) -> DialectType:
    args: list[IRType | int | str] = []
    for name, column in zip(names, columns, strict=True):
        args.extend((name, column))
    return DialectType("columnar", "table", tuple(args))


def describe(t: IRType) -> ColumnInfo | None:
    """The `ColumnInfo` of a column type, or None for anything else."""
    if not isinstance(t, DialectType) or (t.dialect, t.name) != ("columnar", "column"):
        return None
    if _verify_column(t) is not None:
        return None
    dtype = t.args[0]
    assert isinstance(dtype, IRType)
    return ColumnInfo(dtype, len(t.args) == 2)


def describe_table(t: IRType) -> TableInfo | None:
    if not isinstance(t, DialectType) or (t.dialect, t.name) != ("columnar", "table"):
        return None
    if _verify_table(t) is not None:
        return None
    names = tuple(str(t.args[i]) for i in range(0, len(t.args), 2))
    columns = []
    for i in range(1, len(t.args), 2):
        info = describe(t.args[i])  # type: ignore[arg-type]
        assert info is not None
        columns.append(info)
    return TableInfo(names, tuple(columns))


def _element_ok(dtype: object) -> bool:
    return isinstance(dtype, (IntType, FloatType, BoolType))


def _verify_column(t: DialectType) -> str | None:
    if not t.args or len(t.args) > 2 or not _element_ok(t.args[0]):
        return "a column is columnar.column<T> or columnar.column<T, nullable> with a scalar T"
    if len(t.args) == 2 and t.args[1] != "nullable":
        return f"a column's second argument is `nullable`, not {t.args[1]!r}"
    return None


def _verify_table(t: DialectType) -> str | None:
    if len(t.args) % 2 != 0:
        return "a table is columnar.table<name, column, name, column, ...>"
    names: set[str] = set()
    for i in range(0, len(t.args), 2):
        name = t.args[i]
        column = t.args[i + 1]
        if not isinstance(name, str) or isinstance(name, IRType):
            return f"a table's column is named by a word, not {name!r}"
        if name in names:
            return f"a table names column {name!r} twice"
        names.add(name)
        if not isinstance(column, DialectType) or column.name != "column":
            return f"{name}'s type is a column, not {column}"
        reason = _verify_column(column)
        if reason is not None:
            return reason
    return None


def verify_type(t: DialectType) -> str | None:
    if t.name == "column":
        return _verify_column(t)
    if t.name == "table":
        return _verify_table(t)
    return f"columnar defines no type {t.name!r}"


# -- verifiers ----------------------------------------------------------------------------------


def _column(op: Operation, checker: Checker, value: Value, what: str) -> ColumnInfo | None:
    info = describe(value.type)
    if info is None:
        checker.error(op, f"{what} of {op.name} is a column, not {value.type}")
    return info


def _table(op: Operation, checker: Checker, value: Value, what: str) -> TableInfo | None:
    info = describe_table(value.type)
    if info is None:
        checker.error(op, f"{what} of {op.name} is a table, not {value.type}")
    return info


def _result(op: Operation, checker: Checker, expected: DialectType) -> None:
    if op.results[0].type != expected:
        checker.error(
            op, f"{op.name} gives {expected}, the result is written as {op.results[0].type}"
        )


def values_element(dtype: IRType) -> IRType:
    """What a column's values buffer holds: bits packed into bytes for bool."""
    return U8 if isinstance(dtype, BoolType) else dtype


def _verify_from_parts(op: Operation, checker: Checker) -> None:
    info = describe(op.results[0].type)
    if info is None:
        checker.error(op, f"columnar.from_parts gives a column, not {op.results[0].type}")
        return
    wanted = 3 if info.nullable else 2
    if len(op.operands) != wanted:
        checker.error(
            op,
            "columnar.from_parts takes the values buffer"
            + (", the validity bitmap," if info.nullable else "")
            + f" and the length of a {op.results[0].type}",
        )
        return
    values = op.operands[0].type
    element = values_element(info.dtype)
    if not isinstance(values, BufferType) or values.element != element:
        checker.error(op, f"the values of a column of {info.dtype} are a buffer<{element}>")
    if info.nullable and not _is_bitmap(op.operands[1].type):
        checker.error(op, "a validity bitmap is a buffer<u8>, one bit per row")
    if op.operands[-1].type != I64:
        checker.error(op, "a column's length is an i64")


def _is_bitmap(t: IRType) -> bool:
    return isinstance(t, BufferType) and isinstance(t.element, IntType) and t.element.width == 8


def _verify_store(op: Operation, checker: Checker) -> None:
    info = _column(op, checker, op.operands[0], "the column")
    if info is None:
        return
    wanted = 3 if info.nullable else 2
    if len(op.operands) != wanted:
        checker.error(
            op,
            "columnar.store takes the column, its values buffer"
            + (", and its validity bitmap" if info.nullable else ""),
        )
        return
    values = op.operands[1].type
    element = values_element(info.dtype)
    if not isinstance(values, BufferType) or values.element != element:
        checker.error(op, f"the values of a column of {info.dtype} go to a buffer<{element}>")
    if info.nullable and not _is_bitmap(op.operands[2].type):
        checker.error(op, "a validity bitmap is a buffer<u8>, one bit per row")


def _verify_fill(op: Operation, checker: Checker) -> None:
    info = describe(op.results[0].type)
    if info is None or info.nullable:
        checker.error(op, f"columnar.fill gives a column without nulls, not {op.results[0].type}")
        return
    if op.operands[0].type != info.dtype:
        checker.error(op, f"columnar.fill of a {op.operands[0].type} gives a column of it")
    if op.operands[1].type != I64:
        checker.error(op, "columnar.fill takes the row count as an i64")


def _verify_length(op: Operation, checker: Checker) -> None:
    if describe(op.operands[0].type) is None and describe_table(op.operands[0].type) is None:
        checker.error(op, f"columnar.length takes a column or a table, not {op.operands[0].type}")
    if op.results[0].type != I64:
        checker.error(op, "columnar.length is an i64")


def _verify_binary(op: Operation, checker: Checker) -> None:
    a = _column(op, checker, op.operands[0], "the left operand")
    c = _column(op, checker, op.operands[1], "the right operand")
    if a is None or c is None:
        return
    if a.dtype != c.dtype:
        checker.error(
            op, f"{op.name} takes columns of one element type, not {a.dtype} and {c.dtype}"
        )
        return
    nullable = a.nullable or c.nullable
    if op.local_name in ARITHMETIC:
        if isinstance(a.dtype, BoolType):
            checker.error(op, f"{op.name} takes numeric columns")
            return
        if op.local_name == "div" and not isinstance(a.dtype, FloatType):
            checker.error(op, "columnar.div takes floating-point columns")
            return
        _result(op, checker, column_type(a.dtype, nullable))
    elif op.local_name in COMPARISON:
        _result(op, checker, column_type(BOOL, nullable))
    else:
        if not isinstance(a.dtype, BoolType):
            checker.error(op, f"{op.name} takes bool columns")
            return
        _result(op, checker, column_type(BOOL, nullable))


def _verify_unary(op: Operation, checker: Checker) -> None:
    a = _column(op, checker, op.operands[0], "the operand")
    if a is None:
        return
    if op.local_name == "invert" and not isinstance(a.dtype, BoolType):
        checker.error(op, "columnar.invert takes a bool column")
        return
    if op.local_name in {"negate", "abs"} and isinstance(a.dtype, BoolType):
        checker.error(op, f"{op.name} takes a numeric column")
        return
    _result(op, checker, column_type(a.dtype, a.nullable))


def _verify_null_test(op: Operation, checker: Checker) -> None:
    if _column(op, checker, op.operands[0], "the operand") is None:
        return
    _result(op, checker, column_type(BOOL, False))


def _verify_fill_null(op: Operation, checker: Checker) -> None:
    a = _column(op, checker, op.operands[0], "the column")
    if a is None:
        return
    if op.operands[1].type != a.dtype:
        checker.error(op, f"columnar.fill_null fills a column of {a.dtype} with a {a.dtype}")
        return
    _result(op, checker, column_type(a.dtype, False))


def _verify_cast(op: Operation, checker: Checker) -> None:
    a = _column(op, checker, op.operands[0], "the column")
    result = describe(op.results[0].type)
    if a is None:
        return
    if result is None:
        checker.error(op, f"columnar.cast gives a column, not {op.results[0].type}")
        return
    if result.nullable != a.nullable:
        checker.error(op, "columnar.cast keeps a column's nullability")


def _verify_select(op: Operation, checker: Checker) -> None:
    mask = _column(op, checker, op.operands[0], "the mask")
    a = _column(op, checker, op.operands[1], "the left choice")
    c = _column(op, checker, op.operands[2], "the right choice")
    if mask is None or a is None or c is None:
        return
    if not isinstance(mask.dtype, BoolType):
        checker.error(op, "columnar.select chooses by a bool column")
        return
    if a.dtype != c.dtype:
        checker.error(op, f"columnar.select chooses between {a.dtype} and {c.dtype}; make them one")
        return
    _result(op, checker, column_type(a.dtype, mask.nullable or a.nullable or c.nullable))


def _verify_filter(op: Operation, checker: Checker) -> None:
    mask = _column(op, checker, op.operands[1], "the mask")
    if mask is None:
        return
    if not isinstance(mask.dtype, BoolType):
        checker.error(op, "columnar.filter keeps the rows a bool column marks")
        return
    source = op.operands[0].type
    if describe(source) is None and describe_table(source) is None:
        checker.error(op, f"columnar.filter takes a column or a table, not {source}")
        return
    _result(op, checker, source)  # type: ignore[arg-type]


def _verify_take(op: Operation, checker: Checker) -> None:
    indices = _column(op, checker, op.operands[1], "the indices")
    if indices is None:
        return
    if not isinstance(indices.dtype, IntType):
        checker.error(op, "columnar.take takes an integer column of row positions")
        return
    source = op.operands[0].type
    column = describe(source)
    table = describe_table(source)
    if column is None and table is None:
        checker.error(op, f"columnar.take takes a column or a table, not {source}")
        return
    if not indices.nullable:
        _result(op, checker, source)  # type: ignore[arg-type]
        return
    # A null position takes a null: every column of the result is nullable.
    if column is not None:
        _result(op, checker, column_type(column.dtype, True))
    else:
        assert table is not None
        _result(
            op,
            checker,
            table_type(table.names, tuple(column_type(c.dtype, True) for c in table.columns)),
        )


def _verify_concat(op: Operation, checker: Checker) -> None:
    if not op.operands:
        checker.error(op, "columnar.concat takes at least one operand")
        return
    first = op.operands[0].type
    for operand in op.operands[1:]:
        if operand.type != first:
            checker.error(
                op, f"columnar.concat takes operands of one type, not {first} and {operand.type}"
            )
            return
    if describe(first) is None and describe_table(first) is None:
        checker.error(op, f"columnar.concat takes columns or tables, not {first}")
        return
    _result(op, checker, first)  # type: ignore[arg-type]


def _verify_sort_indices(op: Operation, checker: Checker) -> None:
    if _column(op, checker, op.operands[0], "the column") is None:
        return
    _result(op, checker, column_type(I64, False))


def _verify_aggregate(op: Operation, checker: Checker) -> None:
    a = _column(op, checker, op.operands[0], "the column")
    function = op.attributes.get("function")
    if function not in AGGREGATES:
        checker.error(
            op, f"aggregate `function` is one of {', '.join(AGGREGATES)}, not {function!r}"
        )
        return
    if a is None:
        return
    if function in {"any", "all"}:
        if not isinstance(a.dtype, BoolType):
            checker.error(op, f"columnar.aggregate {function} takes a bool column")
            return
        expected = column_type(BOOL, True)
    elif function == "count":
        expected = column_type(I64, False)
    elif function == "mean":
        if not isinstance(a.dtype, FloatType):
            checker.error(op, "columnar.aggregate mean takes a floating-point column")
            return
        expected = column_type(a.dtype, True)
    else:
        if isinstance(a.dtype, BoolType):
            checker.error(op, f"columnar.aggregate {function} takes a numeric column")
            return
        expected = column_type(a.dtype, True)
    _result(op, checker, expected)


def _verify_make(op: Operation, checker: Checker) -> None:
    names = op.attributes.get("names")
    if not isinstance(names, tuple) or len(names) != len(op.operands):
        checker.error(op, "columnar.make names each of its columns")
        return
    columns = []
    for operand in op.operands:
        info = _column(op, checker, operand, "a column")
        if info is None:
            return
        columns.append(column_type(info.dtype, info.nullable))
    _result(op, checker, table_type(tuple(str(n) for n in names), tuple(columns)))


def _verify_column_of(op: Operation, checker: Checker) -> None:
    table = _table(op, checker, op.operands[0], "the table")
    if table is None:
        return
    name = str(op.attributes.get("name"))
    index = table.index(name)
    if index is None:
        checker.error(op, f"the table has no column {name!r}; it has {', '.join(table.names)}")
        return
    info = table.columns[index]
    _result(op, checker, column_type(info.dtype, info.nullable))


def _verify_project(op: Operation, checker: Checker) -> None:
    table = _table(op, checker, op.operands[0], "the table")
    names = op.attributes.get("names")
    if table is None:
        return
    if not isinstance(names, tuple):
        checker.error(op, "columnar.project names the columns it keeps")
        return
    kept = []
    for name in names:
        index = table.index(str(name))
        if index is None:
            checker.error(op, f"the table has no column {name!r}; it has {', '.join(table.names)}")
            return
        info = table.columns[index]
        kept.append(column_type(info.dtype, info.nullable))
    _result(op, checker, table_type(tuple(str(n) for n in names), tuple(kept)))


def _verify_group_by(op: Operation, checker: Checker) -> None:
    table = _table(op, checker, op.operands[0], "the table")
    key = op.attributes.get("key")
    aggregates = op.attributes.get("aggregates")
    if table is None:
        return
    key_index = table.index(str(key))
    if key_index is None:
        checker.error(op, f"the table has no key column {key!r}")
        return
    key_info = table.columns[key_index]
    if isinstance(key_info.dtype, FloatType) or key_info.nullable:
        checker.error(op, "columnar.group_by groups by an integer or bool column without nulls")
        return
    if not isinstance(aggregates, tuple) or not aggregates:
        checker.error(op, "columnar.group_by names (column, function) pairs to aggregate")
        return
    names = [str(key)]
    columns = [column_type(key_info.dtype, False)]
    for entry in aggregates:
        if not isinstance(entry, tuple) or len(entry) != 2:
            checker.error(op, "each aggregate of columnar.group_by is (column, function)")
            return
        name, function = entry
        index = table.index(str(name))
        if index is None:
            checker.error(op, f"the table has no column {name!r} to aggregate")
            return
        if function not in GROUP_AGGREGATES:
            checker.error(
                op, f"a group aggregate is one of {', '.join(GROUP_AGGREGATES)}, not {function!r}"
            )
            return
        info = table.columns[index]
        if function == "count":
            columns.append(column_type(I64, False))
        else:
            if isinstance(info.dtype, BoolType):
                checker.error(op, f"{function} of a bool column has no meaning")
                return
            columns.append(column_type(info.dtype, True))
        names.append(f"{name}_{function}")
    _result(op, checker, table_type(tuple(names), tuple(columns)))


def _verify_join(op: Operation, checker: Checker) -> None:
    left = _table(op, checker, op.operands[0], "the left table")
    right = _table(op, checker, op.operands[1], "the right table")
    key = str(op.attributes.get("key"))
    if left is None or right is None:
        return
    li, ri = left.index(key), right.index(key)
    if li is None or ri is None:
        checker.error(op, f"both tables of columnar.join name the key column {key!r}")
        return
    lk, rk = left.columns[li], right.columns[ri]
    if lk.dtype != rk.dtype or not isinstance(lk.dtype, IntType) or lk.nullable or rk.nullable:
        checker.error(op, "columnar.join joins on one integer key column without nulls")
        return
    names = list(left.names)
    columns = [column_type(c.dtype, c.nullable) for c in left.columns]
    for name, info in zip(right.names, right.columns, strict=True):
        if name == key:
            continue
        if name in names:
            checker.error(op, f"both tables have a column {name!r}; project one away first")
            return
        names.append(name)
        columns.append(column_type(info.dtype, info.nullable))
    _result(op, checker, table_type(tuple(names), tuple(columns)))


class ColumnarDialect(Dialect):
    name = "columnar"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(OpSpec("columnar.from_parts", verify=_verify_from_parts, results=1))
        add(OpSpec("columnar.store", verify=_verify_store, results=0))
        add(OpSpec("columnar.length", pure=True, verify=_verify_length, operands=1, results=1))
        add(OpSpec("columnar.fill", pure=True, verify=_verify_fill, operands=2, results=1))
        for name in (*ARITHMETIC, *COMPARISON, *BOOLEAN):
            add(OpSpec(f"columnar.{name}", pure=True, verify=_verify_binary, operands=2, results=1))
        for name in ("negate", "abs", "invert"):
            add(OpSpec(f"columnar.{name}", pure=True, verify=_verify_unary, operands=1, results=1))
        for name in ("is_null", "is_valid"):
            add(
                OpSpec(
                    f"columnar.{name}", pure=True, verify=_verify_null_test, operands=1, results=1
                )
            )
        add(
            OpSpec("columnar.fill_null", pure=True, verify=_verify_fill_null, operands=2, results=1)
        )
        add(OpSpec("columnar.cast", pure=True, verify=_verify_cast, operands=1, results=1))
        add(OpSpec("columnar.select", pure=True, verify=_verify_select, operands=3, results=1))
        add(OpSpec("columnar.filter", pure=True, verify=_verify_filter, operands=2, results=1))
        add(OpSpec("columnar.take", pure=True, verify=_verify_take, operands=2, results=1))
        add(OpSpec("columnar.concat", pure=True, verify=_verify_concat, results=1))
        add(
            OpSpec(
                "columnar.sort_indices",
                pure=True,
                verify=_verify_sort_indices,
                operands=1,
                results=1,
            )
        )
        add(
            OpSpec(
                "columnar.aggregate",
                pure=True,
                verify=_verify_aggregate,
                operands=1,
                results=1,
                required_attributes=("function",),
            )
        )
        add(
            OpSpec(
                "columnar.make",
                pure=True,
                verify=_verify_make,
                results=1,
                required_attributes=("names",),
            )
        )
        add(
            OpSpec(
                "columnar.column_of",
                pure=True,
                verify=_verify_column_of,
                operands=1,
                results=1,
                required_attributes=("name",),
            )
        )
        add(
            OpSpec(
                "columnar.project",
                pure=True,
                verify=_verify_project,
                operands=1,
                results=1,
                required_attributes=("names",),
            )
        )
        add(
            OpSpec(
                "columnar.group_by",
                pure=True,
                verify=_verify_group_by,
                operands=1,
                results=1,
                required_attributes=("key", "aggregates"),
            )
        )
        add(
            OpSpec(
                "columnar.join",
                pure=True,
                verify=_verify_join,
                operands=2,
                results=1,
                required_attributes=("key",),
            )
        )

    def verify_type(self, t: DialectType) -> str | None:
        return verify_type(t)


# -- builders -----------------------------------------------------------------------------------


def _typed(
    b: Builder,
    name: str,
    operands: tuple[Value, ...],
    result: DialectType,
    attributes=None,
    hint=None,
):  # type: ignore[no-untyped-def]
    return b.create(
        f"columnar.{name}", operands, (result,), attributes or {}, result_names=(hint,)
    ).result


def from_parts(
    b: Builder,
    values: Value,
    validity: Value | None,
    length: Value,
    dtype: IRType,
    name: str | None = None,
) -> Value:
    """A column of `length` rows over `values`; nullable when a validity bitmap comes with it."""
    operands = (values, length) if validity is None else (values, validity, length)
    return _typed(b, "from_parts", operands, column_type(dtype, validity is not None), hint=name)


def store(b: Builder, column: Value, values: Value, validity: Value | None = None) -> Operation:
    operands = (column, values) if validity is None else (column, values, validity)
    return b.create("columnar.store", operands, ())


def fill(b: Builder, scalar: Value, rows: Value, name: str | None = None) -> Value:
    """`rows` rows of `scalar`, with no nulls."""
    return _typed(b, "fill", (scalar, rows), column_type(scalar.type), hint=name)


def length(b: Builder, value: Value, name: str | None = None) -> Value:
    return b.create("columnar.length", (value,), (I64,), result_names=(name,)).result


def binary(b: Builder, op: str, a: Value, c: Value, name: str | None = None) -> Value:
    left, right = describe(a.type), describe(c.type)
    assert left is not None and right is not None
    nullable = left.nullable or right.nullable
    dtype = BOOL if op in COMPARISON or op in BOOLEAN else left.dtype
    return _typed(b, op, (a, c), column_type(dtype, nullable), hint=name)


def unary(b: Builder, op: str, a: Value, name: str | None = None) -> Value:
    info = describe(a.type)
    assert info is not None
    return _typed(b, op, (a,), column_type(info.dtype, info.nullable), hint=name)


def is_null(b: Builder, a: Value, name: str | None = None) -> Value:
    return _typed(b, "is_null", (a,), column_type(BOOL), hint=name)


def is_valid(b: Builder, a: Value, name: str | None = None) -> Value:
    return _typed(b, "is_valid", (a,), column_type(BOOL), hint=name)


def fill_null(b: Builder, a: Value, scalar: Value, name: str | None = None) -> Value:
    info = describe(a.type)
    assert info is not None
    return _typed(b, "fill_null", (a, scalar), column_type(info.dtype), hint=name)


def cast(b: Builder, a: Value, dtype: IRType, name: str | None = None) -> Value:
    info = describe(a.type)
    assert info is not None
    return _typed(b, "cast", (a,), column_type(dtype, info.nullable), hint=name)


def select(b: Builder, mask: Value, a: Value, c: Value, name: str | None = None) -> Value:
    m, left, right = describe(mask.type), describe(a.type), describe(c.type)
    assert m is not None and left is not None and right is not None
    nullable = m.nullable or left.nullable or right.nullable
    return _typed(b, "select", (mask, a, c), column_type(left.dtype, nullable), hint=name)


def filter_(b: Builder, source: Value, mask: Value, name: str | None = None) -> Value:
    return _typed(b, "filter", (source, mask), source.type, hint=name)  # type: ignore[arg-type]


def take(b: Builder, source: Value, indices: Value, name: str | None = None) -> Value:
    positions = describe(indices.type)
    assert positions is not None
    result: DialectType = source.type  # type: ignore[assignment]
    if positions.nullable:
        column = describe(source.type)
        if column is not None:
            result = column_type(column.dtype, True)
        else:
            table = describe_table(source.type)
            assert table is not None
            result = table_type(
                table.names, tuple(column_type(c.dtype, True) for c in table.columns)
            )
    return _typed(b, "take", (source, indices), result, hint=name)


def concat(b: Builder, parts: tuple[Value, ...], name: str | None = None) -> Value:
    return _typed(b, "concat", parts, parts[0].type, hint=name)  # type: ignore[arg-type]


def sort_indices(b: Builder, a: Value, name: str | None = None) -> Value:
    return _typed(b, "sort_indices", (a,), column_type(I64), hint=name)


def aggregate(b: Builder, function: str, a: Value, name: str | None = None) -> Value:
    info = describe(a.type)
    assert info is not None
    if function == "count":
        result = column_type(I64)
    elif function in {"any", "all"}:
        result = column_type(BOOL, True)
    else:
        result = column_type(info.dtype, True)
    return _typed(b, "aggregate", (a,), result, {"function": function}, hint=name)


def make(
    b: Builder, names: tuple[str, ...], columns: tuple[Value, ...], name: str | None = None
) -> Value:
    infos = [describe(c.type) for c in columns]
    assert all(i is not None for i in infos)
    result = table_type(names, tuple(column_type(i.dtype, i.nullable) for i in infos))  # type: ignore[union-attr]
    return _typed(b, "make", columns, result, {"names": names}, hint=name)


def column_of(b: Builder, table: Value, name: str, hint: str | None = None) -> Value:
    info = describe_table(table.type)
    assert info is not None
    index = info.index(name)
    assert index is not None
    column = info.columns[index]
    return _typed(
        b,
        "column_of",
        (table,),
        column_type(column.dtype, column.nullable),
        {"name": name},
        hint=hint,
    )


def project(b: Builder, table: Value, names: tuple[str, ...], hint: str | None = None) -> Value:
    info = describe_table(table.type)
    assert info is not None
    kept = []
    for name in names:
        index = info.index(name)
        assert index is not None
        kept.append(column_type(info.columns[index].dtype, info.columns[index].nullable))
    return _typed(
        b, "project", (table,), table_type(names, tuple(kept)), {"names": names}, hint=hint
    )


def group_by(
    b: Builder,
    table: Value,
    key: str,
    aggregates: tuple[tuple[str, str], ...],
    hint: str | None = None,
) -> Value:
    info = describe_table(table.type)
    assert info is not None
    key_index = info.index(key)
    assert key_index is not None
    names = [key]
    columns = [column_type(info.columns[key_index].dtype)]
    for column, function in aggregates:
        index = info.index(column)
        assert index is not None
        names.append(f"{column}_{function}")
        columns.append(
            column_type(I64)
            if function == "count"
            else column_type(info.columns[index].dtype, True)
        )
    return _typed(
        b,
        "group_by",
        (table,),
        table_type(tuple(names), tuple(columns)),
        {"key": key, "aggregates": aggregates},
        hint=hint,
    )


def join(b: Builder, left: Value, right: Value, key: str, hint: str | None = None) -> Value:
    left_info, right_info = describe_table(left.type), describe_table(right.type)
    assert left_info is not None and right_info is not None
    names = list(left_info.names)
    columns = [column_type(c.dtype, c.nullable) for c in left_info.columns]
    for name, info in zip(right_info.names, right_info.columns, strict=True):
        if name != key:
            names.append(name)
            columns.append(column_type(info.dtype, info.nullable))
    return _typed(
        b, "join", (left, right), table_type(tuple(names), tuple(columns)), {"key": key}, hint=hint
    )
