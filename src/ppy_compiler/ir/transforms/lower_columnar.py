"""Columns and Arrow arrays as memory, columnar operations as loops.

A column is a values buffer -- one element per row, or one bit for `bool`
-- a validity bitmap when it is nullable, and a length; that is Arrow's
layout, so an Arrow array is a column without a copy and a column is
written out as an Arrow array. Each operation becomes core loops: a null
in reads a null out, `filter` counts then copies, `take` checks its
positions, `sort_indices` is a stable merge sort with nulls last,
`aggregate` folds the valid rows, `group_by` sorts by the key and folds each
run, `join` is a sort-merge of two key columns, `map` evaluates a whole
expression per row in one loop with nothing materialized between the
operations. The `_FunctionLowering` of
`lower-tensor` mixes this in, so one pass makes memory of every dialect
that needs it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..dialects import arrow, columnar, core
from ..dialects import tensor as tensors
from ..model import Builder, Operation, Successor, Value
from ..types import BOOL, F64, I32, I64, U8, BoolType, BufferType, FloatType, IRType, PtrType

__all__ = ["ColumnarLowering"]


@dataclass(slots=True)
class _Column:
    """A column as memory: values (bits, for bool), a validity bitmap or none, the length."""

    info: columnar.ColumnInfo
    values: Value
    validity: Value | None
    length: Value
    #: One stored row read at every position: a `fill`.
    constant: bool = False


@dataclass(slots=True)
class _Table:
    info: columnar.TableInfo
    columns: list[_Column]


@dataclass(slots=True)
class _Arrow:
    """An Arrow array as its struct says: counts and the two buffer pointers."""

    dtype: IRType
    length: Value
    null_count: Value
    offset: Value
    validity: Value
    validity_address: Value
    values: Value


_PREDICATES = {
    "equal": "eq",
    "not_equal": "ne",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
}


class ColumnarLowering:
    """The `op_columnar_*` and `op_arrow_*` handlers; mixed into the tensor lowering."""

    # Provided by the host lowering.
    module: object
    function: object
    columns: dict[int, _Column]
    tables: dict[int, _Table]
    arrays: dict[int, _Arrow]

    def _setup_columnar(self) -> None:
        self.columns = {}
        self.tables = {}
        self.arrays = {}

    # -- bits -----------------------------------------------------------------------------

    def _i64(self, b: Builder, value: int) -> Value:
        return core.const(b, value, I64)

    def _u8(self, b: Builder, value: int) -> Value:
        return core.const(b, value, U8)

    def _bitwise(self, b: Builder, name: str, x: Value, y: Value) -> Value:
        return b.create(f"core.{name}", (x, y), (x.type,)).result

    def _bytes_for(self, b: Builder, rows: Value) -> Value:
        """How many bytes hold `rows` bits."""
        return self._bitwise(
            b, "shr", core.add(b, rows, self._i64(b, 7), overflow="wrap"), self._i64(b, 3)
        )

    def _bit(self, b: Builder, bitmap: Value, i: Value) -> Value:
        """Bit `i` of `bitmap`, as a bool."""
        byte = core.buffer_load(b, bitmap, self._bitwise(b, "shr", i, self._i64(b, 3)))
        shift = core.cast(b, self._bitwise(b, "and", i, self._i64(b, 7)), U8)
        bit = self._bitwise(b, "and", self._bitwise(b, "shr", byte, shift), self._u8(b, 1))
        return core.cmp(b, "ne", bit, self._u8(b, 0))

    def _write_bit(self, b: Builder, bitmap: Value, i: Value, flag: Value) -> None:
        """Set or clear bit `i` of `bitmap` after `flag`."""
        position = self._bitwise(b, "shr", i, self._i64(b, 3))
        byte = core.buffer_load(b, bitmap, position)
        shift = core.cast(b, self._bitwise(b, "and", i, self._i64(b, 7)), U8)
        mask = self._bitwise(b, "shl", self._u8(b, 1), shift)
        set_ = self._bitwise(b, "or", byte, mask)
        cleared = self._bitwise(b, "and", byte, self._bitwise(b, "xor", mask, self._u8(b, 0xFF)))
        core.buffer_store(b, core.select(b, flag, set_, cleared), bitmap, position)

    def _zeroed(self, b: Builder, dtype: IRType, count: Value, hint: str) -> Value:
        """A heap buffer of `count` elements of `dtype`, every byte zero."""
        buffer = self.allocate_dynamic(b, dtype, count, hint)  # type: ignore[attr-defined]
        width = core.const(b, max(dtype.width // 8, 1) if hasattr(dtype, "width") else 1, I64)  # type: ignore[attr-defined]
        core.call_extern(
            b,
            "memset",
            (
                core.cast(b, core.buffer_data(b, buffer), PtrType(U8)),
                core.const(b, 0, I32),
                core.mul(b, count, width, overflow="wrap"),
            ),
            (PtrType(U8),),
        )
        return buffer

    def _bitmap(self, b: Builder, rows: Value, hint: str) -> Value:
        return self._zeroed(b, U8, self._bytes_for(b, rows), hint)

    # -- columns as memory ------------------------------------------------------------------

    def column(self, value: Value) -> _Column:
        found = self.columns.get(id(value))
        if found is None:
            raise self._error(f"column %{value.name or '?'} was never made")
        return found

    def table(self, value: Value) -> _Table:
        found = self.tables.get(id(value))
        if found is None:
            raise self._error(f"table %{value.name or '?'} was never made")
        return found

    def _error(self, message: str):  # type: ignore[no-untyped-def]
        from .lower_tensor import LoweringError

        return LoweringError(f"@{self.function.name}: {message}")  # type: ignore[attr-defined]

    def _new_column(
        self, b: Builder, dtype: IRType, nullable: bool, rows: Value, hint: str, zero: bool = False
    ) -> _Column:
        """Fresh memory for `rows` of `dtype`; bitmaps start at zero, values on request."""
        if isinstance(dtype, BoolType):
            values = self._bitmap(b, rows, f"{hint}.bits")
        elif zero:
            values = self._zeroed(b, dtype, rows, hint)
        else:
            values = self.allocate_dynamic(b, dtype, rows, hint)  # type: ignore[attr-defined]
        validity = self._bitmap(b, rows, f"{hint}.valid") if nullable else None
        return _Column(columnar.ColumnInfo(dtype, nullable), values, validity, rows)

    def _get(self, b: Builder, column: _Column, i: Value) -> tuple[Value, Value]:
        """Row `i` of `column`: its value and whether it is valid."""
        if column.constant:
            i = self._i64(b, 0)
        if isinstance(column.info.dtype, BoolType):
            value = self._bit(b, column.values, i)
        else:
            value = core.buffer_load(b, column.values, i)
        valid = (
            self._bit(b, column.validity, i)
            if column.validity is not None
            else core.const(b, True, BOOL)
        )
        return value, valid

    def _put(self, b: Builder, column: _Column, i: Value, value: Value, valid: Value) -> None:
        if isinstance(column.info.dtype, BoolType):
            self._write_bit(b, column.values, i, value)
        else:
            core.buffer_store(b, value, column.values, i)
        if column.validity is not None:
            self._write_bit(b, column.validity, i, valid)

    def _zero_of(self, b: Builder, dtype: IRType) -> Value:
        if isinstance(dtype, BoolType):
            return core.const(b, False, BOOL)
        return core.const(b, 0.0 if isinstance(dtype, FloatType) else 0, dtype)

    def _same_length(self, b: Builder, a: _Column, c: _Column) -> Value:
        if a.constant:
            return c.length
        if c.constant:
            return a.length
        if a.length is not c.length:
            equal = core.cmp(b, "eq", a.length, c.length)
            core.guard(b, equal, "contract", "the columns differ in length")
        return a.length

    def _branch(self, op: Operation, condition: Value, hint: str) -> tuple[Builder, Builder]:
        """Split at `op`: a `then` and an `else` builder, both flowing on to `op`."""
        block = op.parent
        assert block is not None
        continuation = self.function.body.add_block(self.fresh(f"{hint}.next"))  # type: ignore[attr-defined]
        index = block.operations.index(op)
        for later in list(block.operations[index:]):
            block.remove(later)
            continuation.append(later)
        then_block = self.function.body.add_block(self.fresh(f"{hint}.then"))  # type: ignore[attr-defined]
        else_block = self.function.body.add_block(self.fresh(f"{hint}.else"))  # type: ignore[attr-defined]
        core.cond_br(Builder(block), condition, Successor(then_block), Successor(else_block))
        then_leave = core.br(Builder(then_block), Successor(continuation))
        else_leave = core.br(Builder(else_block), Successor(continuation))
        return Builder().before(then_leave), Builder().before(else_leave)

    def _when(self, inner: Builder, condition: Value, hint: str) -> tuple[Builder, Builder]:
        """Inside a loop body: a `then` builder and the builder that continues after it."""
        anchor = inner.anchor
        assert anchor is not None
        then_block = self.function.body.add_block(self.fresh(f"{hint}.then"))  # type: ignore[attr-defined]
        after = self.function.body.add_block(self.fresh(f"{hint}.after"))  # type: ignore[attr-defined]
        core.cond_br(inner, condition, Successor(then_block), Successor(after))
        self._continue_in(inner, after)  # type: ignore[attr-defined]
        leave = core.br(Builder(then_block), Successor(after))
        return Builder().before(leave), Builder().before(anchor)

    # -- from parts, store, length ------------------------------------------------------------

    def op_columnar_from_parts(self, op: Operation) -> None:
        info = columnar.describe(op.result.type)
        assert info is not None
        b = Builder().before(op)
        values = op.operands[0]
        length = op.operands[-1]
        validity = op.operands[1] if info.nullable else None
        if isinstance(info.dtype, BoolType):
            enough = core.cmp(b, "ge", self.length(b, values), self._bytes_for(b, length))  # type: ignore[attr-defined]
        else:
            enough = core.cmp(b, "ge", self.length(b, values), length)  # type: ignore[attr-defined]
        core.guard(b, enough, "bounds", "the values buffer holds fewer rows than the column")
        if validity is not None:
            wide = core.cmp(b, "ge", self.length(b, validity), self._bytes_for(b, length))  # type: ignore[attr-defined]
            core.guard(b, wide, "bounds", "the validity bitmap holds fewer bits than the column")
        self.columns[id(op.result)] = _Column(info, values, validity, length)

    def op_columnar_store(self, op: Operation) -> None:
        source = self.column(op.operands[0])
        values = op.operands[1]
        validity = op.operands[2] if source.info.nullable else None
        b = Builder().before(op)
        if isinstance(source.info.dtype, BoolType):
            enough = core.cmp(b, "ge", self.length(b, values), self._bytes_for(b, source.length))  # type: ignore[attr-defined]
        else:
            enough = core.cmp(b, "ge", self.length(b, values), source.length)  # type: ignore[attr-defined]
        core.guard(b, enough, "bounds", "the values buffer holds fewer rows than the column")
        if validity is not None:
            wide = core.cmp(b, "ge", self.length(b, validity), self._bytes_for(b, source.length))  # type: ignore[attr-defined]
            core.guard(b, wide, "bounds", "the validity bitmap holds fewer bits than the column")
        target = _Column(source.info, values, validity, source.length)
        inner, i, _next = self.range_loop(op, self._i64(b, 0), source.length)  # type: ignore[attr-defined]
        value, valid = self._get(inner, source, i)
        self._put(inner, target, i, value, valid)

    def op_columnar_fill(self, op: Operation) -> None:
        info = columnar.describe(op.result.type)
        assert info is not None
        b = Builder().before(op)
        one = self._new_column(b, info.dtype, False, self._i64(b, 1), "fill")
        self._put(b, one, self._i64(b, 0), op.operands[0], core.const(b, True, BOOL))
        self.columns[id(op.result)] = _Column(info, one.values, None, op.operands[1], constant=True)

    def op_columnar_length(self, op: Operation) -> None:
        operand = op.operands[0]
        if id(operand) in self.columns:
            op.result.replace_all_uses_with(self.column(operand).length)
            return
        table = self.table(operand)
        b = Builder().before(op)
        length = table.columns[0].length if table.columns else self._i64(b, 0)
        op.result.replace_all_uses_with(length)

    # -- elementwise ------------------------------------------------------------------------

    def _elementwise2(
        self, op: Operation, compute: Callable[[Builder, Value, Value], Value]
    ) -> None:
        a, c = self.column(op.operands[0]), self.column(op.operands[1])
        info = columnar.describe(op.result.type)
        assert info is not None
        b = Builder().before(op)
        rows = self._same_length(b, a, c)
        result = self._new_column(b, info.dtype, info.nullable, rows, op.local_name)
        inner, i, _next = self.range_loop(op, self._i64(b, 0), rows)  # type: ignore[attr-defined]
        x, vx = self._get(inner, a, i)
        y, vy = self._get(inner, c, i)
        self._put(inner, result, i, compute(inner, x, y), self._bitwise(inner, "and", vx, vy))
        self.columns[id(op.result)] = result

    def _arithmetic(self, name: str):  # type: ignore[no-untyped-def]
        def handler(op: Operation) -> None:
            self._elementwise2(
                op,
                lambda b, x, y: tensors.scalar_binary(b, name, x, y, self.module),  # type: ignore[arg-type]
            )

        return handler

    def _compare(self, b: Builder, name: str, x: Value, y: Value) -> Value:
        """`name` of two values, IEEE's way: `not_equal` is the negation of `equal`,
        so a NaN differs from everything, as pandas and Arrow have it."""
        if name == "not_equal":
            return self._bitwise(b, "xor", core.cmp(b, "eq", x, y), core.const(b, True, BOOL))
        return core.cmp(b, _PREDICATES[name], x, y)

    def _comparison(self, name: str):  # type: ignore[no-untyped-def]
        def handler(op: Operation) -> None:
            self._elementwise2(op, lambda b, x, y: self._compare(b, name, x, y))

        return handler

    def _boolean(self, name: str):  # type: ignore[no-untyped-def]
        def handler(op: Operation) -> None:
            self._elementwise2(op, lambda b, x, y: self._bitwise(b, name, x, y))

        return handler

    def _elementwise1(
        self,
        op: Operation,
        compute: Callable[[Builder, Value, Value], tuple[Value, Value]],
        hint: str,
    ) -> None:
        a = self.column(op.operands[0])
        info = columnar.describe(op.result.type)
        assert info is not None
        b = Builder().before(op)
        result = self._new_column(b, info.dtype, info.nullable, a.length, hint)
        inner, i, _next = self.range_loop(op, self._i64(b, 0), a.length)  # type: ignore[attr-defined]
        x, vx = self._get(inner, a, i)
        value, valid = compute(inner, x, vx)
        self._put(inner, result, i, value, valid)
        self.columns[id(op.result)] = result

    def op_columnar_negate(self, op: Operation) -> None:
        self._elementwise1(
            op,
            lambda b, x, v: (tensors.scalar_unary(b, "neg", x, self.module), v),
            "negate",  # type: ignore[arg-type]
        )

    def op_columnar_abs(self, op: Operation) -> None:
        self._elementwise1(
            op,
            lambda b, x, v: (tensors.scalar_unary(b, "abs", x, self.module), v),
            "abs",  # type: ignore[arg-type]
        )

    def op_columnar_invert(self, op: Operation) -> None:
        self._elementwise1(
            op, lambda b, x, v: (self._bitwise(b, "xor", x, core.const(b, True, BOOL)), v), "invert"
        )

    def op_columnar_is_null(self, op: Operation) -> None:
        self._elementwise1(
            op,
            lambda b, x, v: (
                self._bitwise(b, "xor", v, core.const(b, True, BOOL)),
                core.const(b, True, BOOL),
            ),
            "is_null",
        )

    def op_columnar_is_valid(self, op: Operation) -> None:
        self._elementwise1(op, lambda b, x, v: (v, core.const(b, True, BOOL)), "is_valid")

    def op_columnar_fill_null(self, op: Operation) -> None:
        scalar = op.operands[1]
        self._elementwise1(
            op,
            lambda b, x, v: (core.select(b, v, x, scalar), core.const(b, True, BOOL)),
            "fill_null",
        )

    def op_columnar_cast(self, op: Operation) -> None:
        info = columnar.describe(op.result.type)
        assert info is not None
        self._elementwise1(
            op,
            lambda b, x, v: (x if x.type == info.dtype else core.cast(b, x, info.dtype), v),
            "cast",
        )

    def op_columnar_select(self, op: Operation) -> None:
        mask = self.column(op.operands[0])
        a, c = self.column(op.operands[1]), self.column(op.operands[2])
        info = columnar.describe(op.result.type)
        assert info is not None
        b = Builder().before(op)
        rows = self._same_length(b, mask, a)
        self._same_length(b, a, c)
        result = self._new_column(b, info.dtype, info.nullable, rows, "select")
        inner, i, _next = self.range_loop(op, self._i64(b, 0), rows)  # type: ignore[attr-defined]
        m, vm = self._get(inner, mask, i)
        x, vx = self._get(inner, a, i)
        y, vy = self._get(inner, c, i)
        value = core.select(inner, m, x, y)
        valid = self._bitwise(inner, "and", vm, core.select(inner, m, vx, vy))
        self._put(inner, result, i, value, valid)
        self.columns[id(op.result)] = result

    # -- map: one loop over an expression tree ------------------------------------------------

    def op_columnar_map(self, op: Operation) -> None:
        """One loop; each row reads its inputs once and writes the answer straight out.

        Every node gives a (value, valid) pair. Under the `bits` model an
        input's validity is its bitmap and the answer's goes to the validity
        buffer, bit by bit. Under the `nan` model an `f64` input is valid
        where it is not NaN, a `bool` input is a byte per row, and the
        answer is stored as computed: an arithmetic or `select` over a null
        is already NaN, a comparison is IEEE's (`NaN != x` is true, as
        pandas has it), and `is_null`/`fill_null` test the value itself.
        """
        expression = columnar.parse_expression(str(op.attributes["expression"]))
        model = str(op.attributes["model"])
        bool_answer = str(op.attributes["gives"]) == "bool"
        out, outv = op.operands[0], op.operands[1]
        columns: list[_Column] = []
        scalars: list[Value] = []
        for operand in op.operands[2:]:
            if columnar.describe(operand.type) is not None:
                columns.append(self.column(operand))
            else:
                scalars.append(operand)
        b = Builder().before(op)
        rows = columns[0].length
        for other in columns[1:]:
            rows = self._same_length(b, columns[0], other)
        bits_out = model == "bits" and bool_answer
        wanted = self._bytes_for(b, rows) if bits_out else rows
        enough = core.cmp(b, "ge", self.length(b, out), wanted)  # type: ignore[attr-defined]
        core.guard(b, enough, "bounds", "the values buffer holds fewer rows than the column")
        if model == "bits":
            wide = core.cmp(b, "ge", self.length(b, outv), self._bytes_for(b, rows))  # type: ignore[attr-defined]
            core.guard(b, wide, "bounds", "the validity bitmap holds fewer bits than the column")
        inner, i, _next = self.range_loop(op, self._i64(b, 0), rows)  # type: ignore[attr-defined]
        true = core.const(inner, True, BOOL)

        def leaf(index: int) -> tuple[Value, Value]:
            column = columns[index]
            if model == "bits":
                return self._get(inner, column, i)
            if isinstance(column.info.dtype, BoolType):
                byte = core.buffer_load(inner, column.values, i)
                return core.cmp(inner, "ne", byte, self._u8(inner, 0)), true
            value = core.buffer_load(inner, column.values, i)
            return value, core.cmp(inner, "eq", value, value)

        def evaluate(node: columnar.Node) -> tuple[Value, Value]:
            if node.op == "array":
                return leaf(node.array)
            if node.op == "scalar":
                return scalars[node.scalar], true
            if node.op == "constant":
                return core.const(inner, node.constant, F64), true
            if node.op == "fill_null":
                x, vx = evaluate(node.operands[0])
                filler, _ = evaluate(node.operands[1])
                return core.select(inner, vx, x, filler), true
            pairs = [evaluate(child) for child in node.operands]
            if node.op in columnar.ARITHMETIC:
                (x, vx), (y, vy) = pairs
                value = tensors.scalar_binary(inner, node.op, x, y, self.module)  # type: ignore[arg-type]
                return value, self._bitwise(inner, "and", vx, vy)
            if node.op in columnar.COMPARISON:
                (x, vx), (y, vy) = pairs
                return self._compare(inner, node.op, x, y), self._bitwise(inner, "and", vx, vy)
            if node.op in columnar.BOOLEAN:
                (x, vx), (y, vy) = pairs
                return self._bitwise(inner, node.op, x, y), self._bitwise(inner, "and", vx, vy)
            if node.op in {"negate", "abs"}:
                ((x, vx),) = pairs
                return tensors.scalar_unary(
                    inner, "neg" if node.op == "negate" else "abs", x, self.module
                ), vx  # type: ignore[arg-type]
            if node.op == "invert":
                ((x, vx),) = pairs
                return self._bitwise(inner, "xor", x, true), vx
            if node.op == "is_null":
                ((_x, vx),) = pairs
                return self._bitwise(inner, "xor", vx, true), true
            if node.op == "is_valid":
                ((_x, vx),) = pairs
                return vx, true
            if node.op == "select":
                (m, vm), (x, vx), (y, vy) = pairs
                value = core.select(inner, m, x, y)
                return value, self._bitwise(inner, "and", vm, core.select(inner, m, vx, vy))
            raise self._error(f"columnar.map cannot evaluate {node.op!r}")

        value, valid = evaluate(expression)
        if model == "bits":
            target = _Column(
                columnar.ColumnInfo(BOOL if bool_answer else F64, True), out, outv, rows
            )
            self._put(inner, target, i, value, valid)
        elif bool_answer:
            byte = core.select(inner, value, self._u8(inner, 1), self._u8(inner, 0))
            core.buffer_store(inner, byte, out, i)
        else:
            core.buffer_store(inner, value, out, i)

    # -- filter, take, concat -----------------------------------------------------------------

    def _hits(self, op: Operation, mask: _Column) -> Value:
        """How many rows a mask keeps: a true that is valid."""
        b = Builder().before(op)
        count = self.slot(b, I64, self._i64(b, 0), "hits")  # type: ignore[attr-defined]
        inner, i, _next = self.range_loop(op, self._i64(b, 0), mask.length)  # type: ignore[attr-defined]
        m, vm = self._get(inner, mask, i)
        hit = core.cast(inner, self._bitwise(inner, "and", m, vm), I64)
        core.store(inner, core.add(inner, core.load(inner, count), hit, overflow="wrap"), count)
        return core.load(Builder().before(op), count)

    def _filter_column(self, op: Operation, source: _Column, mask: _Column, kept: Value) -> _Column:
        """The rows of `source` the mask keeps, `kept` of them, in order.

        Every row is written at the running position -- a row not kept is
        overwritten by the next one kept -- so the loop has no branch; one
        spare row absorbs the last write past the end.
        """
        b = Builder().before(op)
        spare = core.add(b, kept, self._i64(b, 1), overflow="wrap")
        result = self._new_column(b, source.info.dtype, source.info.nullable, spare, "filter")
        result.length = kept
        position = self.slot(b, I64, self._i64(b, 0), "pos")  # type: ignore[attr-defined]
        inner, i, _next = self.range_loop(op, self._i64(b, 0), mask.length)  # type: ignore[attr-defined]
        m, vm = self._get(inner, mask, i)
        hit = self._bitwise(inner, "and", m, vm)
        at = core.load(inner, position)
        value, valid = self._get(inner, source, i)
        self._put(inner, result, at, value, valid)
        step = core.cast(inner, hit, I64)
        core.store(inner, core.add(inner, at, step, overflow="wrap"), position)
        return result

    def op_columnar_filter(self, op: Operation) -> None:
        mask = self.column(op.operands[1])
        b = Builder().before(op)
        source = op.operands[0]
        if id(source) in self.columns:
            column = self.column(source)
            self._same_length(b, column, mask)
            kept = self._hits(op, mask)
            self.columns[id(op.result)] = self._filter_column(op, column, mask, kept)
            return
        table = self.table(source)
        for column in table.columns:
            self._same_length(b, column, mask)
        kept = self._hits(op, mask)
        self.tables[id(op.result)] = _Table(
            table.info, [self._filter_column(op, c, mask, kept) for c in table.columns]
        )

    def _take_column(
        self, op: Operation, source: _Column, indices: _Column, nullable: bool
    ) -> _Column:
        """Row `indices[i]` of `source` at row `i`; a null position is a null row."""
        b = Builder().before(op)
        result = self._new_column(b, source.info.dtype, nullable, indices.length, "take")
        inner, i, _next = self.range_loop(op, self._i64(b, 0), indices.length)  # type: ignore[attr-defined]
        index, has_index = self._get(inner, indices, i)
        index = index if index.type == I64 else core.cast(inner, index, I64)
        # A fresh column's bitmaps are zero: a row not written is a null row.
        then, _after = self._when(inner, has_index, "take")
        in_range = self._bitwise(
            then,
            "and",
            core.cmp(then, "ge", index, self._i64(then, 0)),
            core.cmp(then, "lt", index, source.length),
        )
        core.guard(then, in_range, "bounds", "a take position is outside the column")
        value, valid = self._get(then, source, index)
        self._put(then, result, i, value, valid)
        return result

    def op_columnar_take(self, op: Operation) -> None:
        indices = self.column(op.operands[1])
        source = op.operands[0]
        if id(source) in self.columns:
            column = self.column(source)
            info = columnar.describe(op.result.type)
            assert info is not None
            self.columns[id(op.result)] = self._take_column(op, column, indices, info.nullable)
            return
        table = self.table(source)
        info = columnar.describe_table(op.result.type)
        assert info is not None
        self.tables[id(op.result)] = _Table(
            info,
            [
                self._take_column(op, c, indices, out.nullable)
                for c, out in zip(table.columns, info.columns, strict=True)
            ],
        )

    def _concat_columns(
        self, op: Operation, parts: list[_Column], info: columnar.ColumnInfo
    ) -> _Column:
        b = Builder().before(op)
        total: Value = self._i64(b, 0)
        for part in parts:
            total = core.add(b, total, part.length, overflow="wrap")
        result = self._new_column(b, info.dtype, info.nullable, total, "concat")
        offset: Value = self._i64(b, 0)
        for part in parts:
            inner, i, _next = self.range_loop(op, self._i64(b, 0), part.length)  # type: ignore[attr-defined]
            value, valid = self._get(inner, part, i)
            self._put(inner, result, core.add(inner, offset, i, overflow="wrap"), value, valid)
            b = Builder().before(op)
            offset = core.add(b, offset, part.length, overflow="wrap")
        return result

    def op_columnar_concat(self, op: Operation) -> None:
        first = op.operands[0]
        if id(first) in self.columns:
            info = columnar.describe(op.result.type)
            assert info is not None
            parts = [self.column(v) for v in op.operands]
            self.columns[id(op.result)] = self._concat_columns(op, parts, info)
            return
        info = columnar.describe_table(op.result.type)
        assert info is not None
        tables = [self.table(v) for v in op.operands]
        self.tables[id(op.result)] = _Table(
            info,
            [
                self._concat_columns(op, [t.columns[k] for t in tables], column)
                for k, column in enumerate(info.columns)
            ],
        )

    # -- sorting ---------------------------------------------------------------------------

    def _before_or_equal(self, b: Builder, a: Value, va: Value, c: Value, vc: Value) -> Value:
        """Does row `a` sort no later than row `c`? Ascending, nulls last, ties kept."""
        both = core.cmp(b, "le", a, c)
        not_vc = self._bitwise(b, "xor", vc, core.const(b, True, BOOL))
        return core.select(b, va, self._bitwise(b, "or", not_vc, both), not_vc)

    def _sorted_positions(self, op: Operation, column: _Column) -> Value:
        """A buffer of the row positions of `column` in sorted order: a bottom-up merge sort."""
        b = Builder().before(op)
        n = column.length
        order = self.allocate_dynamic(b, I64, n, "order")  # type: ignore[attr-defined]
        scratch = self.allocate_dynamic(b, I64, n, "scratch")  # type: ignore[attr-defined]
        inner, i, _next = self.range_loop(op, self._i64(b, 0), n)  # type: ignore[attr-defined]
        core.buffer_store(inner, i, order, i)
        b = Builder().before(op)
        width = self.slot(b, I64, self._i64(b, 1), "width")  # type: ignore[attr-defined]
        # while width < n: merge every pair of runs of `width` rows, then double it.
        cond, body, _done = self.while_(op)  # type: ignore[attr-defined]
        self.close_while(cond, core.cmp(cond, "lt", core.load(cond, width), n))  # type: ignore[attr-defined]
        w = core.load(body, width)
        start = self.slot(body, I64, self._i64(body, 0), "start")  # type: ignore[attr-defined]
        anchor = body.anchor
        assert anchor is not None
        cond2, body2, _done2 = self.while_(anchor)  # type: ignore[attr-defined]
        self.close_while(cond2, core.cmp(cond2, "lt", core.load(cond2, start), n))  # type: ignore[attr-defined]
        s = core.load(body2, start)
        mid = _minimum(body2, core.add(body2, s, w, overflow="wrap"), n)
        end = _minimum(body2, core.add(body2, mid, w, overflow="wrap"), n)
        left = self.slot(body2, I64, s, "left")  # type: ignore[attr-defined]
        right = self.slot(body2, I64, mid, "right")  # type: ignore[attr-defined]
        anchor2 = body2.anchor
        assert anchor2 is not None
        inner, k, _next = self.range_loop(anchor2, s, end)  # type: ignore[attr-defined]
        li = core.load(inner, left)
        ri = core.load(inner, right)
        left_left = core.cmp(inner, "lt", li, mid)
        right_left = core.cmp(inner, "lt", ri, end)
        safe_l = core.select(inner, left_left, li, s)
        safe_r = core.select(inner, right_left, ri, mid)
        a_row = core.buffer_load(inner, order, safe_l)
        c_row = core.buffer_load(inner, order, core.select(inner, right_left, safe_r, safe_l))
        a, va = self._get(inner, column, a_row)
        c, vc = self._get(inner, column, c_row)
        take_left = self._bitwise(
            inner,
            "or",
            self._bitwise(inner, "xor", right_left, core.const(inner, True, BOOL)),
            self._bitwise(inner, "and", left_left, self._before_or_equal(inner, a, va, c, vc)),
        )
        core.buffer_store(inner, core.select(inner, take_left, a_row, c_row), scratch, k)
        one = self._i64(inner, 1)
        core.store(
            inner,
            core.select(inner, take_left, core.add(inner, li, one, overflow="wrap"), li),
            left,
        )
        core.store(
            inner,
            core.select(inner, take_left, ri, core.add(inner, ri, one, overflow="wrap")),
            right,
        )
        after = Builder().before(anchor2)
        core.store(after, end, start)
        # The pass is done: the merged order comes back from the scratch buffer.
        settled = Builder().before(anchor)
        copy, k2, _next = self.range_loop(anchor, self._i64(settled, 0), n)  # type: ignore[attr-defined]
        core.buffer_store(copy, core.buffer_load(copy, scratch, k2), order, k2)
        after_pass = Builder().before(anchor)
        core.store(after_pass, core.add(after_pass, w, w, overflow="wrap"), width)
        return order

    def op_columnar_sort_indices(self, op: Operation) -> None:
        column = self.column(op.operands[0])
        order = self._sorted_positions(op, column)
        self.columns[id(op.result)] = _Column(
            columnar.ColumnInfo(I64, False), order, None, column.length
        )

    # -- aggregation -----------------------------------------------------------------------

    def op_columnar_aggregate(self, op: Operation) -> None:
        column = self.column(op.operands[0])
        function = str(op.attributes["function"])
        info = columnar.describe(op.result.type)
        assert info is not None
        dtype = column.info.dtype
        b = Builder().before(op)
        result = self._new_column(b, info.dtype, info.nullable, self._i64(b, 1), function)
        count = self.slot(b, I64, self._i64(b, 0), "count")  # type: ignore[attr-defined]
        if function in {"sum", "mean"}:
            initial = self._zero_of(b, dtype)
        elif function == "any":
            initial = core.const(b, False, BOOL)
        elif function == "all":
            initial = core.const(b, True, BOOL)
        else:
            initial = self._zero_of(b, dtype)
        total = self.slot(b, dtype, initial, "acc") if function != "count" else None  # type: ignore[attr-defined]
        inner, i, _next = self.range_loop(op, self._i64(b, 0), column.length)  # type: ignore[attr-defined]
        x, valid = self._get(inner, column, i)
        seen = core.load(inner, count)
        core.store(
            inner, core.add(inner, seen, core.cast(inner, valid, I64), overflow="wrap"), count
        )
        if total is not None:
            current = core.load(inner, total)
            if function in {"sum", "mean"}:
                folded = tensors.scalar_binary(inner, "add", current, x, self.module)  # type: ignore[arg-type]
            elif function in {"min", "max"}:
                first = core.cmp(inner, "eq", seen, self._i64(inner, 0))
                extreme = tensors.scalar_extremum(inner, function, current, x)
                folded = core.select(inner, first, x, extreme)
            elif function == "any":
                folded = self._bitwise(inner, "or", current, x)
            else:
                folded = self._bitwise(inner, "and", current, x)
            core.store(inner, core.select(inner, valid, folded, current), total)
        b = Builder().before(op)
        seen = core.load(b, count)
        valid = core.cmp(b, "gt", seen, self._i64(b, 0))
        if function == "count":
            value: Value = seen
        else:
            assert total is not None
            value = core.load(b, total)
            if function == "mean":
                rows = core.select(b, valid, seen, self._i64(b, 1))
                value = core.div(b, value, core.cast(b, rows, dtype))
        self._put(b, result, self._i64(b, 0), value, valid)
        self.columns[id(op.result)] = result

    # -- tables ------------------------------------------------------------------------------

    def op_columnar_make(self, op: Operation) -> None:
        info = columnar.describe_table(op.result.type)
        assert info is not None
        b = Builder().before(op)
        parts = [self.column(v) for v in op.operands]
        for part in parts[1:]:
            self._same_length(b, parts[0], part)
        self.tables[id(op.result)] = _Table(info, parts)

    def op_columnar_column_of(self, op: Operation) -> None:
        table = self.table(op.operands[0])
        index = table.info.index(str(op.attributes["name"]))
        assert index is not None
        self.columns[id(op.result)] = table.columns[index]

    def op_columnar_project(self, op: Operation) -> None:
        table = self.table(op.operands[0])
        info = columnar.describe_table(op.result.type)
        assert info is not None
        kept = []
        for name in info.names:
            index = table.info.index(name)
            assert index is not None
            kept.append(table.columns[index])
        self.tables[id(op.result)] = _Table(info, kept)

    def op_columnar_group_by(self, op: Operation) -> None:
        table = self.table(op.operands[0])
        info = columnar.describe_table(op.result.type)
        assert info is not None
        key_index = table.info.index(str(op.attributes["key"]))
        assert key_index is not None
        key = table.columns[key_index]
        aggregates = tuple(op.attributes["aggregates"])  # type: ignore[arg-type]
        order = self._sorted_positions(op, key)
        b = Builder().before(op)
        n = key.length
        # Pass one: how many runs of equal keys.
        groups = self.slot(b, I64, self._i64(b, 0), "groups")  # type: ignore[attr-defined]
        inner, p, _next = self.range_loop(op, self._i64(b, 0), n)  # type: ignore[attr-defined]
        row = core.buffer_load(inner, order, p)
        k, _valid = self._get(inner, key, row)
        previous_row = core.buffer_load(
            inner,
            order,
            core.select(
                inner,
                core.cmp(inner, "gt", p, self._i64(inner, 0)),
                core.sub(inner, p, self._i64(inner, 1), overflow="wrap"),
                p,
            ),
        )
        previous, _pv = self._get(inner, key, previous_row)
        starts_run = self._bitwise(
            inner,
            "or",
            core.cmp(inner, "eq", p, self._i64(inner, 0)),
            core.cmp(inner, "ne", k, previous),
        )
        core.store(
            inner,
            core.add(
                inner, core.load(inner, groups), core.cast(inner, starts_run, I64), overflow="wrap"
            ),
            groups,
        )
        b = Builder().before(op)
        count = core.load(b, groups)
        zeroed = [False] + [function in {"sum", "count"} for _name, function in aggregates]
        outputs = [
            self._new_column(b, c.dtype, c.nullable, count, name, zero=zero)
            for (name, c), zero in zip(
                zip(info.names, info.columns, strict=True), zeroed, strict=True
            )
        ]
        # Pass two: fold each row into its group.
        group = self.slot(b, I64, self._i64(b, -1), "group")  # type: ignore[attr-defined]
        inner, p, _next = self.range_loop(op, self._i64(b, 0), n)  # type: ignore[attr-defined]
        row = core.buffer_load(inner, order, p)
        k, _valid = self._get(inner, key, row)
        previous_row = core.buffer_load(
            inner,
            order,
            core.select(
                inner,
                core.cmp(inner, "gt", p, self._i64(inner, 0)),
                core.sub(inner, p, self._i64(inner, 1), overflow="wrap"),
                p,
            ),
        )
        previous, _pv = self._get(inner, key, previous_row)
        starts_run = self._bitwise(
            inner,
            "or",
            core.cmp(inner, "eq", p, self._i64(inner, 0)),
            core.cmp(inner, "ne", k, previous),
        )
        g = core.add(
            inner, core.load(inner, group), core.cast(inner, starts_run, I64), overflow="wrap"
        )
        core.store(inner, g, group)
        self._put(inner, outputs[0], g, k, core.const(inner, True, BOOL))
        for position, (name, function) in enumerate(aggregates, start=1):
            source_index = table.info.index(str(name))
            assert source_index is not None
            source = table.columns[source_index]
            output = outputs[position]
            x, valid = self._get(inner, source, row)
            current, has = self._get(inner, output, g)
            if function == "count":
                folded = core.add(inner, current, core.cast(inner, valid, I64), overflow="wrap")
                self._put(inner, output, g, folded, core.const(inner, True, BOOL))
                continue
            if function == "sum":
                folded = tensors.scalar_binary(inner, "add", current, x, self.module)  # type: ignore[arg-type]
            else:
                extreme = tensors.scalar_extremum(inner, function, current, x)
                folded = core.select(inner, has, extreme, x)
            self._put(
                inner,
                output,
                g,
                core.select(inner, valid, folded, current),
                self._bitwise(inner, "or", has, valid),
            )
        self.tables[id(op.result)] = _Table(info, outputs)

    def op_columnar_join(self, op: Operation) -> None:
        left, right = self.table(op.operands[0]), self.table(op.operands[1])
        info = columnar.describe_table(op.result.type)
        assert info is not None
        key = str(op.attributes["key"])
        li, ri = left.info.index(key), right.info.index(key)
        assert li is not None and ri is not None
        left_key, right_key = left.columns[li], right.columns[ri]
        left_order = self._sorted_positions(op, left_key)
        right_order = self._sorted_positions(op, right_key)
        b = Builder().before(op)
        # Two sweeps of the same merge: the first counts the pairs, the second writes them.
        matches = self._merge(op, left_key, right_key, left_order, right_order, None)
        b = Builder().before(op)
        left_rows = self.allocate_dynamic(b, I64, matches, "left_rows")  # type: ignore[attr-defined]
        right_rows = self.allocate_dynamic(b, I64, matches, "right_rows")  # type: ignore[attr-defined]
        self._merge(op, left_key, right_key, left_order, right_order, (left_rows, right_rows))
        left_positions = _Column(columnar.ColumnInfo(I64, False), left_rows, None, matches)
        right_positions = _Column(columnar.ColumnInfo(I64, False), right_rows, None, matches)
        columns = [self._take_column(op, c, left_positions, c.info.nullable) for c in left.columns]
        for name, column in zip(right.info.names, right.columns, strict=True):
            if name != key:
                columns.append(self._take_column(op, column, right_positions, column.info.nullable))
        self.tables[id(op.result)] = _Table(info, columns)

    def _merge(
        self,
        op: Operation,
        left_key: _Column,
        right_key: _Column,
        left_order: Value,
        right_order: Value,
        outputs: tuple[Value, Value] | None,
    ) -> Value:
        """Walk both sorted key columns together; count the equal pairs, or write them."""
        b = Builder().before(op)
        i = self.slot(b, I64, self._i64(b, 0), "i")  # type: ignore[attr-defined]
        j = self.slot(b, I64, self._i64(b, 0), "j")  # type: ignore[attr-defined]
        pairs = self.slot(b, I64, self._i64(b, 0), "pairs")  # type: ignore[attr-defined]
        nl, nr = left_key.length, right_key.length
        cond, body, _done = self.while_(op)  # type: ignore[attr-defined]
        both = self._bitwise(
            cond,
            "and",
            core.cmp(cond, "lt", core.load(cond, i), nl),
            core.cmp(cond, "lt", core.load(cond, j), nr),
        )
        self.close_while(cond, both)  # type: ignore[attr-defined]
        ci, cj = core.load(body, i), core.load(body, j)
        kl, _vl = self._get(body, left_key, core.buffer_load(body, left_order, ci))
        kr, _vr = self._get(body, right_key, core.buffer_load(body, right_order, cj))
        one = self._i64(body, 1)
        less = core.cmp(body, "lt", kl, kr)
        greater = core.cmp(body, "gt", kl, kr)
        # The end of each run of the shared key, found by two counted scans.
        left_end = self.slot(body, I64, ci, "left_end")  # type: ignore[attr-defined]
        right_end = self.slot(body, I64, cj, "right_end")  # type: ignore[attr-defined]
        anchor = body.anchor
        assert anchor is not None
        cond2, body2, _d2 = self.while_(anchor)  # type: ignore[attr-defined]
        e = core.load(cond2, left_end)
        in_range = core.cmp(cond2, "lt", e, nl)
        safe = core.select(cond2, in_range, e, ci)
        ke, _ve = self._get(cond2, left_key, core.buffer_load(cond2, left_order, safe))
        self.close_while(
            cond2, self._bitwise(cond2, "and", in_range, core.cmp(cond2, "eq", ke, kl))
        )  # type: ignore[attr-defined]
        core.store(
            body2, core.add(body2, core.load(body2, left_end), one, overflow="wrap"), left_end
        )
        cond3, body3, _d3 = self.while_(anchor)  # type: ignore[attr-defined]
        e = core.load(cond3, right_end)
        in_range = core.cmp(cond3, "lt", e, nr)
        safe = core.select(cond3, in_range, e, cj)
        ke, _ve = self._get(cond3, right_key, core.buffer_load(cond3, right_order, safe))
        self.close_while(
            cond3, self._bitwise(cond3, "and", in_range, core.cmp(cond3, "eq", ke, kr))
        )  # type: ignore[attr-defined]
        core.store(
            body3, core.add(body3, core.load(body3, right_end), one, overflow="wrap"), right_end
        )
        tail = Builder().before(anchor)
        le, re = core.load(tail, left_end), core.load(tail, right_end)
        equal = self._bitwise(
            tail,
            "and",
            self._bitwise(tail, "xor", less, core.const(tail, True, BOOL)),
            self._bitwise(tail, "xor", greater, core.const(tail, True, BOOL)),
        )
        if outputs is not None:
            left_rows, right_rows = outputs
            # Every pair of the two runs, in left-major order, from the running pair count.
            then, tail = self._when(tail, equal, "pairs")
            base = core.load(then, pairs)
            outer, a, _n1 = self.range_loop(then.anchor, ci, le)  # type: ignore[attr-defined,arg-type]
            width = core.sub(outer, re, cj, overflow="wrap")
            inner, c, _n2 = self.range_loop(outer.anchor, cj, re)  # type: ignore[attr-defined,arg-type]
            slot_index = core.add(
                inner,
                base,
                core.add(
                    inner,
                    core.mul(
                        inner, core.sub(inner, a, ci, overflow="wrap"), width, overflow="wrap"
                    ),
                    core.sub(inner, c, cj, overflow="wrap"),
                    overflow="wrap",
                ),
                overflow="wrap",
            )
            core.buffer_store(inner, core.buffer_load(inner, left_order, a), left_rows, slot_index)
            core.buffer_store(
                inner, core.buffer_load(inner, right_order, c), right_rows, slot_index
            )
        product = core.mul(
            tail,
            core.sub(tail, le, ci, overflow="wrap"),
            core.sub(tail, re, cj, overflow="wrap"),
            overflow="wrap",
        )
        zero = self._i64(tail, 0)
        core.store(
            tail,
            core.add(
                tail,
                core.load(tail, pairs),
                core.select(tail, equal, product, zero),
                overflow="wrap",
            ),
            pairs,
        )
        # Advance past the smaller key's run, or past both runs when the keys are equal.
        next_i = core.select(
            tail, less, core.add(tail, ci, one, overflow="wrap"), core.select(tail, equal, le, ci)
        )
        next_j = core.select(
            tail,
            greater,
            core.add(tail, cj, one, overflow="wrap"),
            core.select(tail, equal, re, cj),
        )
        core.store(tail, next_i, i)
        core.store(tail, next_j, j)
        return core.load(Builder().before(op), pairs)

    # -- arrow -------------------------------------------------------------------------------

    def _field(self, b: Builder, struct: Value, offset: int) -> Value:
        address = core.ptr_offset(b, struct, self._i64(b, offset))
        return core.load(b, core.cast(b, address, PtrType(I64)))

    def op_arrow_import(self, op: Operation) -> None:
        dtype = arrow.element_of(op.result.type)
        assert dtype is not None
        b = Builder().before(op)
        struct = op.operands[0]
        length = self._field(b, struct, arrow.LENGTH)
        null_count = self._field(b, struct, arrow.NULL_COUNT)
        offset = self._field(b, struct, arrow.OFFSET)
        buffers_field = core.ptr_offset(b, struct, self._i64(b, arrow.BUFFERS))
        pointers = core.load(b, core.cast(b, buffers_field, PtrType(PtrType(PtrType(U8)))))
        addresses = core.load(b, core.cast(b, buffers_field, PtrType(PtrType(I64))))
        validity = core.load(b, core.ptr_offset(b, pointers, self._i64(b, 0)))
        validity_address = core.load(b, core.ptr_offset(b, addresses, self._i64(b, 0)))
        values = core.load(b, core.ptr_offset(b, pointers, self._i64(b, 1)))
        self.arrays[id(op.result)] = _Arrow(
            dtype, length, null_count, offset, validity, validity_address, values
        )

    def _array(self, value: Value) -> _Arrow:
        found = self.arrays.get(id(value))
        if found is None:
            raise self._error(f"arrow array %{value.name or '?'} was never imported")
        return found

    def op_arrow_length(self, op: Operation) -> None:
        op.result.replace_all_uses_with(self._array(op.operands[0]).length)

    def op_arrow_null_count(self, op: Operation) -> None:
        op.result.replace_all_uses_with(self._array(op.operands[0]).null_count)

    def op_arrow_offset(self, op: Operation) -> None:
        op.result.replace_all_uses_with(self._array(op.operands[0]).offset)

    def op_arrow_to_column(self, op: Operation) -> None:
        array = self._array(op.operands[0])
        b = Builder().before(op)
        n = array.length
        if isinstance(array.dtype, BoolType):
            aligned = core.cmp(
                b, "eq", self._bitwise(b, "and", array.offset, self._i64(b, 7)), self._i64(b, 0)
            )
            core.guard(b, aligned, "contract", "a bool array sliced inside a byte is not viewed")
            start = core.ptr_offset(
                b, array.values, self._bitwise(b, "shr", array.offset, self._i64(b, 3))
            )
            values = core.call_intrinsic(
                b, "ppy.buffer_from_parts", (start, self._bytes_for(b, n)), (BufferType(U8),)
            ).results[0]
        else:
            typed = core.cast(b, array.values, PtrType(array.dtype))
            start = core.ptr_offset(b, typed, array.offset)
            values = core.call_intrinsic(
                b, "ppy.buffer_from_parts", (start, n), (BufferType(array.dtype),)
            ).results[0]
        # The validity bitmap: absent, or sliced at any bit. A fresh bitmap is
        # filled from it -- all ones when there is none -- so the column's rows
        # start at bit zero, as the columnar operations read them.
        validity = self._bitmap(b, n, "valid")
        has_nulls = self._bitwise(
            b,
            "and",
            core.cmp(b, "ne", array.validity_address, self._i64(b, 0)),
            core.cmp(b, "ne", array.null_count, self._i64(b, 0)),
        )
        then, otherwise = self._branch(op, has_nulls, "nulls")
        source_bits = core.call_intrinsic(
            then,
            "ppy.buffer_from_parts",
            (
                array.validity,
                self._bytes_for(then, core.add(then, n, array.offset, overflow="wrap")),
            ),
            (BufferType(U8),),
        ).results[0]
        copy, i, _n = self.range_loop(then.anchor, self._i64(then, 0), n)  # type: ignore[attr-defined,arg-type]
        bit = self._bit(copy, source_bits, core.add(copy, i, array.offset, overflow="wrap"))
        self._write_bit(copy, validity, i, bit)
        fill, i2, _n2 = self.range_loop(
            otherwise.anchor, self._i64(otherwise, 0), self._bytes_for(otherwise, n)
        )  # type: ignore[attr-defined,arg-type]
        core.buffer_store(fill, self._u8(fill, 0xFF), validity, i2)
        self.columns[id(op.result)] = _Column(
            columnar.ColumnInfo(array.dtype, True), values, validity, n
        )


def _minimum(b: Builder, a: Value, c: Value) -> Value:
    return core.select(b, core.cmp(b, "le", a, c), a, c)


def _install(kind: str, names: tuple[str, ...]) -> None:
    """One `op_columnar_<name>` handler per operation of `kind`."""
    for name in names:

        def handler(
            self: ColumnarLowering, op: Operation, _name: str = name, _kind: str = kind
        ) -> None:
            getattr(self, _kind)(_name)(op)

        setattr(ColumnarLowering, f"op_columnar_{name}", handler)


_install("_arithmetic", columnar.ARITHMETIC)
_install("_comparison", columnar.COMPARISON)
_install("_boolean", columnar.BOOLEAN)
