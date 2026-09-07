"""The core dialect: what every backend must lower.

Scalar arithmetic with its overflow semantics spelled on the operation,
comparisons, casts, control flow, stack memory, pointers, buffers, tuples,
structs, calls, and guards. Nothing here is specific to a backend or a
library; a dialect adds what is.

Overflow is an attribute, never a guess: `python` means the true value is
what Python computes and the backend must guard for it, `checked` means an
overflow is a failure the guard catches, and `wrap` means two's-complement
wrap like C. Integer division and remainder carry their rounding the same
way (`floor` is Python's, `trunc` is C's).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Attribute, Builder, Operation, Successor, SymbolRef, Value
from ..types import (
    BOOL,
    I64,
    INDEX,
    BoolType,
    BufferType,
    FloatType,
    IndexType,
    IntType,
    IRType,
    PtrType,
    StructType,
    TupleType,
    VectorType,
    is_arithmetic,
    is_integer,
    is_scalar,
    scalar_of,
)

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "ADDRESS_SPACES",
    "OVERFLOW_MODES",
    "PREDICATES",
    "ROUNDING_MODES",
    "CoreDialect",
]

#: `proven`: a proof -- a corner check hoisted ahead of the loop, or the
#: solver -- established that the true value fits, so the backend emits the
#: plain operation and may tell the optimizer it never wraps.
OVERFLOW_MODES = ("python", "checked", "wrap", "proven")
ROUNDING_MODES = ("floor", "trunc")
PREDICATES = ("eq", "ne", "lt", "le", "gt", "ge")
ADDRESS_SPACES = frozenset({"generic", "stack"})
GUARD_KINDS = ("overflow", "bounds", "zero_division", "range", "contract", "assert")


class CoreDialect(Dialect):
    name = "core"
    version = 1

    def address_spaces(self) -> frozenset[str]:
        return ADDRESS_SPACES

    def register_patterns(self, registry: object) -> None:
        from .core_patterns import register

        register(registry)  # type: ignore[arg-type]

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(
            OpSpec(
                "core.const",
                pure=True,
                inline_attribute="value",
                verify=_verify_const,
                operands=0,
                results=1,
                required_attributes=("value",),
            )
        )
        for name in ("add", "sub", "mul"):
            add(
                OpSpec(
                    f"core.{name}",
                    pure=True,
                    commutative=name != "sub",
                    verify=_verify_arith,
                    operands=2,
                    results=1,
                )
            )
        for name in ("div", "mod"):
            add(OpSpec(f"core.{name}", pure=True, verify=_verify_arith, operands=2, results=1))
        add(OpSpec("core.neg", pure=True, verify=_verify_arith, operands=1, results=1))
        for name in ("and", "or", "xor"):
            add(
                OpSpec(
                    f"core.{name}",
                    pure=True,
                    commutative=True,
                    verify=_verify_bitwise,
                    operands=2,
                    results=1,
                )
            )
        for name in ("shl", "shr"):
            add(OpSpec(f"core.{name}", pure=True, verify=_verify_shift, operands=2, results=1))
        add(
            OpSpec(
                "core.cmp",
                pure=True,
                variant_attribute="predicate",
                variants=PREDICATES,
                verify=_verify_cmp,
                operands=2,
                results=1,
                required_attributes=("predicate",),
            )
        )
        add(OpSpec("core.select", pure=True, verify=_verify_select, operands=3, results=1))
        add(OpSpec("core.cast", pure=True, verify=_verify_cast, operands=1, results=1))
        add(
            OpSpec(
                "core.br", terminator=True, verify=_verify_br, operands=0, results=0, successors=1
            )
        )
        add(
            OpSpec(
                "core.cond_br",
                terminator=True,
                verify=_verify_cond_br,
                operands=1,
                results=0,
                successors=2,
            )
        )
        add(OpSpec("core.ret", terminator=True, verify=_verify_ret, results=0, successors=0))
        add(OpSpec("core.unreachable", terminator=True, operands=0, results=0, successors=0))
        add(OpSpec("core.alloca", verify=_verify_alloca, operands=0, results=1))
        add(OpSpec("core.load", verify=_verify_load, operands=1, results=1))
        add(OpSpec("core.store", verify=_verify_store, operands=2, results=0))
        add(OpSpec("core.ptr_offset", pure=True, verify=_verify_ptr_offset, operands=2, results=1))
        add(
            OpSpec("core.buffer_data", pure=True, verify=_verify_buffer_data, operands=1, results=1)
        )
        add(OpSpec("core.buffer_len", pure=True, verify=_verify_buffer_len, operands=1, results=1))
        add(OpSpec("core.buffer_load", verify=_verify_buffer_load, operands=2, results=1))
        add(OpSpec("core.buffer_store", verify=_verify_buffer_store, operands=3, results=0))
        add(OpSpec("core.tuple_make", pure=True, verify=_verify_tuple_make, results=1))
        add(
            OpSpec(
                "core.tuple_extract",
                pure=True,
                verify=_verify_tuple_extract,
                operands=1,
                results=1,
                required_attributes=("index",),
            )
        )
        add(OpSpec("core.struct_make", pure=True, verify=_verify_struct_make, results=1))
        add(
            OpSpec(
                "core.struct_extract",
                pure=True,
                verify=_verify_struct_extract,
                operands=1,
                results=1,
                required_attributes=("field",),
            )
        )
        add(OpSpec("core.call", verify=_verify_call, required_attributes=("callee",)))
        add(OpSpec("core.call_extern", verify=_verify_call_extern, required_attributes=("callee",)))
        add(OpSpec("core.call_intrinsic", required_attributes=("intrinsic",)))
        add(
            OpSpec(
                "core.guard",
                verify=_verify_guard,
                operands=1,
                results=0,
                required_attributes=("kind",),
            )
        )


# -- verification -----------------------------------------------------------


def _verify_const(op: Operation, checker: Checker) -> None:
    value = op.attributes.get("value")
    t = op.results[0].type
    if isinstance(t, BoolType):
        if not isinstance(value, bool):
            checker.error(op, f"a bool constant needs a bool value, not {value!r}")
    elif isinstance(t, IntType):
        if not isinstance(value, int) or isinstance(value, bool):
            checker.error(op, f"an integer constant needs an integer value, not {value!r}")
        elif not t.fits(value):
            checker.error(op, f"{value} does not fit {t}")
    elif isinstance(t, IndexType):
        if not isinstance(value, int) or isinstance(value, bool):
            checker.error(op, f"an index constant needs an integer value, not {value!r}")
    elif isinstance(t, FloatType):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            checker.error(op, f"a float constant needs a number, not {value!r}")
    else:
        checker.error(op, f"no constant of type {t}")


def _same_operand_types(op: Operation, checker: Checker) -> IRType | None:
    types = {str(v.type) for v in op.operands}
    if len(types) > 1:
        checker.error(op, f"operands differ in type: {', '.join(sorted(types))}")
        return None
    return op.operands[0].type if op.operands else None


def _result_matches(op: Operation, checker: Checker, expected: IRType) -> None:
    if op.results[0].type != expected:
        checker.error(op, f"result is {op.results[0].type}, expected {expected}")


def _verify_arith(op: Operation, checker: Checker) -> None:
    t = _same_operand_types(op, checker)
    if t is None:
        return
    if not is_arithmetic(t):
        checker.error(op, f"{op.name} needs numbers, not {t}")
        return
    _result_matches(op, checker, t)
    scalar = scalar_of(t)
    if is_integer(scalar):
        if op.local_name in {"div", "mod"}:
            rounding = op.attributes.get("rounding")
            if rounding not in ROUNDING_MODES:
                checker.error(op, f"integer {op.name} needs rounding in {ROUNDING_MODES}")
        overflow = op.attributes.get("overflow")
        if overflow not in OVERFLOW_MODES:
            checker.error(op, f"integer {op.name} needs overflow in {OVERFLOW_MODES}")


def _verify_bitwise(op: Operation, checker: Checker) -> None:
    t = _same_operand_types(op, checker)
    if t is None:
        return
    scalar = scalar_of(t)
    if not isinstance(scalar, (IntType, BoolType)):
        checker.error(op, f"{op.name} needs integers or bools, not {t}")
        return
    _result_matches(op, checker, t)


def _verify_shift(op: Operation, checker: Checker) -> None:
    value, amount = op.operands
    if not is_integer(scalar_of(value.type)):
        checker.error(op, f"{op.name} shifts integers, not {value.type}")
        return
    if not is_integer(scalar_of(amount.type)):
        checker.error(op, f"a shift amount is an integer, not {amount.type}")
    _result_matches(op, checker, value.type)


def _verify_cmp(op: Operation, checker: Checker) -> None:
    predicate = op.attributes.get("predicate")
    if predicate not in PREDICATES:
        checker.error(op, f"predicate {predicate!r} is not one of {PREDICATES}")
    t = _same_operand_types(op, checker)
    if t is None:
        return
    scalar = scalar_of(t)
    if not is_scalar(scalar) and not isinstance(scalar, PtrType):
        checker.error(op, f"{op.name} compares scalars, not {t}")
        return
    expected: IRType = VectorType(BOOL, t.count) if isinstance(t, VectorType) else BOOL
    _result_matches(op, checker, expected)


def _verify_select(op: Operation, checker: Checker) -> None:
    condition, a, b = op.operands
    if a.type != b.type:
        checker.error(op, f"select arms differ: {a.type} and {b.type}")
        return
    if isinstance(a.type, VectorType):
        if condition.type != VectorType(BOOL, a.type.count):
            checker.error(op, f"a vector select needs a vector<bool, {a.type.count}> condition")
    elif condition.type != BOOL:
        checker.error(op, f"a select condition is bool, not {condition.type}")
    _result_matches(op, checker, a.type)


def _verify_cast(op: Operation, checker: Checker) -> None:
    source = op.operands[0].type
    target = op.results[0].type
    if isinstance(source, VectorType) and isinstance(target, VectorType):
        if source.count != target.count:
            checker.error(op, f"a cast keeps the lane count: {source} to {target}")
            return
        source, target = source.element, target.element
    if isinstance(source, PtrType) and isinstance(target, PtrType):
        if source.address_space != target.address_space:
            checker.error(op, f"a cast keeps the address space: {source} to {target}")
        return
    if not (is_scalar(source) and is_scalar(target)):
        checker.error(op, f"no cast from {source} to {target}")


def _verify_successor_arguments(op: Operation, checker: Checker) -> None:
    for successor in op.successors:
        block = successor.block
        if len(successor.arguments) != len(block.arguments):
            checker.error(
                op,
                f"^{block.name} takes {len(block.arguments)} argument(s), "
                f"{len(successor.arguments)} passed",
            )
            continue
        for passed, argument in zip(successor.arguments, block.arguments, strict=True):
            if passed.type != argument.type:
                checker.error(
                    op,
                    f"^{block.name} argument {argument.index} is {argument.type}, "
                    f"{passed.type} passed",
                )


def _verify_br(op: Operation, checker: Checker) -> None:
    _verify_successor_arguments(op, checker)


def _verify_cond_br(op: Operation, checker: Checker) -> None:
    if op.operands[0].type != BOOL:
        checker.error(op, f"a branch condition is bool, not {op.operands[0].type}")
    _verify_successor_arguments(op, checker)


def _verify_ret(op: Operation, checker: Checker) -> None:
    function = checker.function
    if function is None:
        return
    given = tuple(v.type for v in op.operands)
    if given != function.results:
        checker.error(
            op,
            f"returns ({', '.join(map(str, given))}), "
            f"@{function.name} declares ({', '.join(map(str, function.results))})",
        )
    for value in op.operands:
        if isinstance(value.type, PtrType) and value.type.address_space == "stack":
            checker.error(op, "a stack pointer escapes through the return")
        if _borrowed_parameter(value, checker) is not None:
            checker.error(op, f"%{value.name} is borrowed and may not be returned")


def _borrowed_parameter(value: Value, checker: Checker) -> str | None:
    """The ownership a parameter declares, when `value` is that parameter and
    the declaration is a borrow."""
    function = checker.function
    if function is None or function.entry is None:
        return None
    for argument, attributes in zip(
        function.entry.arguments, function.param_attributes, strict=False
    ):
        if argument is value and attributes.get("ownership") in {"borrowed", "mut"}:
            return str(attributes["ownership"])
    return None


def _verify_alloca(op: Operation, checker: Checker) -> None:
    t = op.results[0].type
    if not isinstance(t, PtrType) or t.address_space != "stack":
        checker.error(op, f"alloca yields ptr<T, stack>, not {t}")
    count = op.attributes.get("count", 1)
    if not isinstance(count, int) or count < 1:
        checker.error(op, f"alloca count must be a positive integer, not {count!r}")


def _verify_load(op: Operation, checker: Checker) -> None:
    pointer = op.operands[0].type
    if not isinstance(pointer, PtrType):
        checker.error(op, f"load reads through a pointer, not {pointer}")
        return
    _result_matches(op, checker, pointer.pointee)


def _verify_store(op: Operation, checker: Checker) -> None:
    value, pointer = op.operands
    if not isinstance(pointer.type, PtrType):
        checker.error(op, f"store writes through a pointer, not {pointer.type}")
        return
    if pointer.type.pointee != value.type:
        checker.error(op, f"stores {value.type} through {pointer.type}")
    if not pointer.type.mutable:
        checker.error(op, "writes through a const pointer")
    # A store into the function's own stack -- a local's slot -- keeps the
    # value inside the call; anywhere else is memory that may outlive it.
    if pointer.type.address_space != "stack":
        if isinstance(value.type, PtrType) and value.type.address_space == "stack":
            checker.error(op, "a stack pointer escapes into memory")
        if _borrowed_parameter(value, checker) is not None:
            checker.error(op, f"%{value.name} is borrowed and may not be stored")


def _verify_ptr_offset(op: Operation, checker: Checker) -> None:
    pointer, offset = op.operands
    if not isinstance(pointer.type, PtrType):
        checker.error(op, f"ptr_offset moves a pointer, not {pointer.type}")
        return
    if not is_integer(offset.type):
        checker.error(op, f"an offset is an integer, not {offset.type}")
    _result_matches(op, checker, pointer.type)


def _verify_buffer_data(op: Operation, checker: Checker) -> None:
    buffer = op.operands[0].type
    if not isinstance(buffer, BufferType):
        checker.error(op, f"buffer_data needs a buffer, not {buffer}")
        return
    result = op.results[0].type
    if not isinstance(result, PtrType) or result.pointee != buffer.element:
        checker.error(op, f"buffer_data yields ptr<{buffer.element}>, not {result}")
    elif result.address_space != "generic":
        checker.error(op, "a buffer's data lives in the generic address space")


def _verify_buffer_len(op: Operation, checker: Checker) -> None:
    if not isinstance(op.operands[0].type, BufferType):
        checker.error(op, f"buffer_len needs a buffer, not {op.operands[0].type}")
    _result_matches(op, checker, INDEX)


def _verify_buffer_load(op: Operation, checker: Checker) -> None:
    buffer, index = op.operands
    if not isinstance(buffer.type, BufferType):
        checker.error(op, f"buffer_load needs a buffer, not {buffer.type}")
        return
    if not is_integer(index.type):
        checker.error(op, f"a buffer index is an integer, not {index.type}")
    _result_matches(op, checker, buffer.type.element)


def _verify_buffer_store(op: Operation, checker: Checker) -> None:
    value, buffer, index = op.operands
    if not isinstance(buffer.type, BufferType):
        checker.error(op, f"buffer_store needs a buffer, not {buffer.type}")
        return
    if not is_integer(index.type):
        checker.error(op, f"a buffer index is an integer, not {index.type}")
    if buffer.type.element != value.type:
        checker.error(op, f"stores {value.type} into {buffer.type}")


def _verify_tuple_make(op: Operation, checker: Checker) -> None:
    expected = TupleType(tuple(v.type for v in op.operands))
    _result_matches(op, checker, expected)


def _verify_tuple_extract(op: Operation, checker: Checker) -> None:
    source = op.operands[0].type
    index = op.attributes.get("index")
    if not isinstance(source, TupleType):
        checker.error(op, f"tuple_extract needs a tuple, not {source}")
        return
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(source.items):
        checker.error(op, f"index {index!r} is outside {source}")
        return
    _result_matches(op, checker, source.items[index])


def _verify_struct_make(op: Operation, checker: Checker) -> None:
    result = op.results[0].type
    if not isinstance(result, StructType):
        checker.error(op, f"struct_make yields a struct, not {result}")
        return
    given = tuple(v.type for v in op.operands)
    expected = tuple(t for _name, t in result.fields)
    if given != expected:
        checker.error(
            op,
            f"{result} takes ({', '.join(map(str, expected))}), "
            f"given ({', '.join(map(str, given))})",
        )


def _verify_struct_extract(op: Operation, checker: Checker) -> None:
    source = op.operands[0].type
    name = op.attributes.get("field")
    if not isinstance(source, StructType):
        checker.error(op, f"struct_extract needs a struct, not {source}")
        return
    if not isinstance(name, str) or source.field_type(name) is None:
        checker.error(op, f"{source} has no field {name!r}")
        return
    _result_matches(op, checker, source.field_type(name))  # type: ignore[arg-type]


def _verify_call(op: Operation, checker: Checker) -> None:
    callee = op.attributes.get("callee")
    if not isinstance(callee, SymbolRef):
        checker.error(op, "call needs a `callee` symbol")
        return
    module = checker.module
    if module is None:
        return
    target = module.functions.get(callee.name)
    if target is None:
        checker.error(op, f"call to undefined @{callee.name}")
        return
    given = tuple(v.type for v in op.operands)
    expected = tuple(t for _name, t in target.params)
    if given != expected:
        checker.error(
            op,
            f"@{callee.name} takes ({', '.join(map(str, expected))}), "
            f"called with ({', '.join(map(str, given))})",
        )
    results = tuple(r.type for r in op.results)
    expected_results = tuple(target.results)
    if op.attributes.get("capture_status"):
        # The callee's status comes back as a trailing i64 instead of
        # sending this function to its fallback.
        expected_results = (*expected_results, I64)
    if results != expected_results:
        checker.error(
            op,
            f"@{callee.name} returns ({', '.join(map(str, expected_results))}), "
            f"call yields ({', '.join(map(str, results))})",
        )


def _verify_call_extern(op: Operation, checker: Checker) -> None:
    callee = op.attributes.get("callee")
    if not isinstance(callee, str) or not callee:
        checker.error(op, "call_extern needs a `callee` name")


def _verify_guard(op: Operation, checker: Checker) -> None:
    if op.operands[0].type != BOOL:
        checker.error(op, f"a guard condition is bool, not {op.operands[0].type}")
    kind = op.attributes.get("kind")
    if kind not in GUARD_KINDS:
        checker.error(op, f"guard kind {kind!r} is not one of {GUARD_KINDS}")


# -- construction -------------------------------------------------------------
# Typed constructors over `Builder.create`, so a frontend never spells an
# operation name or infers a result type by hand.


def const(b: Builder, value: Attribute, t: IRType, name: str | None = None) -> Value:
    return b.create("core.const", (), (t,), {"value": value}, result_names=(name,)).result


def _binary(
    b: Builder, op: str, lhs: Value, rhs: Value, attributes: dict[str, Attribute], name: str | None
) -> Value:
    return b.create(op, (lhs, rhs), (lhs.type,), attributes, result_names=(name,)).result


def _int_attrs(t: IRType, overflow: str, **more: Attribute) -> dict[str, Attribute]:
    attributes: dict[str, Attribute] = dict(more)
    if is_integer(scalar_of(t)):
        attributes["overflow"] = overflow
    return attributes


def add(
    b: Builder, lhs: Value, rhs: Value, *, overflow: str = "python", name: str | None = None
) -> Value:
    return _binary(b, "core.add", lhs, rhs, _int_attrs(lhs.type, overflow), name)


def sub(
    b: Builder, lhs: Value, rhs: Value, *, overflow: str = "python", name: str | None = None
) -> Value:
    return _binary(b, "core.sub", lhs, rhs, _int_attrs(lhs.type, overflow), name)


def mul(
    b: Builder, lhs: Value, rhs: Value, *, overflow: str = "python", name: str | None = None
) -> Value:
    return _binary(b, "core.mul", lhs, rhs, _int_attrs(lhs.type, overflow), name)


def div(
    b: Builder,
    lhs: Value,
    rhs: Value,
    *,
    overflow: str = "python",
    rounding: str = "floor",
    name: str | None = None,
) -> Value:
    attributes = _int_attrs(lhs.type, overflow)
    if is_integer(scalar_of(lhs.type)):
        attributes["rounding"] = rounding
    return _binary(b, "core.div", lhs, rhs, attributes, name)


def mod(
    b: Builder,
    lhs: Value,
    rhs: Value,
    *,
    overflow: str = "python",
    rounding: str = "floor",
    name: str | None = None,
) -> Value:
    attributes = _int_attrs(lhs.type, overflow)
    if is_integer(scalar_of(lhs.type)):
        attributes["rounding"] = rounding
    return _binary(b, "core.mod", lhs, rhs, attributes, name)


def neg(b: Builder, value: Value, *, overflow: str = "python", name: str | None = None) -> Value:
    return b.create(
        "core.neg", (value,), (value.type,), _int_attrs(value.type, overflow), result_names=(name,)
    ).result


def bitwise(b: Builder, op: str, lhs: Value, rhs: Value, name: str | None = None) -> Value:
    return _binary(b, f"core.{op}", lhs, rhs, {}, name)


def shift(b: Builder, op: str, value: Value, amount: Value, name: str | None = None) -> Value:
    return b.create(f"core.{op}", (value, amount), (value.type,), result_names=(name,)).result


def cmp(b: Builder, predicate: str, lhs: Value, rhs: Value, name: str | None = None) -> Value:
    result: IRType = VectorType(BOOL, lhs.type.count) if isinstance(lhs.type, VectorType) else BOOL
    return b.create(
        "core.cmp", (lhs, rhs), (result,), {"predicate": predicate}, result_names=(name,)
    ).result


def select(b: Builder, condition: Value, a: Value, c: Value, name: str | None = None) -> Value:
    return b.create("core.select", (condition, a, c), (a.type,), result_names=(name,)).result


def cast(b: Builder, value: Value, t: IRType, name: str | None = None) -> Value:
    return b.create("core.cast", (value,), (t,), result_names=(name,)).result


def br(b: Builder, target: Successor) -> Operation:
    return b.create("core.br", successors=(target,))


def cond_br(b: Builder, condition: Value, then: Successor, otherwise: Successor) -> Operation:
    return b.create("core.cond_br", (condition,), successors=(then, otherwise))


def ret(b: Builder, *values: Value) -> Operation:
    return b.create("core.ret", values)


def unreachable(b: Builder) -> Operation:
    return b.create("core.unreachable")


def alloca(b: Builder, t: IRType, count: int = 1, name: str | None = None) -> Value:
    attributes: dict[str, Attribute] = {"count": count} if count != 1 else {}
    return b.create(
        "core.alloca", (), (PtrType(t, "stack"),), attributes, result_names=(name,)
    ).result


def load(b: Builder, pointer: Value, name: str | None = None) -> Value:
    assert isinstance(pointer.type, PtrType)
    return b.create("core.load", (pointer,), (pointer.type.pointee,), result_names=(name,)).result


def store(b: Builder, value: Value, pointer: Value) -> Operation:
    return b.create("core.store", (value, pointer))


def ptr_offset(b: Builder, pointer: Value, offset: Value, name: str | None = None) -> Value:
    return b.create(
        "core.ptr_offset", (pointer, offset), (pointer.type,), result_names=(name,)
    ).result


def buffer_data(b: Builder, buffer: Value, name: str | None = None) -> Value:
    assert isinstance(buffer.type, BufferType)
    return b.create(
        "core.buffer_data", (buffer,), (PtrType(buffer.type.element),), result_names=(name,)
    ).result


def buffer_len(b: Builder, buffer: Value, name: str | None = None) -> Value:
    return b.create("core.buffer_len", (buffer,), (INDEX,), result_names=(name,)).result


def buffer_load(b: Builder, buffer: Value, index: Value, name: str | None = None) -> Value:
    assert isinstance(buffer.type, BufferType)
    return b.create(
        "core.buffer_load", (buffer, index), (buffer.type.element,), result_names=(name,)
    ).result


def buffer_store(b: Builder, value: Value, buffer: Value, index: Value) -> Operation:
    return b.create("core.buffer_store", (value, buffer, index))


def tuple_make(b: Builder, *items: Value, name: str | None = None) -> Value:
    return b.create(
        "core.tuple_make", items, (TupleType(tuple(i.type for i in items)),), result_names=(name,)
    ).result


def tuple_extract(b: Builder, value: Value, index: int, name: str | None = None) -> Value:
    assert isinstance(value.type, TupleType)
    return b.create(
        "core.tuple_extract",
        (value,),
        (value.type.items[index],),
        {"index": index},
        result_names=(name,),
    ).result


def struct_make(b: Builder, t: StructType, *fields: Value, name: str | None = None) -> Value:
    return b.create("core.struct_make", fields, (t,), result_names=(name,)).result


def struct_extract(b: Builder, value: Value, field: str, name: str | None = None) -> Value:
    assert isinstance(value.type, StructType)
    field_type = value.type.field_type(field)
    assert field_type is not None
    return b.create(
        "core.struct_extract", (value,), (field_type,), {"field": field}, result_names=(name,)
    ).result


def call(
    b: Builder,
    callee: str,
    operands: tuple[Value, ...],
    results: tuple[IRType, ...],
    names: tuple[str | None, ...] = (),
    *,
    capture_status: bool = False,
) -> Operation:
    """Call `@callee`. A failed call takes this function's fallback -- unless
    `capture_status`, when the status is the call's trailing i64 result and
    the caller decides, which a caller with threads to join first needs."""
    attributes: dict[str, Attribute] = {"callee": SymbolRef(callee)}
    if capture_status:
        attributes["capture_status"] = True
        results = (*results, I64)
    return b.create("core.call", operands, results, attributes, result_names=names)


def call_extern(
    b: Builder,
    callee: str,
    operands: tuple[Value, ...],
    results: tuple[IRType, ...],
    *,
    abi: str = "c",
    names: tuple[str | None, ...] = (),
) -> Operation:
    return b.create(
        "core.call_extern", operands, results, {"callee": callee, "abi": abi}, result_names=names
    )


def call_intrinsic(
    b: Builder,
    intrinsic: str,
    operands: tuple[Value, ...],
    results: tuple[IRType, ...],
    names: tuple[str | None, ...] = (),
) -> Operation:
    return b.create(
        "core.call_intrinsic", operands, results, {"intrinsic": intrinsic}, result_names=names
    )


def guard(b: Builder, condition: Value, kind: str, message: str = "", label: str = "") -> Operation:
    attributes: dict[str, Attribute] = {"kind": kind}
    if message:
        attributes["message"] = message
    if label:
        attributes["label"] = label
    return b.create("core.guard", (condition,), (), attributes)


def checked(b: Builder, op: str, lhs: Value, rhs: Value) -> tuple[Value, Value]:
    """`lhs op rhs` as (the wrapped result, whether it overflowed): what a
    corner check asks, one branch for every corner rather than one each."""
    created = b.create(
        "core.call_intrinsic",
        (lhs, rhs),
        (lhs.type, BOOL),
        {"intrinsic": f"ppy.checked_{op}"},
    )
    return created.results[0], created.results[1]
