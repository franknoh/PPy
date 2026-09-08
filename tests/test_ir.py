"""The typed canonical IR: construction, verification, text, and the codec."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppy_compiler.ir import (
    BOOL,
    F64,
    I8,
    I32,
    I64,
    INDEX,
    U8,
    BufferType,
    Builder,
    CodecError,
    Dialect,
    DialectRegistry,
    DialectType,
    IRModule,
    Operation,
    OpSpec,
    ParseError,
    PtrType,
    StructType,
    Successor,
    TupleType,
    VectorType,
    decode,
    encode,
    parse_module,
    parse_type,
    print_module,
    registry,
    verify,
    verify_or_raise,
)
from ppy_compiler.ir.dialects import core
from ppy_compiler.ir.types import TypeError_
from ppy_compiler.ir.verify import VerificationError

ABS_TEXT = """\
ppyir 1
module @demo
dialect core 1

func @abs(%x: i64) -> i64 {
^entry:
    %zero = core.const 0 : i64
    %negative = core.cmp.lt %x, %zero : bool
    core.cond_br %negative, ^neg, ^positive
^neg:
    %value = core.neg %x {overflow = "python"} : i64
    core.br ^exit(%value)
^positive:
    core.br ^exit(%x)
^exit(%result: i64):
    core.ret %result
}
"""


def _abs_module() -> IRModule:
    module = IRModule("demo")
    function = module.add_function("abs", [("x", I64)], [I64])
    entry = function.add_entry_block()
    negative_block = function.body.add_block("neg")
    positive_block = function.body.add_block("positive")
    exit_block = function.body.add_block("exit", [("result", I64)])
    b = Builder(entry)
    x = entry.arguments[0]
    zero = core.const(b, 0, I64, "zero")
    negative = core.cmp(b, "lt", x, zero, "negative")
    core.cond_br(b, negative, Successor(negative_block), Successor(positive_block))
    b.at_end(negative_block)
    value = core.neg(b, x, name="value")
    core.br(b, Successor(exit_block, [value]))
    b.at_end(positive_block)
    core.br(b, Successor(exit_block, [x]))
    b.at_end(exit_block)
    core.ret(b, exit_block.arguments[0])
    return module


def _function(module: IRModule, name: str = "f", params=(("x", I64),), results=(I64,)):
    function = module.add_function(name, list(params), list(results))
    entry = function.add_entry_block()
    return function, entry, Builder(entry)


def _errors(module: IRModule) -> list[str]:
    return [error.message for error in verify(module)]


# -- construction and text --------------------------------------------------


def test_the_directive_example_builds_verifies_and_prints_to_one_spelling():
    module = _abs_module()
    assert not verify(module)
    assert encode(module) == ABS_TEXT


def test_the_text_reads_back_to_the_same_text_and_verifies():
    module = decode(ABS_TEXT)
    assert not verify(module)
    assert encode(module) == ABS_TEXT
    function = module.functions["abs"]
    assert [block.name for block in function.blocks()] == ["entry", "neg", "positive", "exit"]
    assert function.entry is not None
    assert function.entry.arguments[0].name == "x"


def test_names_are_deterministic_and_never_collide():
    """A value keeps its hint unless an earlier value took it; the rest are numbered."""
    module = IRModule("names")
    _fn, entry, b = _function(module)
    x = entry.arguments[0]
    a = core.add(b, x, x, name="x")  # the parameter already took %x
    c = core.add(b, a, x)  # no hint at all
    d = core.add(b, c, x, name="d")
    core.ret(b, d)
    text = encode(module)
    body = [line.strip() for line in text.splitlines() if "core." in line]
    assert body[0].startswith("%0 = core.add %x, %x")
    assert body[1].startswith("%1 = core.add %0, %x")
    assert body[2].startswith("%d = core.add %1, %x")
    assert encode(module) == text, "printing is a pure function of the module"
    assert encode(decode(text)) == text


def test_attributes_print_sorted_and_read_back_exactly():
    module = IRModule("attrs")
    module.attributes.update({"zeta": (1, 2.5, "s"), "alpha": {"n": True, "t": I32}})
    function, entry, b = _function(module)
    function.attributes["effects"] = ("alloc", "read_memory")
    function.param_attributes[0]["ownership"] = "borrowed"
    core.ret(b, entry.arguments[0])
    text = encode(module)
    assert 'attrs {alpha = {n = true, t = !i32}, zeta = [1, 2.5, "s"]}' in text
    assert (
        'func @f(%x: i64 {ownership = "borrowed"}) -> i64 attrs {effects = ["alloc", "read_memory"]}'
        in text
    )
    again = decode(text)
    assert again.attributes == module.attributes
    assert again.functions["f"].attributes == function.attributes
    assert again.functions["f"].param_attributes == function.param_attributes


def test_floats_symbols_locations_and_globals_round_trip():
    from ppy_compiler.ir import SourceLocation, SymbolRef

    module = IRModule("misc")
    module.add_global("scale", F64, 2.0)
    module.add_global("counter", I64, 0, constant=False, visibility="private")
    callee = module.add_function("twice", [("v", F64)], [F64], visibility="private")
    entry = callee.add_entry_block()
    b = Builder(entry, SourceLocation("m.ppy", 3, 4))
    two = core.const(b, 2.0, F64)
    core.ret(b, core.mul(b, entry.arguments[0], two))
    _fn, entry, b = _function(module, "f", (("x", F64),), (F64,))
    result = core.call(b, "twice", (entry.arguments[0],), (F64,)).results[0]
    inf = core.const(b, float("inf"), F64)
    core.ret(b, core.add(b, result, inf))
    assert not verify(module)
    text = encode(module)
    assert "global @scale : f64 = 2.0" in text
    assert "private mutable global @counter : i64 = 0" in text
    assert 'loc("m.ppy":3:4)' in text
    assert "core.const inf : f64" in text
    again = decode(text)
    assert encode(again) == text
    assert next(again.functions["f"].operations()).attributes["callee"] == SymbolRef("twice")


def test_every_type_spelling_parses_to_itself():
    spellings = [
        "void",
        "bool",
        "i8",
        "u64",
        "f16",
        "index",
        "ptr<i64>",
        "ptr<f64, stack>",
        "ptr<f64, generic, const>",
        "ptr<i8, global, const>",
        "buffer<f64>",
        "vector<f32, 8>",
        "tuple<>",
        "tuple<i64, f64>",
        "struct<Point, x: f64, y: f64>",
        "future<i64>",
        "tensor.tensor<f32, 4, dynamic>",
        "arrow.array<i64>",
    ]
    for spelling in spellings:
        assert str(parse_type(spelling)) == spelling
    assert parse_type("ptr<i64, generic, mut>") == PtrType(I64)
    assert parse_type("tensor.tensor<f32, 4, dynamic>") == DialectType(
        "tensor", "tensor", (parse_type("f32"), 4, "dynamic")
    )
    for bad in ("i7", "vector<bool>", "ptr<>", "i64 extra", "vector<buffer<i64>, 4>"):
        with pytest.raises(TypeError_):
            parse_type(bad)


# -- verification -----------------------------------------------------------


def test_a_use_before_its_definition_is_an_error():
    module = IRModule("ssa")
    function, entry, b = _function(module)
    later = function.body.add_block("later")
    core.br(b, Successor(later))
    b.at_end(later)
    x = entry.arguments[0]
    one = core.const(b, 1, I64, "one")
    core.ret(b, core.add(b, x, one))
    assert not verify(module)
    # Move the constant after its use: the read no longer dominates.
    one_op = one.owner
    later.remove(one_op)
    later.insert(len(later.operations) - 1, one_op)
    assert _errors(module) == ["%one is used before it is defined"]


def test_a_value_defined_on_one_branch_is_not_visible_on_the_other():
    module = IRModule("dom")
    function, entry, b = _function(module)
    then = function.body.add_block("then")
    otherwise = function.body.add_block("else")
    x = entry.arguments[0]
    zero = core.const(b, 0, I64)
    core.cond_br(b, core.cmp(b, "lt", x, zero), Successor(then), Successor(otherwise))
    b.at_end(then)
    doubled = core.add(b, x, x, name="doubled")
    core.ret(b, doubled)
    b.at_end(otherwise)
    core.ret(b, doubled)
    assert _errors(module) == ["%doubled is used before it is defined"]


def test_duplicate_blocks_are_errors_and_duplicate_name_hints_print_apart():
    """A name hint may repeat -- the printer numbers the later one -- but a
    block label is an identity and may not."""
    module = IRModule("dup")
    function, _entry, b = _function(module)
    a = core.const(b, 1, I64, "same")
    c = core.const(b, 2, I64, "same")
    core.ret(b, core.add(b, a, c))
    assert not verify(module)
    text = encode(module)
    assert "%same = core.const 1" in text and "%0 = core.const 2" in text
    function.body.add_block("entry").append(Operation("core.unreachable"))
    assert "block ^entry is defined twice" in _errors(module)


def test_block_structure_errors_name_the_block():
    module = IRModule("blocks")
    function, entry, b = _function(module)
    target = function.body.add_block("target", [("v", I64)])
    core.br(b, Successor(target))  # no argument for %v
    b.at_end(target)
    core.ret(b, target.arguments[0])
    assert _errors(module) == ["^target takes 1 argument(s), 0 passed"]
    entry.operations[-1].erase()
    core.br(Builder(entry), Successor(target, [core.const(Builder(entry), 1.0, F64)]))
    assert _errors(module) == ["^target argument 0 is i64, f64 passed"]
    empty = function.body.add_block("empty")
    assert "block ^empty is empty: it needs a terminator" in _errors(module)
    empty.append(Operation("core.const", (), (I64,), {"value": 1}))
    assert "block ^empty does not end in a terminator" in _errors(module)
    core.ret(Builder(empty), empty.operations[0].results[0])
    Builder().before(entry.operations[-1]).create("core.unreachable")
    assert "a terminator in the middle of a block" in _errors(module)


def test_a_branch_may_not_leave_its_region():
    module = IRModule("regions")
    _fn, _entry, b = _function(module)
    other = module.add_function("g", [], [I64])
    other_entry = other.add_entry_block()
    core.ret(Builder(other_entry), core.const(Builder(other_entry), 1, I64))
    core.br(b, Successor(other_entry))
    assert "branch to ^entry leaves the region" in _errors(module)


def test_return_and_call_signatures_are_checked():
    module = IRModule("calls")
    _fn, entry, b = _function(module)
    x = entry.arguments[0]
    core.ret(b, core.cast(b, x, F64))
    assert _errors(module) == ["returns (f64), @f declares (i64)"]
    entry.operations[-1].erase()
    b = Builder(entry)
    missing = core.call(b, "nope", (x,), (I64,)).results[0]
    core.ret(b, missing)
    assert _errors(module) == ["call to undefined @nope"]
    module.add_function("nope", [("a", F64)], [F64])
    assert _errors(module) == [
        "@nope takes (f64), called with (i64)",
        "@nope returns (f64), call yields (i64)",
    ]


def test_pointer_rules_pointee_constness_address_space_and_escape():
    module = IRModule("pointers")
    _fn, entry, b = _function(
        module, params=(("p", PtrType(I64)), ("c", PtrType(I64, mutable=False))), results=(I64,)
    )
    p, c = entry.arguments
    one = core.const(b, 1.0, F64)
    b.create("core.store", (one, p))
    assert "stores f64 through ptr<i64>" in _errors(module)
    entry.operations[-1].erase()
    core.store(b, core.const(b, 1, I64), c)
    assert "writes through a const pointer" in _errors(module)
    entry.operations[-1].erase()
    b.create("core.load", (p,), (F64,))
    assert "result is f64, expected i64" in _errors(module)
    entry.operations[-1].erase()
    b.create("core.cast", (p,), (PtrType(I64, "stack"),))
    assert "a cast keeps the address space: ptr<i64> to ptr<i64, stack>" in _errors(module)
    entry.operations[-1].erase()
    stack = core.alloca(b, I64)
    b.create(
        "core.store", (stack, b.create("core.cast", (p,), (PtrType(PtrType(I64, "stack")),)).result)
    )
    assert "a stack pointer escapes into memory" in _errors(module)
    entry.operations[-1].erase()
    entry.operations[-1].erase()
    core.ret(b, core.load(b, p))
    assert not verify(module)
    escaping = module.add_function("escape", [], [PtrType(I64, "stack")])
    b = Builder(escaping.add_entry_block())
    core.ret(b, core.alloca(b, I64))
    assert "a stack pointer escapes through the return" in _errors(module)
    b.create("core.load", (core.const(b, 1, I64),), (I64,))
    assert "load reads through a pointer, not i64" in _errors(module)


def test_arithmetic_needs_its_overflow_and_rounding_semantics_spelled():
    module = IRModule("arith")
    _fn, entry, b = _function(module)
    x = entry.arguments[0]
    bare = b.create("core.add", (x, x), (I64,))
    core.ret(b, bare.result)
    assert _errors(module) == [
        "integer core.add needs overflow in ('python', 'checked', 'wrap', 'proven')"
    ]
    bare.attributes["overflow"] = "wrap"
    assert not verify(module)
    entry.operations[-1].erase()
    entry.operations[-1].erase()
    b = Builder(entry)
    quotient = b.create("core.div", (x, x), (I64,), {"overflow": "python"})
    core.ret(b, quotient.result)
    assert _errors(module) == ["integer core.div needs rounding in ('floor', 'trunc')"]
    quotient.attributes["rounding"] = "floor"
    assert not verify(module)
    f = module.add_function("fl", [("a", F64)], [F64])
    fb = Builder(f.add_entry_block())
    core.ret(fb, core.div(fb, f.entry.arguments[0], f.entry.arguments[0]))
    assert not verify(module), "floats carry no overflow attribute"


def test_type_mismatches_across_operands_results_and_vectors():
    module = IRModule("types")
    _fn, entry, b = _function(module, params=(("x", I64), ("y", F64)), results=(I64,))
    x, y = entry.arguments
    mixed = b.create("core.add", (x, y), (I64,), {"overflow": "wrap"})
    core.ret(b, mixed.result)
    assert _errors(module) == ["operands differ in type: f64, i64"]
    entry.operations[0].erase()
    entry.operations[0].erase()
    b = Builder(entry)
    v4 = core.const(b, 0, I64)  # placeholder to build vectors from casts
    wide = b.create("core.cast", (v4,), (VectorType(I64, 4),))
    narrow = b.create("core.cast", (v4,), (VectorType(I64, 2),))
    b.create("core.add", (wide.result, narrow.result), (VectorType(I64, 4),), {"overflow": "wrap"})
    core.ret(b, x)
    errors = _errors(module)
    assert "no cast from i64 to vector<i64, 4>" in errors
    assert "operands differ in type: vector<i64, 2>, vector<i64, 4>" in errors
    lanes = b.create("core.cast", (wide.result,), (VectorType(F64, 2),))
    assert "a cast keeps the lane count: vector<i64, 4> to vector<f64, 2>" in _errors(module)
    lanes.erase()
    cmp = b.create("core.cmp", (wide.result, wide.result), (BOOL,), {"predicate": "eq"})
    assert "result is bool, expected vector<bool, 4>" in _errors(module)
    cmp.erase()
    b.create("core.select", (core.const(b, True, BOOL), x, y), (I64,))
    assert "select arms differ: i64 and f64" in _errors(module)


def test_constants_must_fit_their_type():
    module = IRModule("consts")
    _fn, entry, b = _function(module, params=(), results=(I8,))
    too_big = core.const(b, 200, I8)
    core.ret(b, too_big)
    assert _errors(module) == ["200 does not fit i8"]
    entry.operations[0].attributes["value"] = -128
    assert not verify(module)
    b.create("core.const", (), (U8,), {"value": -1})
    assert "-1 does not fit u8" in _errors(module)
    b.create("core.const", (), (BOOL,), {"value": 1})
    assert "a bool constant needs a bool value, not 1" in _errors(module)
    b.create("core.const", (), (F64,), {"value": "x"})
    assert "a float constant needs a number, not 'x'" in _errors(module)


def test_buffers_tuples_and_structs():
    point = StructType("Point", (("x", F64), ("y", F64)))
    module = IRModule("aggregates")
    _fn, entry, b = _function(
        module, params=(("buf", BufferType(F64)), ("i", INDEX)), results=(TupleType((F64, F64)),)
    )
    buf, i = entry.arguments
    length = core.buffer_len(b, buf)
    core.guard(b, core.cmp(b, "lt", i, length), "bounds", "index out of range")
    first = core.buffer_load(b, buf, i)
    core.buffer_store(b, first, buf, i)
    data = core.buffer_data(b, buf)
    second = core.load(b, core.ptr_offset(b, data, i))
    made = core.struct_make(b, point, first, second)
    y = core.struct_extract(b, made, "y")
    pair = core.tuple_make(b, first, y)
    core.ret(b, pair)
    assert not verify(module)
    text = encode(module)
    assert "core.struct_make %" in text and ": struct<Point, x: f64, y: f64>" in text
    assert encode(decode(text)) == text
    b = Builder().before(entry.operations[-1])
    bad = b.create("core.tuple_extract", (pair,), (F64,), {"index": 2})
    assert "index 2 is outside tuple<f64, f64>" in _errors(module)
    bad.erase()
    bad = b.create("core.struct_extract", (made,), (F64,), {"field": "z"})
    assert "struct<Point, x: f64, y: f64> has no field 'z'" in _errors(module)
    bad.erase()
    bad = b.create("core.buffer_load", (buf, core.const(b, 1.0, F64)), (F64,))
    assert "a buffer index is an integer, not f64" in _errors(module)
    bad.erase()
    bad = b.create("core.guard", (first,), (), {"kind": "bounds"})
    assert "a guard condition is bool, not f64" in _errors(module)
    bad.erase()
    b.create("core.guard", (core.cmp(b, "eq", i, i),), (), {"kind": "vibes"})
    assert "guard kind 'vibes' is not one of" in _errors(module)[0]


def test_operation_shape_is_checked_against_its_spec():
    module = IRModule("shapes")
    _fn, entry, b = _function(module)
    x = entry.arguments[0]
    core.ret(b, x)
    b = Builder().before(entry.operations[-1])
    bad = b.create("core.neg", (x, x), (I64,), {"overflow": "wrap"})
    assert "takes 1 operand(s), given 2" in _errors(module)
    bad.erase()
    bad = b.create("core.cmp", (x, x), (BOOL,), {"predicate": "approximately"})
    assert "predicate 'approximately' is not one of" in _errors(module)[0]
    bad.erase()
    bad = b.create("core.tuple_extract", (x,), (I64,))
    assert "needs attribute 'index'" in _errors(module)
    bad.erase()
    b.create("core.const", (), (I64,), {"value": 1}).successors.append(Successor(entry))
    assert "only a terminator has successors" in _errors(module)


def test_unknown_dialects_and_versions_are_refused():
    module = IRModule("dialects")
    _fn, entry, b = _function(module)
    x = entry.arguments[0]
    core.ret(b, x)
    b = Builder().before(entry.operations[-1])
    bad = b.create("tpu.barrier", (x,), (I64,))
    assert _errors(module) == ["unknown dialect 'tpu'"]
    bad.erase()
    bad = b.create("core.frobnicate", (x,), (I64,))
    assert _errors(module) == ["dialect 'core' defines no operation 'frobnicate'"]
    bad.erase()
    module.dialects["core"] = 99
    assert "dialect 'core' version 99 is newer than the registered version 1" in _errors(module)
    module.dialects = {"nope": 1}
    assert "unknown dialect 'nope'" in _errors(module)
    assert "the module does not declare dialect 'core'" in _errors(module)
    with pytest.raises(VerificationError, match="unknown dialect"):
        verify_or_raise(module)


def test_a_registered_dialect_owns_its_operations_and_types():
    class ToyDialect(Dialect):
        name = "toy"
        version = 3

        def register_operations(self, registry: DialectRegistry) -> None:
            registry.add_op(OpSpec("toy.twice", pure=True, operands=1, results=1))

        def verify_type(self, t: DialectType) -> str | None:
            return None if t.name == "thing" else f"toy has no type {t.name!r}"

        def address_spaces(self) -> frozenset[str]:
            return frozenset({"toybox"})

    own = DialectRegistry()
    own.register(core.CoreDialect())
    own.register(ToyDialect())
    module = IRModule("toy", {"core": 1, "toy": 3})
    function, entry, b = _function(
        module,
        params=(("t", DialectType("toy", "thing")), ("p", PtrType(I64, "toybox"))),
        results=(I64,),
    )
    twice = b.create("toy.twice", (entry.arguments[0],), (I64,))
    core.load(b, entry.arguments[1])
    core.ret(b, twice.result)
    assert not verify(module, own)
    assert "unknown dialect 'toy'" in _errors(module), "the default registry has no toy"
    entry.arguments[1].type = PtrType(I64, "attic")
    function.params = (function.params[0], ("p", PtrType(I64, "attic")))
    assert "unknown address space 'attic'" in [e.message for e in verify(module, own)]
    function.params = (("t", DialectType("toy", "gadget")), function.params[1])
    assert "toy has no type 'gadget'" in [e.message for e in verify(module, own)]
    text = encode(module, own)
    assert "dialect toy 3" in text
    with pytest.raises(CodecError, match="dialect 'toy', which this compiler does not have"):
        decode(text)
    assert encode(decode(text, own), own) == text
    with pytest.raises(ValueError, match="already registered"):
        own.register(type("Other", (Dialect,), {"name": "toy"})())


# -- codec ------------------------------------------------------------------


def test_the_codec_refuses_what_it_cannot_read(tmp_path: Path):
    with pytest.raises(CodecError, match=r"not a \.ppyir file"):
        decode("module @x\n")
    with pytest.raises(CodecError, match="schema 7; this compiler reads schema 1"):
        decode("ppyir 7\nmodule @x\n")
    with pytest.raises(CodecError, match="dialect 'core' at version 9"):
        decode("ppyir 1\nmodule @x\ndialect core 9\n")
    with pytest.raises(CodecError, match="line 6: %y is not defined"):
        decode(
            "ppyir 1\nmodule @x\ndialect core 1\nfunc @f() -> i64 {\n^entry:\n    core.ret %y\n}\n"
        )
    with pytest.raises(CodecError, match="cannot read"):
        from ppy_compiler.ir import read

        read(tmp_path / "missing.ppyir")


def test_parse_errors_carry_the_line():
    with pytest.raises(ParseError, match="line 7: %x is defined twice"):
        parse_module(
            "ppyir 1\nmodule @x\ndialect core 1\nfunc @f() -> i64 {\n^entry:\n"
            "    %x = core.const 1 : i64\n    %x = core.const 2 : i64\n    core.ret %x\n}\n"
        )
    with pytest.raises(ParseError, match="branch to undefined block \\^nowhere"):
        parse_module(
            "ppyir 1\nmodule @x\ndialect core 1\nfunc @f() -> () {\n^entry:\n    core.br ^nowhere\n}\n"
        )
    with pytest.raises(ParseError, match="names 1 result\\(s\\) but gives 0 type"):
        parse_module(
            "ppyir 1\nmodule @x\ndialect core 1\nfunc @f() -> () {\n^entry:\n    %a = core.unreachable\n}\n"
        )


def test_a_written_module_reads_back_equal(tmp_path: Path):
    from ppy_compiler.ir import read, write

    module = _abs_module()
    path = tmp_path / "abs.ppyir"
    write(module, path)
    assert path.read_text(encoding="utf-8") == ABS_TEXT
    assert print_module(read(path)) == ABS_TEXT


def test_use_lists_follow_replacement_and_erasure():
    module = IRModule("uses")
    _fn, entry, b = _function(module)
    x = entry.arguments[0]
    one = core.const(b, 1, I64)
    two = core.const(b, 2, I64)
    total = core.add(b, x, one)
    core.ret(b, total)
    assert [op.name for op, _index in one.uses] == ["core.add"]
    one.replace_all_uses_with(two)
    assert one.uses == [] and [op.name for op, _i in two.uses] == ["core.add"]
    assert total.owner.operands[1] is two
    one.owner.erase()
    assert [op.name for op in entry.operations] == ["core.const", "core.add", "core.ret"]
    assert not verify(module)
    assert registry().op_spec("core.add") is not None and registry().op_spec("core.add").commutative


def test_an_operation_ends_at_its_line():
    """A bare `core.ret` before a block, or any operation with nothing after
    its name, leaves the next line's label or value alone."""
    module = IRModule("bare")
    f = module.add_function("f", [("x", F64)], [F64])
    entry = f.add_entry_block()
    other = f.body.add_block("other")
    core.ret(Builder(entry), entry.arguments[0])
    b = Builder(other)
    core.ret(b, core.const(b, 1.0, F64))
    g = module.add_function("g", [], [])
    core.ret(Builder(g.add_entry_block()))
    core.ret(Builder(g.body.add_block("later")))
    text = encode(module)
    assert "    core.ret %x\n^other:\n" in text and "    core.ret\n^later:\n" in text
    again = decode(text)
    assert encode(again) == text
    assert [block.name for block in again.functions["g"].blocks()] == ["entry", "later"]
    assert not verify(again)
