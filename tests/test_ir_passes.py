"""Patterns, passes, and the pass manager over the canonical IR."""

from __future__ import annotations

import pytest

from ppy_compiler.ir import (
    BOOL,
    F64,
    I8,
    I32,
    I64,
    Builder,
    Dialect,
    DialectRegistry,
    FunctionPass,
    IRModule,
    Operation,
    OpSpec,
    Pass,
    PassContext,
    PassManager,
    PassVerificationError,
    Pattern,
    PatternSet,
    Rewriter,
    RewriteResult,
    Successor,
    encode,
    verify,
)
from ppy_compiler.ir.dialects import core
from ppy_compiler.ir.dialects.core import CoreDialect
from ppy_compiler.ir.pattern import GreedyRewriteDriver, RewriteDidNotConverge
from ppy_compiler.ir.transforms import (
    Canonicalize,
    ConstantFold,
    DeadCodeElimination,
    SimplifyCFG,
    default_pipeline,
)


def _module(name: str = "m", params=(("x", I64),), results=(I64,)):
    module = IRModule(name)
    function = module.add_function("f", list(params), list(results))
    entry = function.add_entry_block()
    return module, function, entry, Builder(entry)


def _ops(function) -> list[str]:
    return [op.name for op in function.operations()]


def _canonicalize(module: IRModule, registry: DialectRegistry | None = None) -> PassContext:
    ctx = PassContext(registry, verify_after_each=True)
    PassManager(ctx).add(Canonicalize()).run(module)
    return ctx


# -- canonicalization patterns ------------------------------------------------


def test_identity_elements_fold_for_integers_but_not_floats():
    module, function, entry, b = _module()
    x = entry.arguments[0]
    zero = core.const(b, 0, I64)
    one = core.const(b, 1, I64)
    v = core.add(b, zero, x)
    v = core.add(b, v, zero)
    v = core.sub(b, v, zero)
    v = core.mul(b, one, v)
    v = core.mul(b, v, one)
    v = core.div(b, v, one)
    core.ret(b, v)
    ctx = _canonicalize(module)
    PassManager(ctx).add(DeadCodeElimination()).run(module)
    assert _ops(function) == ["core.ret"]
    assert entry.operations[0].operands[0] is x

    module, function, entry, b = _module(params=(("x", F64),), results=(F64,))
    x = entry.arguments[0]
    zero = core.const(b, 0.0, F64)
    core.ret(b, core.mul(b, core.add(b, x, zero), zero))
    _canonicalize(module)
    assert _ops(function) == ["core.const", "core.add", "core.mul", "core.ret"], (
        "x + 0.0 changes -0.0 and x * 0.0 may be NaN: floats keep their arithmetic"
    )


def test_multiplying_an_integer_by_zero_is_zero():
    module, function, entry, b = _module()
    zero = core.const(b, 0, I64)
    core.ret(b, core.mul(b, entry.arguments[0], zero))
    _canonicalize(module)
    PassManager().add(DeadCodeElimination()).run(module)
    assert _ops(function) == ["core.const", "core.ret"]
    assert entry.operations[0].attributes["value"] == 0


def test_constant_folding_follows_the_overflow_attribute():
    module, function, _entry, b = _module(params=(), results=(I8,))
    big = core.const(b, 127, I8)
    one = core.const(b, 1, I8)
    python = core.add(b, big, one, overflow="python")
    checked = core.add(b, big, one, overflow="checked")
    wrapped = core.add(b, big, one, overflow="wrap")
    total = core.add(b, core.add(b, python, checked, overflow="wrap"), wrapped, overflow="wrap")
    core.ret(b, total)
    ctx = _canonicalize(module)
    names = [
        (op.name, op.attributes.get("overflow"), op.attributes.get("value"))
        for op in function.operations()
    ]
    assert ("core.add", "python", None) in names, "127 + 1 does not fit i8: the guard decides"
    assert ("core.add", "checked", None) in names
    assert ("core.const", None, -128) in names, "wrap semantics wrap"
    assert any("folded 127 and 1" in line for line in ctx.remarks)


def test_integer_division_folds_by_its_rounding_and_never_by_zero():
    module, function, _entry, b = _module(params=(), results=(I64,))
    seven = core.const(b, -7, I64)
    two = core.const(b, 2, I64)
    zero = core.const(b, 0, I64)
    floor_div = core.div(b, seven, two, rounding="floor")
    trunc_div = core.div(b, seven, two, rounding="trunc")
    floor_mod = core.mod(b, seven, two, rounding="floor")
    trunc_mod = core.mod(b, seven, two, rounding="trunc")
    by_zero = core.div(b, seven, zero)
    core.ret(
        b,
        core.add(
            b,
            core.add(b, core.add(b, core.add(b, floor_div, trunc_div), floor_mod), trunc_mod),
            by_zero,
        ),
    )
    _canonicalize(module)
    constants = [op.attributes["value"] for op in function.operations() if op.name == "core.const"]
    assert -4 in constants and -3 in constants and 1 in constants and -1 in constants
    assert "core.div" in _ops(function), "division by zero stays for the guard"


def test_float_comparison_and_negation_rules_respect_nan():
    module, function, entry, b = _module(params=(("x", F64),), results=(BOOL,))
    x = entry.arguments[0]
    same = core.cmp(b, "eq", x, x)
    core.ret(b, same)
    _canonicalize(module)
    assert "core.cmp" in _ops(function), "NaN != NaN"

    module, function, entry, b = _module(results=(BOOL,))
    x = entry.arguments[0]
    core.ret(b, core.cmp(b, "lt", x, x))
    _canonicalize(module)
    assert _ops(function) == ["core.const", "core.ret"]
    assert entry.operations[0].attributes["value"] is False


def test_double_negation_is_removed_except_under_checked_semantics():
    module, function, entry, b = _module()
    x = entry.arguments[0]
    core.ret(b, core.neg(b, core.neg(b, x, overflow="wrap"), overflow="wrap"))
    _canonicalize(module)
    PassManager().add(DeadCodeElimination()).run(module)
    assert _ops(function) == ["core.ret"]

    module, function, entry, b = _module()
    x = entry.arguments[0]
    core.ret(b, core.neg(b, core.neg(b, x, overflow="checked"), overflow="checked"))
    _canonicalize(module)
    assert _ops(function) == ["core.neg", "core.neg", "core.ret"]


def test_casts_fold_only_when_nothing_is_lost():
    module, function, entry, b = _module(params=(("x", I32),), results=(I32,))
    x = entry.arguments[0]
    core.ret(b, core.cast(b, core.cast(b, x, I64), I32))
    _canonicalize(module)
    PassManager().add(DeadCodeElimination()).run(module)
    assert _ops(function) == ["core.ret"], "i32 -> i64 -> i32 is the identity"

    module, function, entry, b = _module()
    x = entry.arguments[0]
    core.ret(b, core.cast(b, core.cast(b, x, I32), I64))
    _canonicalize(module)
    assert _ops(function) == ["core.cast", "core.cast", "core.ret"], "i64 -> i32 truncates"

    module, function, entry, b = _module()
    core.ret(b, core.cast(b, entry.arguments[0], I64))
    _canonicalize(module)
    PassManager().add(DeadCodeElimination()).run(module)
    assert _ops(function) == ["core.ret"], "a cast to the same type is nothing"

    module, function, entry, b = _module(params=(), results=(I64,))
    core.ret(b, core.cast(b, core.const(b, -3.7, F64), I64))
    _canonicalize(module)
    PassManager().add(DeadCodeElimination()).run(module)
    assert [op.attributes.get("value") for op in function.operations()] == [-3, None]


def test_select_and_comparison_fold_on_constants():
    module, function, entry, b = _module()
    x = entry.arguments[0]
    three = core.const(b, 3, I64)
    four = core.const(b, 4, I64)
    smaller = core.cmp(b, "lt", three, four)
    core.ret(b, core.select(b, smaller, x, four))
    _canonicalize(module)
    PassManager().add(DeadCodeElimination()).run(module)
    assert _ops(function) == ["core.ret"]
    assert entry.operations[0].operands[0] is x


# -- structural passes ------------------------------------------------------


def test_simplify_cfg_folds_constant_branches_and_merges_chains():
    module, function, entry, b = _module()
    x = entry.arguments[0]
    then = function.body.add_block("then")
    otherwise = function.body.add_block("else")
    join = function.body.add_block("join", [("v", I64)])
    core.cond_br(b, core.const(b, True, BOOL), Successor(then), Successor(otherwise))
    b.at_end(then)
    doubled = core.add(b, x, x, name="doubled")
    core.br(b, Successor(join, [doubled]))
    b.at_end(otherwise)
    core.br(b, Successor(join, [x]))
    b.at_end(join)
    core.ret(b, join.arguments[0])
    ctx = PassContext(verify_after_each=True)
    report = PassManager(ctx).add(SimplifyCFG()).add(DeadCodeElimination()).run(module)
    assert report.changed
    assert [block.name for block in function.blocks()] == ["entry"]
    assert _ops(function) == ["core.add", "core.ret"]
    assert entry.operations[-1].operands[0] is doubled
    assert any("constant condition" in r for r in ctx.remarks)
    assert any("^else removed" in r for r in ctx.remarks)
    assert not verify(module)


def test_a_conditional_branch_to_one_block_becomes_a_branch():
    module, function, entry, b = _module()
    x = entry.arguments[0]
    join = function.body.add_block("join", [("v", I64)])
    zero = core.const(b, 0, I64)
    core.cond_br(b, core.cmp(b, "lt", x, zero), Successor(join, [x]), Successor(join, [x]))
    b.at_end(join)
    core.ret(b, join.arguments[0])
    PassManager(PassContext(verify_after_each=True)).add(SimplifyCFG()).add(
        DeadCodeElimination()
    ).run(module)
    assert _ops(function) == ["core.ret"]


def test_dead_code_elimination_keeps_effects():
    module, function, entry, b = _module(params=(("p", core.PtrType(I64)),), results=(I64,))
    p = entry.arguments[0]
    unused = core.add(b, core.const(b, 1, I64), core.const(b, 2, I64))
    core.mul(b, unused, unused)
    value = core.load(b, p)
    core.store(b, value, p)
    core.guard(b, core.cmp(b, "ge", value, core.const(b, 0, I64)), "range")
    core.call_extern(b, "tick", (), ())
    core.ret(b, value)
    PassManager(PassContext(verify_after_each=True)).add(DeadCodeElimination()).run(module)
    assert _ops(function) == [
        "core.load",
        "core.store",
        "core.const",
        "core.cmp",
        "core.guard",
        "core.call_extern",
        "core.ret",
    ]


# -- the pass manager -------------------------------------------------------


def test_the_default_pipeline_is_deterministic_and_verified():
    def build():
        module, function, entry, b = _module()
        x = entry.arguments[0]
        then = function.body.add_block("then")
        join = function.body.add_block("join", [("v", I64)])
        one = core.const(b, 1, I64)
        core.cond_br(b, core.cmp(b, "eq", one, one), Successor(then), Successor(join, [x]))
        b.at_end(then)
        core.br(b, Successor(join, [core.add(b, core.mul(b, x, one), core.const(b, 0, I64))]))
        b.at_end(join)
        core.ret(b, join.arguments[0])
        return module

    first = build()
    ctx = PassContext(verify_after_each=True)
    report = default_pipeline(1, ctx).run(first)
    assert report.names()[:2] == ["canonicalize", "simplify-cfg"]
    second = build()
    default_pipeline(1).run(second)
    assert encode(first) == encode(second)
    assert [op.name for op in first.functions["f"].operations()] == ["core.ret"]


def test_analyses_are_cached_and_invalidated_by_what_a_pass_preserves():
    module, function, entry, b = _module()
    x = entry.arguments[0]
    then = function.body.add_block("then")
    core.cond_br(b, core.const(b, True, BOOL), Successor(then), Successor(then))
    b.at_end(then)
    core.ret(b, core.add(b, x, core.const(b, 0, I64)))
    ctx = PassContext()
    ctx.analysis("dominators", function)
    assert ctx.cached() == {("dominators", "f")}
    PassManager(ctx).add(Canonicalize()).run(module)
    assert ctx.cached() == {("dominators", "f")}, "patterns never touch the CFG"
    PassManager(ctx).add(SimplifyCFG()).run(module)
    assert ctx.cached() == set(), "simplify-cfg invalidates dominators"
    with pytest.raises(KeyError, match="no analysis named 'aliases'"):
        ctx.analysis("aliases", function)
    dominators = ctx.analysis("dominators", function)
    assert dominators.dominates(entry, entry)


def test_a_pass_that_breaks_the_ir_is_named():
    class Breaker(Pass):
        name = "breaker"

        def run(self, module, ctx):
            function = module.functions["f"]
            function.entry.operations[-1].erase()  # no terminator any more
            return True

    module, _function, _entry, b = _module()
    core.ret(b, b.block.arguments[0])
    with pytest.raises(PassVerificationError, match="pass 'breaker' broke the IR") as caught:
        PassManager(PassContext(verify_after_each=True)).add(Breaker()).run(module)
    assert caught.value.pass_name == "breaker"
    assert "^entry is empty: it needs a terminator" in str(caught.value)


def test_plugin_passes_run_at_their_stage():
    ran: list[str] = []

    class Note(FunctionPass):
        name = "note"
        preserves = "all"

        def run_on_function(self, function, ctx):
            ran.append(function.name)
            return False

    module, _function, _entry, b = _module()
    core.ret(b, b.block.arguments[0])
    manager = default_pipeline(1)
    manager.register_stage_pass("after-canonicalization", Note)
    report = manager.run(module)
    names = report.names()
    assert names.index("note") == names.index("canonicalize") + 1
    assert ran == ["f"]
    with pytest.raises(ValueError, match="unknown stage 'sometime'"):
        manager.register_stage_pass("sometime", Note)
    with pytest.raises(ValueError, match="unknown stage"):
        PassManager().add_stage("never")


def test_a_pattern_that_never_settles_is_an_error_naming_it():
    class Restless(Pattern):
        root = "core.const"
        name = "restless"

        def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
            rewriter.set_attribute(op, "value", op.attributes["value"] + 1)
            return RewriteResult.success()

    _unused, function, _entry, b = _module(params=(), results=(I64,))
    core.ret(b, core.const(b, 0, I64))
    with pytest.raises(RewriteDidNotConverge, match="the last pattern to apply was restless"):
        GreedyRewriteDriver(PatternSet([Restless()]), max_iterations=50).run(function)


def test_a_dialect_contributes_its_own_patterns():
    class Twice(Pattern):
        root = "toy.twice"
        name = "twice-to-add"

        def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
            b = rewriter.builder(op)
            doubled = core.add(b, op.operands[0], op.operands[0], overflow="wrap")
            rewriter.replace_op(op, [doubled], "toy.twice lowered to core.add")
            return RewriteResult.success()

    class ToyDialect(Dialect):
        name = "toy"

        def register_operations(self, registry: DialectRegistry) -> None:
            registry.add_op(OpSpec("toy.twice", pure=True, operands=1, results=1))

        def register_patterns(self, registry) -> None:
            registry.add(Twice())

    own = DialectRegistry()
    own.register(CoreDialect())
    own.register(ToyDialect())
    module = IRModule("m", {"core": 1, "toy": 1})
    function = module.add_function("f", [("x", I64)], [I64])
    entry = function.add_entry_block()
    b = Builder(entry)
    twice = b.create("toy.twice", (entry.arguments[0],), (I64,))
    core.ret(b, twice.result)
    ctx = _canonicalize(module, own)
    assert _ops(function) == ["core.add", "core.ret"]
    assert ctx.remarks == ["@f: toy.twice lowered to core.add"]


def test_constant_fold_alone_leaves_identities_in_place():
    module, function, entry, b = _module()
    x = entry.arguments[0]
    zero = core.const(b, 0, I64)
    one = core.const(b, 1, I64)
    core.ret(b, core.add(b, core.add(b, one, one), core.add(b, x, zero)))
    PassManager().add(ConstantFold()).run(module)
    assert _ops(function).count("core.add") == 2, "x + 0 is canonicalization, not folding"
    assert 2 in [op.attributes.get("value") for op in function.operations()]
