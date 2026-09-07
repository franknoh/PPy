"""Canonicalization: every dialect's patterns, applied until none applies."""

from __future__ import annotations

from ..dialect import DialectRegistry
from ..dialects.core_patterns import folding_patterns
from ..model import IRFunction
from ..passes import FunctionPass, PassContext
from ..pattern import GreedyRewriteDriver, PatternSet, Rewriter

__all__ = ["Canonicalize", "ConstantFold", "canonicalization_patterns"]


def canonicalization_patterns(registry: DialectRegistry) -> PatternSet:
    """What every registered dialect contributes."""
    patterns = PatternSet()
    for dialect in registry.dialects.values():
        dialect.register_patterns(patterns)
    for pattern in registry.patterns:
        patterns.add(pattern)  # type: ignore[arg-type]
    return patterns


class Canonicalize(FunctionPass):
    name = "canonicalize"
    #: Patterns replace values and erase operations; they never touch a
    #: terminator, so the control-flow graph is what it was.
    preserves = ("dominators",)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        patterns = canonicalization_patterns(ctx.registry)
        rewriter = Rewriter()
        changes = GreedyRewriteDriver(patterns).run(function, rewriter)
        for line in rewriter.log:
            ctx.remark(f"@{function.name}: {line}")
        return changes > 0


class ConstantFold(FunctionPass):
    name = "constant-fold"
    preserves = ("dominators",)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        patterns = PatternSet(folding_patterns())
        rewriter = Rewriter()
        changes = GreedyRewriteDriver(patterns).run(function, rewriter)
        for line in rewriter.log:
            ctx.remark(f"@{function.name}: {line}")
        return changes > 0
