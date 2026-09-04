"""Answering an obligation with Z3.

The question is whether `hypotheses and not (goal fits a 64-bit word)` has a
model. Unsatisfiable means no input the hypotheses allow overflows, and the
lowering may leave the guard out. Anything else -- a model, a timeout, an
unknown -- keeps the guard, which is the answer that is never wrong.

The theory is integer arithmetic, not bit-vectors: the goal is that the
*true* value fits the word, and bit-vectors would assume the wrap the proof
exists to rule out. Multiplication makes the theory nonlinear, which Z3
decides reliably for the small products a loop body has and gives up on
otherwise, within the timeout.

The solver is optional. `ppy-lang[solver]` installs it; without it every
obligation is unproven and the code is exactly what it was.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from .obligations import (
    I64_MAX,
    I64_MIN,
    BinOp,
    Const,
    Obligation,
    Var,
    fits_by_intervals,
    variables,
)

if TYPE_CHECKING:
    from .obligations import Relation, Term

__all__ = ["Prover"]

#: Per obligation. A loop body has a handful of chains, a proof takes tens
#: of milliseconds once the solver is warm, and one that needs longer than
#: this is one the guard handles for free.
_TIMEOUT_MS = 500


class Prover:
    """Proves obligations, remembering each answer for the process."""

    def __init__(self, timeout_ms: int = _TIMEOUT_MS) -> None:
        self.timeout_ms = timeout_ms
        self._memo: dict[str, bool] = {}
        #: How many obligations the solver, rather than intervals, settled.
        self.solved = 0

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("z3") is not None

    @staticmethod
    def version() -> str:
        """The solver's version, for a cache key: a proof is the solver's word."""
        if not Prover.available():
            return "none"
        import z3

        return str(z3.get_version_string())

    def proves(self, obligation: Obligation) -> bool:
        key = obligation.digest()
        known = self._memo.get(key)
        if known is not None:
            return known
        answer = self._prove(obligation)
        self._memo[key] = answer
        return answer

    def _prove(self, obligation: Obligation) -> bool:
        if fits_by_intervals(obligation.goal):
            return True
        if any(v.low is None and v.high is None for v in variables(obligation.goal)) and not (
            obligation.hypotheses
        ):
            # A variable nothing bounds can be anything; no solver needed to
            # find the overflowing input.
            return False
        if not self.available():
            return False
        import z3

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        symbols: dict[str, object] = {}

        def term(node: Term):  # type: ignore[no-untyped-def]
            if isinstance(node, Const):
                return z3.IntVal(node.value)
            if isinstance(node, Var):
                symbol = symbols.get(node.name)
                if symbol is None:
                    symbol = symbols[node.name] = z3.Int(node.name)
                    if node.low is not None:
                        solver.add(symbol >= node.low)
                    if node.high is not None:
                        solver.add(symbol <= node.high)
                return symbol
            assert isinstance(node, BinOp)
            left, right = term(node.left), term(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            raise _Unencodable(node.op)

        try:
            goal = term(obligation.goal)
            for relation in obligation.hypotheses:
                solver.add(_relation(z3, relation, term))
        except _Unencodable:
            return False
        solver.add(z3.Not(z3.And(goal >= I64_MIN, goal <= I64_MAX)))
        self.solved += 1
        return solver.check() == z3.unsat


class _Unencodable(Exception):
    """An operator the encoding does not model; the guard stays."""


def _relation(z3, relation: Relation, term):  # type: ignore[no-untyped-def]
    left, right = term(relation.left), term(relation.right)
    match relation.op:
        case "<=":
            return left <= right
        case "<":
            return left < right
        case ">=":
            return left >= right
        case ">":
            return left > right
        case "==":
            return left == right
    raise _Unencodable(relation.op)
