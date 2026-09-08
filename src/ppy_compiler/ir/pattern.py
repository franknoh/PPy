"""Rewriting the IR by patterns, to a fixed point.

A pattern looks at one operation and either rewrites it through the
rewriter -- which is the only way anything is changed, so every change is
recorded and every affected operation is looked at again -- or says why it
does not apply. The greedy driver applies a set of patterns until none
applies, with a cap that turns a pattern that never settles into an error
naming it rather than a compiler that never finishes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .model import Attribute, Block, Builder, IRFunction, Operation, Value

__all__ = ["GreedyRewriteDriver", "Pattern", "PatternSet", "RewriteResult", "Rewriter"]


@dataclass(frozen=True, slots=True)
class RewriteResult:
    changed: bool
    reason: str = ""

    @staticmethod
    def success(reason: str = "") -> RewriteResult:
        return RewriteResult(True, reason)

    @staticmethod
    def failure(reason: str = "") -> RewriteResult:
        return RewriteResult(False, reason)


class Pattern:
    """One rewrite: the operation it roots at, and the rewrite itself."""

    #: The operation name this pattern roots at, or "" for any operation.
    root: str = ""
    #: Among patterns for one root, the higher benefit is tried first.
    benefit: int = 1
    #: What a report calls this rewrite; the class name by default.
    name: str = ""

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name or type(self).__name__


class PatternSet:
    def __init__(self, patterns: Iterable[Pattern] = ()) -> None:
        self._by_root: dict[str, list[Pattern]] = {}
        for pattern in patterns:
            self.add(pattern)

    def add(self, pattern: Pattern) -> None:
        bucket = self._by_root.setdefault(pattern.root, [])
        bucket.append(pattern)
        bucket.sort(key=lambda p: -p.benefit)

    def for_op(self, name: str) -> list[Pattern]:
        return self._by_root.get(name, []) + self._by_root.get("", [])

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._by_root.values())

    def __iter__(self):  # type: ignore[no-untyped-def]
        for bucket in self._by_root.values():
            yield from bucket


class Rewriter:
    """How a pattern changes the IR; every change goes through here."""

    def __init__(self) -> None:
        self.changes = 0
        #: Operations touched by the last rewrites, for the driver to revisit.
        self.touched: list[Operation] = []
        self.log: list[str] = []

    def builder(self, op: Operation) -> Builder:
        """A builder inserting before `op`."""
        return Builder().before(op)

    def replace_op(self, op: Operation, values: Sequence[Value | None], reason: str = "") -> None:
        """Every result of `op` now reads the matching value; `op` goes away.

        A None keeps a result in place -- for a result nothing reads.
        """
        if len(values) != len(op.results):
            raise ValueError(f"{op.name} has {len(op.results)} results, {len(values)} given")
        for result, value in zip(op.results, values, strict=True):
            if value is None:
                if result.uses:
                    raise ValueError(f"a used result of {op.name} needs a replacement")
                continue
            self._note_users(result)
            result.replace_all_uses_with(value)
            if isinstance(value.owner, Operation):
                self.touched.append(value.owner)
        self.erase_op(op, reason)

    def erase_op(self, op: Operation, reason: str = "") -> None:
        for result in op.results:
            if result.uses:
                raise ValueError(f"cannot erase {op.name}: a result is still used")
        for operand in op.operands:
            if isinstance(operand.owner, Operation):
                self.touched.append(operand.owner)
        op.erase()
        self._changed(reason)

    def replace_all_uses(self, value: Value, other: Value, reason: str = "") -> None:
        self._note_users(value)
        value.replace_all_uses_with(other)
        self._changed(reason)

    def set_operand(self, op: Operation, index: int, value: Value, reason: str = "") -> None:
        op.set_operand(index, value)
        self.touched.append(op)
        self._changed(reason)

    def set_attribute(self, op: Operation, key: str, value: Attribute, reason: str = "") -> None:
        op.attributes[key] = value
        self.touched.append(op)
        self._changed(reason)

    def _note_users(self, value: Value) -> None:
        for user, _index in value.uses:
            if isinstance(user, Operation):
                self.touched.append(user)
            else:
                parent = getattr(user, "block", None)
                if isinstance(parent, Block) and parent.operations:
                    self.touched.append(parent.operations[-1])

    def _changed(self, reason: str) -> None:
        self.changes += 1
        if reason:
            self.log.append(reason)


class RewriteDidNotConverge(RuntimeError):
    """A pattern set kept rewriting past the iteration cap."""


class GreedyRewriteDriver:
    """Apply patterns until none applies, revisiting what each rewrite touched."""

    def __init__(self, patterns: PatternSet, max_iterations: int = 10_000) -> None:
        self.patterns = patterns
        self.max_iterations = max_iterations

    def run(self, function: IRFunction, rewriter: Rewriter | None = None) -> int:
        """The number of changes made."""
        rewriter = rewriter or Rewriter()
        before = rewriter.changes
        worklist: list[Operation] = list(function.operations())
        queued = {id(op) for op in worklist}
        iterations = 0
        last: str = ""
        while worklist:
            op = worklist.pop()
            queued.discard(id(op))
            if op.parent is None:
                continue
            for pattern in self.patterns.for_op(op.name):
                if op.parent is None:
                    break
                result = pattern.match_and_rewrite(op, rewriter)
                if not result.changed:
                    continue
                iterations += 1
                last = repr(pattern)
                if iterations > self.max_iterations:
                    raise RewriteDidNotConverge(
                        f"rewriting did not settle after {self.max_iterations} rewrites; "
                        f"the last pattern to apply was {last}"
                    )
                for touched in rewriter.touched:
                    if touched.parent is not None and id(touched) not in queued:
                        worklist.append(touched)
                        queued.add(id(touched))
                rewriter.touched.clear()
                if op.parent is not None and id(op) not in queued:
                    worklist.append(op)
                    queued.add(id(op))
                break
        return rewriter.changes - before
