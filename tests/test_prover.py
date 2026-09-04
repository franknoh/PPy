"""The obligations a guard states, and the prover that discharges them."""

from __future__ import annotations

import pytest

from ppy_compiler.backend.llvm.obligations import (
    I64_MAX,
    BinOp,
    Const,
    Obligation,
    Relation,
    Var,
    bounds,
    fits_by_intervals,
    spell,
    variables,
)
from ppy_compiler.backend.llvm.prover import Prover


def _square_of_induction(n_high: int | None) -> Obligation:
    """`i * i` for `i` in `range(n)`, with `n` bounded above or not."""
    n = Var("n", 0, n_high)
    i = Var("i")
    return Obligation(
        BinOp("*", i, i),
        (Relation(Const(0), "<=", i), Relation(i, "<=", BinOp("-", n, Const(1)))),
    )


def test_an_obligation_spells_itself_and_digests_by_content():
    one = _square_of_induction(1000)
    same = _square_of_induction(1000)
    other = _square_of_induction(None)
    assert spell(one.goal) == "(i * i)"
    assert str(one) == "(i * i) given 0 <= i, i <= (n:[0, 1000] - 1)"
    assert one.digest() == same.digest() != other.digest()
    assert [v.name for v in variables(BinOp("+", BinOp("*", Var("a"), Var("b")), Var("a")))] == [
        "a",
        "b",
    ]


def test_intervals_settle_a_chain_of_declared_ranges_without_a_solver():
    small = BinOp("+", BinOp("*", Var("a", 0, 1000), Var("b", -5, 5)), Const(7))
    assert bounds(small) == (-4993, 5007)
    assert fits_by_intervals(small)
    wide = BinOp("*", Var("a", 0, 1 << 40), Var("b", 0, 1 << 40))
    assert not fits_by_intervals(wide)
    assert bounds(BinOp("-", Var("x"), Var("y", 0, 1))) == (None, None)
    prover = Prover()
    assert prover.proves(Obligation(small))
    assert prover.solved == 0, "intervals answered; the solver was not asked"


@pytest.mark.skipif(not Prover.available(), reason="z3-solver is not installed")
def test_the_solver_proves_what_intervals_cannot():
    prover = Prover()
    # `i` has no range of its own; only the relation to `n` bounds it.
    assert prover.proves(_square_of_induction(1000))
    assert prover.solved == 1
    # The same obligation again is remembered, not re-proved.
    assert prover.proves(_square_of_induction(1000))
    assert prover.solved == 1
    # Without a bound on `n`, `i * i` can be anything.
    assert not prover.proves(_square_of_induction(None))
    # A relation that bounds the product tightly enough is a proof.
    a, b = Var("a"), Var("b", 0, 1 << 40)
    assert prover.proves(
        Obligation(
            BinOp("*", a, b),
            (Relation(Const(0), "<=", a), Relation(a, "<=", Const(1 << 20))),
        )
    )
    # And one that leaves room to overflow is not: 2**40 * 2**40 does not fit.
    assert not prover.proves(
        Obligation(
            BinOp("*", a, b),
            (Relation(Const(0), "<=", a), Relation(a, "<=", Const(1 << 40))),
        )
    )
    assert I64_MAX < (1 << 40) * (1 << 40)


def test_an_operator_the_encoding_does_not_model_keeps_the_guard():
    prover = Prover()
    assert not prover.proves(Obligation(BinOp("//", Var("a", 0, 10), Var("b", 1, 10))))
    assert not prover.proves(Obligation(BinOp("+", Var("a"), Const(1))))
