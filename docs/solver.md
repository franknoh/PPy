# Where a solver fits

A plan, not a feature: nothing here is implemented yet. It says where an SMT
solver (Z3) would earn its place in PPY, where it would not, and what the
first cut looks like, so that the work can start from a decision rather than
from a hunch.

## The short answer

Two places, and only two:

1. **Discharging overflow guards in native loops.** The LLVM path runs
   integer arithmetic on 64-bit machine words and guards every operation
   that could overflow with a branch back to CPython, because a Python `int`
   is unbounded and the three paths must agree bit for bit. A guard the
   compiler can *prove* never fires can be dropped, and a loop with no guard
   in its body is a loop LLVM can vectorize. Interval arithmetic already
   proves the convex cases; a solver proves the relational ones.
2. **Translation validation of the optimizer.** Every rewrite the optimizer
   makes to an integer expression can be checked against the original for
   equivalence, offline, over the example corpus. A counterexample is a bug
   in a pass, found before a user finds it.

Not a place: type inference. The checker's inference is a monotone fixpoint
over finite lattices -- types, effects, integer ranges -- which is fast,
predictable, and gives an answer of the kind the rest of the compiler
consumes. A solver gives *satisfiable* or *not*, per query, at a cost per
query that no per-expression pipeline can afford. Also not a place: the
runtime. `ppy_runtime` never imports the compiler, and a guard decided at
run time is decided by a compare instruction, not by a solver.

## 1. Overflow guards

### What the native path does today

`backend/llvm/lowering.py` lowers `+`, `-`, `*` on `int` through the
`llvm.*.with.overflow` intrinsics and branches to the function's fallback
block when the flag is set. That is `safeguards = "inline"`. The default for
`ppy run`, `hoisted`, does better when it can: if both operands carry a
proven range -- from an annotation such as `Annotated[int, ppy.Range(0,
1000)]` or `pydantic.Field(ge=0, le=1000)`, from a `range()` induction, from
a constant -- and the chain contains a multiplication, the extreme corners
are checked once in a guard block ahead of the loop and the body runs the
plain instruction with `nsw`. `off` drops the guards and wraps, which is
what `ppy build` chooses.

The limits are exactly the limits of interval arithmetic:

- a chain with no multiplication is left to LLVM, which folds guarded
  additions of an induction variable and nothing else;
- an operand with no range in hand -- one bounded by a *comparison* rather
  than an annotation, such as `if n > 10**6: return` above the loop, or by
  another variable, `i < n` with `n <= len(a)` -- has no corners to check;
- a range that is convex but wide, `[0, 2**40]` times `[0, 2**40]`, overflows
  at the corners and stays guarded even when the loop never reaches them
  because of a relation between the two.

### What a solver adds

An *obligation* per arithmetic chain: hypotheses `H` over mathematical
integers -- every range fact on the operands, every dominating comparison
the flow analysis narrowed on, `len(...) >= 0`, the induction bounds of the
enclosing loops -- and the goal that the chain's value, computed as a
mathematical integer, lies in `[-2**63, 2**63)`. Ask the solver whether
`H and not goal` is satisfiable. Unsatisfiable means no input the hypotheses
allow overflows, and the chain lowers to plain `nsw` instructions with no
guard. Anything else -- satisfiable, unknown, timeout -- keeps the guards it
has today. The default is the sound one.

The theory is linear integer arithmetic with multiplication where it occurs
(Z3's nonlinear support is incomplete but reliable on the small products a
loop body has), *not* bit-vectors: the question is whether the true value
fits a word, and bit-vectors would assume the wrap being ruled out. `//`
and `%` are encoded with Python's floor semantics; shifts with a bounded
shift count. Anything the encoder does not understand -- a call, a
subscript with unknown bounds, a float -- makes the obligation unprovable,
which keeps the guard.

Where the hypotheses come from is already in the tree:

| hypothesis | source |
|---|---|
| `lo <= x <= hi` | `Facts.int_range` on the operand, from annotations, constants, `len`, narrowing |
| `0 <= i < n` | the `range()` induction the loop lowers |
| `n <= len(a)` | a dominating `if` the checker narrowed on, kept as a relation rather than an interval |
| `len(a) >= 0` | the builtin's facts |

The third row is the one addition to the analysis: today a comparison
against another variable narrows nothing, because an interval cannot hold
a relation. Keeping `(name, op, name)` triples alongside the interval is
cheap and is what turns `i < n and n <= len(a)` into a proof.

### Cost and where it runs

Only at lowering, only for functions with loops, only for chains the
interval rules did not already settle, with a per-obligation timeout of a
few tens of milliseconds and the result cached in the store under the
obligation's digest and the solver's version. `ppy check` never runs it.
`ppy run` pays it once per cache miss. A project with no solver installed
behaves exactly as today.

### The first cut

1. `backend/llvm/obligations.py`: from a lowered chain, its hypotheses and
   goal, as a small AST of its own -- printable, digestible, testable
   without Z3.
2. `backend/llvm/prover.py`: the Z3 encoding of that AST, behind an
   `is_available()` that imports lazily. `pyproject.toml` gains an extra,
   `ppy-lang[solver]`, pinning `z3-solver`.
3. `lowering._checked_binary` asks the prover before emitting a guard, when
   `llvm.prover = "z3"` in `[tool.ppy]` (default `off` until the numbers are
   in) or `--prover z3` on the command line.
4. `ppy explain` says, per chain, *guarded*, *hoisted*, or *proven*, and
   what the proof rested on.
5. The relation facts in `analysis/refinements.py`, and the checker keeping
   them from comparisons.

Acceptance: on `examples/15_algorithms/bench.py`, the number of guard blocks
in the hot loops goes down, the three paths still agree on every example,
and the bench shows the difference. If it does not, the plan stops at step 4
and the switch stays off.

## 2. Translation validation

Every pass in `opt/passes.py` that rewrites integer and boolean expressions
-- `ConstantFold`, `Peephole`, `CommonSubexpression`, `CopyPropagation`,
`LoopInvariantMotion`, `InlineSmallFunctions`, `LoopUnroll` -- claims to
preserve behavior. The claim is testable: encode the function body before
and after the pass over the integer fragment, bounded loops unrolled to a
small depth, and ask whether any input distinguishes them.

This runs nowhere near a user's build. It is `scripts/validate_passes.py`,
run over the examples and the tests' fixtures, gated on the solver being
importable, and a weekly CI job. What it catches is a wrong rewrite; what
it costs is a script. The encoder is the same one as above, which is why
the two applications are one plan.

## What stays as it is

- Type inference, effect inference, and the interval lattice.
- The guard mechanism itself: a proof removes a guard, it does not replace
  one with a different check.
- The runtime, which has no reason to know a solver exists.
- Determinism: a cached proof is keyed by the obligation and the solver
  version, so a build is the same build on every machine that has the same
  cache, and a machine without the solver produces the guarded code, which
  is correct and only slower.
