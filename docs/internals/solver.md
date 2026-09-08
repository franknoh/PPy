# Where a solver fits

Where an SMT solver (Z3) earns its place in PPY, where it does not, and what
is built. The first cut of section 1 is in: `llvm.prover = "z3"` or
`--prover z3`, with `ppy-lang[solver]` installed. Section 2 is still a plan.

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

Only at lowering, only for chains the interval rules did not already settle,
with a per-obligation timeout and the answer remembered for the process;
the lowered module is cached under a key that names the prover and its
version, so `ppy run` pays it once per cache miss. `ppy check` never runs
it. A project with the prover off behaves exactly as before; one that asks
for the prover without the solver installed gets an error, not a silently
slower build.

### What is built

- `backend/llvm/obligations.py`: a chain as a statement about integers --
  variables with the ranges the analysis recorded, constants, `+ - *` --
  with the relations its variables carry as hypotheses. Printable,
  digestible, testable without Z3, and settled by interval arithmetic
  first: a chain of declared ranges that fits by intervals asks no solver.
- `backend/llvm/prover.py`: the Z3 encoding, in integer arithmetic, behind
  a lazy import and a per-obligation timeout; answers are remembered per
  process. `ppy-lang[solver]` installs the solver; `ppy doctor` says
  whether it is there.
- Lowering asks the prover before it emits a guard. Every load is its own
  variable, so two loads of one name are two symbols; an induction
  variable carries `start <= i <= stop - 1` from the `range()` that drives
  it, as terms, so `i * k` with `n <= 100000` and `k` in `[-50, 50]` is a
  proof by relation, not by interval. A proven chain lowers to the plain
  instruction with `nsw`; an unproven one is emitted exactly as before.
- A function whose guards a proof may leave out checks each parameter's
  declared range once on entry, and a call outside it takes the fallback.
  `Range(lo, hi)` is a refinement the checker propagates, not a check the
  runtime makes, and a proof that rests on it must be a proof about every
  call that runs natively.
- The artifact is keyed by the prover and its version, the warm `ppy run`
  directory too. `NativeModule.proved` lists, per function, the chains a
  proof freed.

On the two kernels in `tests/test_native.py` -- a sum of squares with
`n <= 1000`, a weighted sum with `n <= 100000` and `k` in `[-50, 50]` --
the body keeps only the accumulator's guard, which nothing bounds, and the
loop's entry guard block loses its corner checks: eight fallback branches
become two, and the three paths agree, on a call inside the declared ranges
and on one outside them.

### What is next

1. `ppy explain` says, per chain, *guarded*, *hoisted*, or *proven*, and
   what the proof rested on -- today `NativeModule.proved` holds it.
2. Relation facts in `analysis/refinements.py`, kept from comparisons, so
   that `i < n and n <= len(a)` reaches the prover.
3. Terms for buffer elements whose storage width bounds them: an `i8`
   element is in `[-128, 127]` by construction.
4. The bench: on `examples/15_algorithms/bench.py`, whether the guards a
   proof removes are the ones that were keeping a loop from vectorizing.

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
