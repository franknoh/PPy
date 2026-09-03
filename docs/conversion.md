# Conversion and inference

Two commands turn untyped Python into typed `.ppy` without changing what it
does, sharing one engine and one guarantee of determinism — the same input
produces the same bytes everywhere:

- **`ppy convert`** is strict staticization. The output must be valid strict
  PPY: the converter re-analyzes what it produced, and anything `ppy check`
  would reject — a dynamic feature outside a `ppy.dynamic` boundary, a
  parameter left without a type, an unvouched decorator pinning one back —
  fails the conversion with the checker's own diagnostic.
- **`ppy migrate`** is the permissive migration tool for normal existing
  Python. Same engine, no gate: dynamic features convert faithfully with an
  advisory, blocked annotations stay off, and `ppy check` picks up from
  there.

Honesty is machine-checked: in `examples/`, every file named `<name>.ppy`
next to a `<name>.py` is exactly what the converter wrote, and
`examples/verify_conversions.py` regenerates each one to prove it.

## Where types come from

Inference is interprocedural and runs to a true fixpoint
(`analysis/inference.py`) -- no round limit, a convergence guard (`E9001`)
instead of silently keeping whatever the last round held. It has two stages
that never interleave: **evidence** (monotone: call-site types only join,
a field or usage inference only fills an unknown) runs until nothing changes;
**generalization** (presenting `list[T]` as `Sequence[T]`) then decides from
settled facts. A ten-deep call chain, mutual recursion, and a recursive
`fact(n - 1)` return type all settle -- a recursive call's unknown is a
placeholder for the thing being computed, not an answer.

1. **Call sites.** Every call to a project function contributes its argument
   types, positional and keyword alike, matched to parameters by the same
   binder the checker uses (`analysis/binding.py`). Evidence is *joined*
   across rounds, never frozen: a caller that only becomes typeable late
   still widens the callee's signature, so `sink([1])` here and
   `sink(y)` with a dict there infer `list[int] | dict[str, int]`, not
   whichever came first.
2. **Fields.** Every `self.x = value` in the class, joined, types the field
   -- `None` in `__init__` and a value in another method is `T | None` --
   and the annotation is written where the field is first assigned. A field
   the class body annotated keeps what its author wrote.
3. **Usage.** A parameter nothing in the project calls is typed from the
   arithmetic it takes part in.
4. **Empty containers.** `out = []` gets its element type from what is
   appended, and the annotation is written at the assignment.

A parameter that still has no stable type is reported (`E1304`), never
silently made `Any`. That report is the finding; the errors downstream of it
-- every call that received the unknown, every return that carried it -- are
not shown, since each would only say `<unknown>` again. One `W2006` line
gives their count and names the unknown signatures behind them.

## Protocol widening

A read-only container parameter is declared as the protocol its body actually
needs — `Sequence[T]` instead of `list[T]`, `Mapping[K, V]` instead of
`dict[K, V]` — so a caller holding a tuple is not rejected for no reason.

The decision is an allowlist of uses, not a mutation test. `xs.copy()`,
`xs + ys`, `xs[:]`, and `return xs` are all reads, and each one either does
not exist on the protocol or answers with a different type through it; any of
them keeps the concrete type. Each protocol also has its own capability set —
`reversed(xs)` is fine on a `Sequence` and blocks the `Mapping` widening,
because `Mapping` declares no `__reversed__`. Forwarding the parameter to
another function widens only if that callee (after its own widening) accepts
the protocol, keyword arguments included.

Uses are found through a flow-sensitive **alias analysis**
(`analysis/aliasing.py`), because names are not objects: in
`ys = xs; zs = ys; zs.append(1)` the mutation reaches `xs`, whatever name
performed it, and the parameter stays `list`. Reassignment kills an alias
(`ys = xs; ys = []` frees `xs` again), branches join, aliasing through a
tuple is tracked, and `list(xs)` is known to be a copy. The same alias map
feeds purity and escape analysis, so mutating a local through an alias is
still pure, and mutating a parameter through one never is.

When call sites disagree on the concrete type but agree on the protocol —
a list here, a tuple there — the union collapses to the shared
`Sequence[T]`.

## `Final`

`Final` is written only where all three hold:

- the name is bound exactly once, counting every binding form Python has
  (`with ... as`, `except ... as`, `del`, imports, `def`/`class`, match
  captures, `global` in any function);
- no file **anywhere in the project** assigns the attribute — a write index
  (`analysis/global_writes.py`) is built over every source under the project
  root, so converting one file still sees the reverse dependency doing
  `store.NAME = ...` — spelled statically or as `s =
  importlib.import_module("store"); s.NAME = ...`, because the lexical layer
  understands a constant `import_module` as the import it is. The scan rides
  the shared lexical bindings: a
  function-scope `import other as s` shadows nothing at module level,
  `alias = store` counts for as long as the binding lasts, relative imports
  resolve against the file's package, `setattr`/`delattr` count, a computed
  `setattr` name disqualifies the whole module, `global s; s = other` in one
  function makes `s.X = 1` in another a write to *both* candidates, and a
  project file that fails to parse fails the proof closed;
- the name reads as a constant (`UPPER_CASE`).

The last rule is deliberate: `Final` is an interface contract, and a lowercase
global that happens to be bound once is not announcing one.

## Failure is atomic

Neither command writes anything when the settled analysis holds an error: a
half-converted tree (some modules `.ppy`, the broken ones still `.py`) does
not even import, and there is no good half of that outcome to keep. Under
`ppy migrate`, dynamic-feature findings (`E15xx`) are exempt — converting
them faithfully is that command's job, and `ppy check` will still demand
their `ppy.dynamic` boundary afterwards. Under `ppy convert` they fail the
conversion like anything else the strict gate catches. `--dry-run` prints
what conversion would have said either way.

## The strict gate

`ppy convert` ends by re-analyzing the text it is about to write, overlaid on
the project in strict mode. This is the definition of the command, not an
extra check: strict conversion means the output is valid strict PPY, so the
gate simply asks the checker. When the gap is one the converter understands —
inference knew a type but a decorator or a reflective read pinned the
function — the diagnostic says so and names the ways out: annotate it
yourself, mark it `@ppy.reflective`, or run `ppy migrate`.

## What else conversion does

- attaches `@ppy.pure` exactly where the checker proved it;
- keeps a return type that is a finite closed set as `Literal[...]`
  (sound for returns, where the body is the whole evidence; never for
  parameters, where call sites are only a sample);
- moves a class above the function that annotates against it, so
  `'list[Rect]'` becomes `list[Rect]` — but only when the move is provably
  reorder-safe (`--hoist-classes=safe`, the default), on **both** sides: the
  moved class and every definition it crosses. A crossed decorator that
  probes `globals()` would otherwise observe the class ahead of time.
  Creating a class runs more than its spelled expressions: any explicit base
  may carry `__init_subclass__` or a metaclass, and a name-valued class field
  may be a descriptor with `__set_name__` — so safe mode allows only baseless
  (or `object`-based) classes with literal-valued fields and plain methods.
  `aggressive` moves any class, `off` moves none;
- writes signature annotations only onto functions whose decorators are all
  known to tolerate them (`analysis/decorators.py`). Decorator identity is
  *resolved at its point in the file* (`analysis/lexical.py`): `@cache` above
  a local `def cache` is that function even when `from functools import
  cache` appears later, and only the binding in force at the `def` matches
  the known table. The same lexical layer answers "what does this name mean
  here?" for the reflection scan and the write index, so the three cannot
  disagree.
  An unknown decorator, or one that reads `__annotations__` like
  `singledispatch`, saw an untyped function and must keep receiving one — and
  so did any code anywhere in the project calling `inspect.signature`,
  `typing.get_type_hints`, or reading `__annotations__` on it
  (`analysis/reflection.py`) — through any alias chain: `sig = i.signature;
  fn = lib.f; sig(fn)` observes `lib.f`. An unresolvable read blocks
  everything. The
  types are still inferred — they are just not materialized;
- orders imports into PEP 8 groups with `import ppy` ahead of first-party
  imports — the loader must install before the first `.ppy` import;
- rewrites the `input` idioms into the typed reader, where the module reads
  with `input` and never touches `sys.stdin` (two readers of one file
  descriptor would not agree on where it is): `int(input())` becomes
  `ppy.input[int]()`, `a, b = map(int, input().split())` becomes
  `ppy.input[tuple[int, int]]()`, a bare `input()` becomes `ppy.input[str]()`
  carrying its prompt, and a loop that fills a buffer one value at a time
  becomes one bulk `ppy.read_ints` over the same slots;
- with `--promote-buffers`, declares read-only numeric list parameters as
  `Buffer[T]` and rewrites the values feeding them into `array.array`
  (remarked as `R3002`, or `R3003` with the reason it could not);
- with `--format` (or `[tool.ppy.convert] format = true`), hands the result to
  the formatter the project *declares* — `[tool.ppy.format] backend`, or
  detected from `[tool.ruff]`/`ruff.toml`/`[tool.black]` (ruff wins when both
  are configured; no declaration means built-in normalization only). A
  declared formatter that is missing or fails is an error (`E1802`), never a
  silent restyle by something else.

It does not rename, split functions, or restructure algorithms — those are
design decisions, and the converter's output must be attributable to the
input.

## Project conversion

`ppy convert DIR` analyzes the directory as one call graph, so types flow
between files. `--in-place` writes each `.ppy` and removes the `.py` it
replaces; leaving both is warned (`W2005`) because a module may not be
provided twice (`E1003`).
