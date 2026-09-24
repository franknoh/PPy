# Conversion and inference

Two commands turn untyped Python into typed `.ppy` without changing what it
does. They share one engine and one guarantee of determinism: the same input
produces the same bytes everywhere.

| command | what it is |
|---|---|
| `ppy convert` | Strict staticization. The output must be valid strict PPy: the converter re-analyzes what it produced, and anything `ppy check` would reject fails the conversion with the checker's own diagnostic. That includes a dynamic feature outside a `ppy.dynamic` boundary, a parameter left without a type, or an unvouched decorator pinning one back. |
| `ppy migrate` | The permissive migration tool for normal existing Python. Same engine, no gate: dynamic features convert faithfully with an advisory, blocked annotations stay off, and `ppy check` picks up from there. |

Which files to hand them, on a project of any size, is covered in
[Migrating a real project](migrating.md).

The examples are machine-checked. In `examples/`, every file named
`<name>.ppy` next to a `<name>.py` is exactly what the converter wrote, and
`examples/verify_conversions.py` regenerates each one to prove it.

## Where types come from

Inference is interprocedural and runs to a true fixpoint
(`analysis/inference.py`). There is no round limit. A convergence guard
(`E9001`) stops it instead of silently keeping whatever the last round held.

It has two stages that never interleave:

1. **Evidence** runs until nothing changes. It is monotone: call-site types
   only join, and a field or usage inference only fills an unknown.
2. **Generalization** (presenting `list[T]` as `Sequence[T]`) then decides
   from settled facts.

A ten-deep call chain, mutual recursion, and a recursive `fact(n - 1)` return
type all settle. A recursive call's unknown is a placeholder for the thing
being computed, and is not treated as an answer.

### Sources of evidence

1. **Call sites.** Every call to a project function contributes its argument
   types, positional and keyword alike, matched to parameters by the same
   binder the checker uses (`analysis/binding.py`). Evidence is joined
   across rounds, never frozen. A caller that only becomes typeable late
   still widens the callee's signature, so `sink([1])` here and `sink(y)`
   with a dict there infer `list[int] | dict[str, int]` instead of
   whichever came first.
2. **Fields.** Every `self.x = value` in the class, joined, types the field.
   `None` in `__init__` and a value in another method is `T | None`. The
   annotation is written where the field is first assigned. A field the
   class body annotated keeps what its author wrote.
3. **Usage.** A parameter nothing in the project calls is typed from the
   arithmetic it takes part in.
4. **Empty containers.** `out = []` gets its element type from what is
   appended, and the annotation is written at the assignment.

### When a type does not settle

A parameter that still has no stable type is reported (`E1304`), never
silently made `Any`. That report is the finding. The errors downstream of it
(every call that received the unknown, every return that carried it) are not
shown, since each would only say `<unknown>` again. One `W2006` line gives
their count and names the unknown signatures behind them.

## Protocol widening

A read-only container parameter is declared as the protocol its body needs:
`Sequence[T]` instead of `list[T]`, `Mapping[K, V]` instead of
`dict[K, V]`. A caller holding a tuple is then not rejected for no reason.

### Which uses allow it

The decision is an allowlist of uses, and does not test for mutation.

- `xs.copy()`, `xs + ys`, `xs[:]`, and `return xs` are all reads. Each one
  either does not exist on the protocol or answers with a different type
  through it, so any of them keeps the concrete type.
- Each protocol has its own capability set. `reversed(xs)` is fine on a
  `Sequence` and blocks the `Mapping` widening, because `Mapping` declares no
  `__reversed__`.
- Forwarding the parameter to another function widens only if that callee
  (after its own widening) accepts the protocol, keyword arguments included.

### Alias analysis

Uses are found through a flow-sensitive alias analysis
(`analysis/aliasing.py`), because names are not objects. In
`ys = xs; zs = ys; zs.append(1)` the mutation reaches `xs`, whatever name
performed it, and the parameter stays `list`.

- Reassignment kills an alias (`ys = xs; ys = []` frees `xs` again).
- Branches join.
- Aliasing through a tuple is tracked.
- `list(xs)` is known to be a copy.

The same alias map feeds purity and escape analysis. Mutating a local through
an alias is still pure, and mutating a parameter through one never is.

### Disagreeing call sites

When call sites disagree on the concrete type but agree on the protocol (a
list here, a tuple there), the union collapses to the shared `Sequence[T]`.

## `Final`

`Final` is written only where all three conditions hold:

- The name is bound exactly once, counting every binding form Python has
  (`with ... as`, `except ... as`, `del`, imports, `def`/`class`, match
  captures, `global` in any function).
- No file **anywhere in the project** assigns the attribute (see below).
- The name reads as a constant (`UPPER_CASE`).

The last rule is deliberate: `Final` is an interface contract, and a lowercase
global that happens to be bound once is not announcing one.

### The project-wide write index

A write index (`analysis/global_writes.py`) is built over every source under
the project root. Converting one file still sees the reverse dependency doing
`store.NAME = ...`, whether it is spelled statically or as
`s = importlib.import_module("store"); s.NAME = ...`, because the lexical layer
understands a constant `import_module` as the import it is.

The scan keeps its record per file in the cache store, so a file that has not
changed since the last conversion is not parsed again. The scan rides the
shared lexical bindings:

- A function-scope `import other as s` shadows nothing at module level.
- `alias = store` counts for as long as the binding lasts.
- Relative imports resolve against the file's package.
- `setattr`/`delattr` count, and a computed `setattr` name disqualifies the
  whole module.
- `global s; s = other` in one function makes `s.X = 1` in another a write to
  both candidates.
- A project file that fails to parse fails the proof closed.

## Failure is atomic

Neither command writes anything when the settled analysis holds an error. A
half-converted tree (some modules `.ppy`, the broken ones still `.py`) does
not even import, and there is no good half of that outcome to keep.

- Under `ppy migrate`, dynamic-feature findings (`E15xx`) are exempt.
  Converting them faithfully is that command's job, and `ppy check` will still
  demand their `ppy.dynamic` boundary afterwards.
- Under `ppy convert` they fail the conversion like anything else the strict
  gate catches.

`--dry-run` prints what conversion would have said either way.

## The strict gate

`ppy convert` ends by re-analyzing the text it is about to write, overlaid on
the project in strict mode. This is the definition of the command and not an
extra check: strict conversion means the output is valid strict PPy, so the
gate asks the checker.

Sometimes the gap is one the converter understands: inference knew a type,
but a decorator or a reflective read pinned the function. The diagnostic then
says so and names the ways out:

- annotate it yourself
- mark it `@ppy.reflective`
- run `ppy migrate`

## What else conversion does

### Purity and literal returns

- Attaches `@ppy.pure` where the checker proved it.
- Keeps a return type that is a finite closed set as `Literal[...]`. This is
  sound for returns, where the body is the whole evidence. It is never done
  for parameters, where call sites are only a sample.

### Class hoisting

Conversion moves a class above the function that annotates against it, so
`'list[Rect]'` becomes `list[Rect]`. It does so only when the move is provably
reorder-safe (`--hoist-classes=safe`, the default) on **both** sides: the
moved class and every definition it crosses. A crossed decorator that probes
`globals()` would otherwise observe the class ahead of time.

Creating a class runs more than its spelled expressions. Any explicit base may
carry `__init_subclass__` or a metaclass, and a name-valued class field may be
a descriptor with `__set_name__`. So safe mode allows only baseless (or
`object`-based) classes with literal-valued fields and plain methods.

| `--hoist-classes` | effect |
|---|---|
| `safe` (default) | moves only classes that pass the checks above |
| `aggressive` | moves any class |
| `off` | moves none |

### Annotations and decorators

Conversion writes signature annotations only onto functions whose decorators
are all known to tolerate them (`analysis/decorators.py`).

Decorator identity is resolved at its point in the file
(`analysis/lexical.py`). `@cache` above a local `def cache` is that function
even when `from functools import cache` appears later, and only the binding in
force at the `def` matches the known table. The same lexical layer answers
"what does this name mean here?" for the reflection scan and the write index,
so the three cannot disagree.

Some code saw an untyped function and must keep receiving one:

- an unknown decorator, or one that reads `__annotations__` like
  `singledispatch`
- any code anywhere in the project calling `inspect.signature`,
  `typing.get_type_hints`, or reading `__annotations__` on it
  (`analysis/reflection.py`)

This holds through any alias chain: `sig = i.signature; fn = lib.f; sig(fn)`
observes `lib.f`. An unresolvable read blocks everything. The types are still
inferred; they are just not materialized.

### Import order

Conversion orders imports into PEP 8 groups with `import ppy` ahead of
first-party imports. The loader must install before the first `.ppy` import.

### Reading input

Conversion rewrites the `input` idioms into the typed line reader, where the
module reads with `input` and never touches `sys.stdin`. Two readers of one
file descriptor would not agree on where it is.

| original | rewritten |
|---|---|
| `int(input())` | `ppy.input[int]()` |
| `a, b = map(int, input().split())` | `ppy.input[tuple[int, int]]()` |
| `input()` | `ppy.input[str]()` |

A prompt becomes a `print(..., end="", flush=True)` before the statement that
reads. Inside a loop's test or a comprehension it becomes the one-expression
`print(...) or ppy.input[T]()`.

Every rewrite reads the line the original read. `ppy.input` is
line-oriented, so `int(input())` on `1 2` fails before and after. A loop of
`int(input())` stays a loop and does not become a token scan, which would read
across lines the original never crossed.

### Buffer promotion

With `--promote-buffers`, conversion declares read-only numeric list
parameters as `Buffer[T]` and rewrites the values feeding them into
`array.array`. Each is remarked as `R3002`, or `R3003` with the reason it
could not.

### Formatting

With `--format` (or `[tool.ppy.convert] format = true`), conversion hands the
result to the formatter the project declares:

- `[tool.ppy.format] backend`, or
- detected from `[tool.ruff]`/`ruff.toml`/`[tool.black]` (ruff wins when both
  are configured)
- no declaration means built-in normalization only

A declared formatter that is missing or fails is an error (`E1802`), never a
silent restyle by something else.

### What it does not do

Conversion does not rename, split functions, or restructure algorithms. Those
are design decisions, and the converter's output must be attributable to the
input.

## Project conversion

`ppy convert DIR` analyzes the directory as one call graph, so types flow
between files. `--in-place` writes each `.ppy` and removes the `.py` it
replaces. Leaving both is warned (`W2005`) because a module may not be
provided twice (`E1003`).
