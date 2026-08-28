# Conversion and inference

`ppy convert` turns untyped Python into typed `.ppy` without changing what it
does. The command is deterministic — the same input produces the same bytes
everywhere — and its honesty is machine-checked: in `examples/`, every file
named `<name>.ppy` next to a `<name>.py` is exactly what the converter wrote,
and `examples/verify_conversions.py` regenerates each one to prove it.

## Where types come from

Inference is interprocedural and runs to a fixpoint
(`analysis/inference.py`):

1. **Call sites.** Every call to a project function contributes its argument
   types, positional and keyword alike, matched to parameters by the same
   binder the checker uses (`analysis/binding.py`). Evidence is *joined*
   across rounds, never frozen: a caller that only becomes typeable on round
   two still widens the callee's signature, so `sink([1])` here and
   `sink(y)` with a dict there infer `list[int] | dict[str, int]`, not
   whichever came first.
2. **Fields.** `self.x = value` in `__init__` types the field; the annotation
   is written where the field is first assigned.
3. **Usage.** A parameter nothing in the project calls is typed from the
   arithmetic it takes part in.
4. **Empty containers.** `out = []` gets its element type from what is
   appended, and the annotation is written at the assignment.

A parameter that still has no stable type is reported (`E1304`), never
silently made `Any`.

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

When call sites disagree on the concrete type but agree on the protocol —
a list here, a tuple there — the union collapses to the shared
`Sequence[T]`.

## `Final`

`Final` is written only where all three hold:

- the name is bound exactly once, counting every binding form Python has
  (`with ... as`, `except ... as`, `del`, imports, `def`/`class`, match
  captures, `global` in any function);
- no other module in the project assigns the attribute (`other.NAME = ...`);
- the name reads as a constant (`UPPER_CASE`).

The last rule is deliberate: `Final` is an interface contract, and a lowercase
global that happens to be bound once is not announcing one.

## What else conversion does

- attaches `@ppy.pure` exactly where the checker proved it;
- keeps a return type that is a finite closed set as `Literal[...]`
  (sound for returns, where the body is the whole evidence; never for
  parameters, where call sites are only a sample);
- moves a class above the function that annotates against it, so
  `'list[Rect]'` becomes `list[Rect]`;
- orders imports into PEP 8 groups with `import ppy` ahead of first-party
  imports — the loader must install before the first `.ppy` import;
- with `--promote-buffers`, declares read-only numeric list parameters as
  `Buffer[T]` and rewrites the values feeding them into `array.array`
  (remarked as `R3002`, or `R3003` with the reason it could not);
- with `--format` (or `[tool.ppy.convert] format = true`), hands the result to
  the project's own formatter afterwards, telling it exactly which imports are
  first-party. A formatter that is installed but fails is an error (`E1802`),
  not a silent skip.

It does not rename, split functions, or restructure algorithms — those are
design decisions, and the converter's output must be attributable to the
input.

## Project conversion

`ppy convert DIR` analyzes the directory as one call graph, so types flow
between files. `--in-place` writes each `.ppy` and removes the `.py` it
replaces; leaving both is warned (`W2005`) because a module may not be
provided twice (`E1003`).
