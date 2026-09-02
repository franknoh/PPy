# Compatibility

PPY is version 0.1. The surface is wide, and not all of it is equally
settled. This says which parts you may build on, which will move, and what
happens when the two sides of a boundary disagree.

## Stability levels

| level | means |
|---|---|
| **stable** | changes only with a deprecation period and a changelog entry |
| **settling** | the shape is right and the details may still move; breaking changes are listed in the changelog |
| **experimental** | may change or be withdrawn in any release |

| surface | level | notes |
|---|---|---|
| the language subset — statements, expressions, the type system | settling | new constructs are added; accepted code is not un-accepted without a note |
| `ppy.pure`, `ppy.opt`, `ppy.native`, `ppy.jit`, `ppy.dynamic`, `ppy.check` | settling | the directives a program is written around |
| `ppy.input`, `ppy.buffer`, `ppy.read_ints`, `ppy.read_token` | experimental | added in 0.1; the spelling may still change |
| `ppy check` / `ppy run` / `ppy build` and their flags | settling | flags are added; removals get a deprecation release |
| `ppy convert` / `ppy migrate` output | settling | the output is regenerated from source, so a change shows up as a diff, not a break |
| diagnostic codes (`E1xxx`, `W2xxx`, `R3xxx`) | settling | a code keeps its meaning; new codes are added freely |
| the plugin interface in `plugins/base.py` | experimental | written for the plugins in this repository |
| the cache format | internal | see below; never read it yourself |
| the built-artifact ABI | versioned | see below |

## The cache is disposable

The build cache is optimization state, never a source of truth. Every
artifact in it is content-addressed and can be recomputed from the source it
was derived from.

- Deleting any part of it costs a rebuild and nothing else.
- A damaged SQLite index is quarantined next to itself as
  `index.sqlite.corrupt-<timestamp>`, rebuilt empty, and reported once as
  `warning[W2101]`. Compilation continues with cache misses.
- If even a fresh index cannot be written — a read-only directory, a full
  disk — the store keeps working in memory: every lookup is a miss, nothing
  persists, and the answer is the same.

The cache schema version is internal. It changes without notice, and a
mismatch is handled by rebuilding rather than by migrating.

## The built-artifact ABI

`ppy build` writes `ppy-bindings.json` with an `abi_version`. The runtime
that launches an artifact refuses a version it does not speak, with the
remedy in the message:

```
error[E1801]: <path> speaks ABI 2; this runtime speaks 1 -- rebuild the
artifact with `ppy build`
```

The manifest also records the Python version it was built for, and the
launcher refuses a different one for the same reason: the wrappers are
compiled against one interpreter's ABI. An artifact and the `ppy_runtime`
that launches it are expected to come from the same release.

## Python versions

3.12, 3.13, and 3.14 are tested on every change. A release supports the
versions its CI matrix runs; dropping one is a changelog entry.

## What a change to PPY may not do

- Make a program that checked clean produce a different answer on any of the
  three paths. A native path that cannot keep a promise falls back to Python
  rather than answering differently.
- Turn a cache or artifact problem into a failure to compile correct source.
