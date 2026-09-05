# Configuration

Everything lives under `[tool.ppy]` in `pyproject.toml`. Every key has a
default; an empty table is a valid project. The project root is found by
walking up to the nearest `pyproject.toml`, `ppy.toml`, or `.git`.

```toml
[tool.ppy]
python = ">=3.12,<3.15"
strict = true
opt-level = 2
cache-dir = ".ppy-cache"
dynamic-boundaries = "explicit"   # explicit | deny
build-execution = "deny"          # deny | allow  (JAX export, pydantic schema)
source-roots = ["src", "."]

[tool.ppy.python-backend]
enabled = true
interpreter = "python"

[tool.ppy.llvm]
enabled = true
target = "native"
jit = true
lto = "thin"
cpython-api = "version-specific"
host-cpu = false                  # build for this machine, not the baseline
# safeguards = "hoisted"          # unset: the command decides (see below)
# prover = "off"                 # "z3" proves overflow guards away; needs ppy-lang[solver]

[tool.ppy.parallel]
enabled = true
threads = "auto"                  # or an integer

[tool.ppy.inference]
interprocedural = true
write-local-annotations = true
implicit-any = "error"

[tool.ppy.convert]
format = false                    # run the declared formatter after convert
hoist-classes = "safe"            # safe | aggressive | off

[tool.ppy.format]
backend = "auto"                  # auto | ruff | black | none

[tool.ppy.diagnostics]
optimization-remarks = false

[tool.ppy.plugins.numpy]
enabled = true                    # any other keys are plugin options
```

| key | default | meaning |
|---|---|---|
| `python` | `>=3.12,<3.15` | the CPython versions the project promises to run on. |
| `strict` | `true` | implicit `Any` and unsound constructs are errors; `--no-strict` downgrades the ones with a sound fallback. |
| `opt-level` | `2` | project default, overridden per run by `-O` and per function by `@ppy.opt(n)`. |
| `llvm.safeguards` | per command | `hoisted` proves the extreme cases once in a guard block ahead of the loop; `inline` keeps every per-operation guard in the body; `off` drops the overflow guards on data arithmetic — 64-bit wrap semantics — while keeping every bounds check. Unset, the command decides: `ppy run` uses `hoisted` (Python integers, bit for bit), `ppy build` uses `off` (a wrap-semantics artifact, like every native compiler); `run --unsafe` and `build --safe` flip them per invocation. |
| `llvm.prover` | `off` | `z3` proves overflow guards away where the ranges the analysis established allow it: a chain of `+`, `-`, `*` over values with declared ranges and `range()` bounds that provably fits a 64-bit word is emitted without its guard, and the function checks its parameters' declared ranges once on entry so that a call outside them takes the fallback. Needs `ppy-lang[solver]`; without the solver the setting is an error. Never runs in `ppy check`. |
| `llvm.host-cpu` | `false` | compile object code for the CPU doing the build rather than the portable baseline: faster where the code vectorizes, and the artifact then needs a machine with the same instruction set. JIT code always targets the host, which is free because it never leaves the machine. |
| `cache-dir` | `.ppy-cache` | the content-addressed store; relative to the root. The `PPY_CACHE_DIR` environment variable overrides it with a per-project tree underneath — the escape hatch for a repo on a slow filesystem, such as a Windows-mounted drive under WSL. |
| `native-import` | `true` | whether `import ppy` may serve a `.ppy` module from its native build when the compiler is installed; `false` loads every `.ppy` as source. `PPY_IMPORT=python` does the same for one process. |
| `dynamic-boundaries` | `explicit` | `explicit` requires `ppy.dynamic` around dynamic features; `deny` forbids them outright (`E1505`). |
| `build-execution` | `deny` | whether build-time stages may execute project code (JAX StableHLO export, pydantic schema builds). |
| `source-roots` | `["src", "."]` | where modules are resolved from, in order. |
| `llvm.jit` | `true` | keep compiled code in-process via MCJIT; `false` always links a shared library. |
| `parallel.threads` | `auto` | worker pool size; `auto` uses every core, honouring `OMP_NUM_THREADS` when set. |
| `inference.write-local-annotations` | `true` | conversion annotates module globals and empty containers, not just signatures. |
| `convert.format` | `false` | same as passing `--format` to every `ppy convert` (and `ppy migrate`, which shares the engine). |
| `convert.hoist-classes` | `safe` | which classes conversion may reorder: only provably inert definitions, any (`aggressive`), or none (`off`). |
| `format.backend` | `auto` | which formatter runs after the built-in pass; `auto` reads the project's own ruff/black configuration, `none` is built-in only. |
| `diagnostics.optimization-remarks` | `false` | emit `R3001` remarks for applied optimizations. |

CLI flags win over the file for one invocation; the file wins over defaults.
The effective configuration participates in cache keys where it changes the
artifact (opt level, directives, plugin options), so flipping a knob never
serves a stale build.
