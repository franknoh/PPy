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
| `cache-dir` | `.ppy-cache` | the content-addressed store; relative to the root. |
| `dynamic-boundaries` | `explicit` | `explicit` requires `ppy.dynamic` around dynamic features; `deny` forbids them outright (`E1505`). |
| `build-execution` | `deny` | whether build-time stages may execute project code (JAX StableHLO export, pydantic schema builds). |
| `source-roots` | `["src", "."]` | where modules are resolved from, in order. |
| `llvm.jit` | `true` | keep compiled code in-process via MCJIT; `false` always links a shared library. |
| `parallel.threads` | `auto` | worker pool size; `auto` uses every core, honouring `OMP_NUM_THREADS` when set. |
| `inference.write-local-annotations` | `true` | conversion annotates module globals and empty containers, not just signatures. |
| `convert.format` | `false` | same as passing `--format` to every `ppy convert`. |
| `convert.hoist-classes` | `safe` | which classes conversion may reorder: only provably inert definitions, any (`aggressive`), or none (`off`). |
| `format.backend` | `auto` | which formatter runs after the built-in pass; `auto` reads the project's own ruff/black configuration, `none` is built-in only. |
| `diagnostics.optimization-remarks` | `false` | emit `R3001` remarks for applied optimizations. |

CLI flags win over the file for one invocation; the file wins over defaults.
The effective configuration participates in cache keys where it changes the
artifact (opt level, directives, plugin options), so flipping a knob never
serves a stale build.
