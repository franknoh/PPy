# The toolbox

One small program, and every way 0.2.0 has of looking inside its build:
the IR at each stage, the source backends, the optimization report, the
sanitizers, and profile-guided optimization.

## Provenance

Hand-written. `toolbox.ppy` is written directly; there is no `.py` source
and no conversion step involved.

## What it shows

The program is ordinary -- a loop with a branch, an indexed read, a
function nobody calls -- so that the tools have something to say.

```bash
ppy inspect toolbox.ppy --stage analysis   # what the checker knows of each function
ppy inspect toolbox.ppy --stage ir         # the frontend's module, before any pass
ppy inspect toolbox.ppy --stage optimized  # what the backend receives
ppy emit ir toolbox.ppy                    # the same module as .ppyir text
ppy emit c toolbox.ppy                     # one C11 translation unit; `cpp` for C++17
ppy emit llvm-ir toolbox.ppy               # what LLVM is handed
```

```bash
ppy build toolbox.ppy --report-opt         # native or not and why, every remark by category
ppy build toolbox.ppy --report-opt-json report.json
```

```bash
ppy run --sanitize bounds,overflow toolbox.ppy
```

A sanitizer instruments the IR with checks the program did not ask for:
every buffer index, every wrapping or proven integer operation. A check that
fails is not a fallback: the function returns a sanitizer status and the
boundary raises `SanitizerFailure` naming the kind and the function.

```bash
ppy run --profile toolbox.ppy              # runs, then writes toolbox.ppyprof
ppy build --pgo toolbox.ppyprof toolbox.ppy --report-opt
ppy run --pgo toolbox.ppyprof toolbox.ppy
```

The profiling run counts every block and the taken edge of every branch, and
records what each native function was called with (`list[400]`, fifty
times). The guided build annotates what still matches: `compute` is hot,
`unused` is cold and never inlined, the branch on `x % 4` carries the
weights the run measured, and LLVM receives them as `!prof` metadata. The
report lists the profile first. A profile changes what is fast, never what
is computed: the program prints the same line under every command above.

## Run it

```bash
python  toolbox.ppy
ppy run toolbox.ppy
```
