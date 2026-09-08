# Every way of looking inside a build, on one program

`toolbox.ppy` is deliberately ordinary — a loop with a branch, an indexed
read, a function nobody calls — so that the tools have something to say.
The IR at each stage, the C and LLVM the backends write, the optimization
report, the sanitizers, and a profile-guided build all run on it below, and
the program prints the same line under every one of them. A tool changes
what you can see, or what is fast; never what is computed.

## The IR at each stage

```bash
ppy inspect toolbox.ppy --stage analysis   # what the checker knows of each function
ppy inspect toolbox.ppy --stage ir         # the frontend's module, before any pass
ppy inspect toolbox.ppy --stage optimized  # what the backend receives
ppy emit ir toolbox.ppy                    # the same module as .ppyir text
ppy emit c toolbox.ppy                     # one C11 translation unit; `cpp` for C++17
ppy emit llvm-ir toolbox.ppy               # what LLVM is handed
```

`analysis` says of each function whether it is native, whether it gets a
Python boundary, and why: `unused` is native but has "native callers only",
because a two-instruction body is not worth a boundary crossing. `emit c`
writes a translation unit that compiles alone with any C11 compiler and
answers what the LLVM road answers, guards and fallbacks included.

## The report

```bash
ppy build toolbox.ppy --report-opt
ppy build toolbox.ppy --report-opt-json report.json
```

Native or not and why, then every remark the passes left, by stable
category — `block merged`, `dead code removed`, `function inlined`,
`bounds guard removed` — so a tool can count them across versions.

## Sanitizers

```bash
ppy run --sanitize bounds,overflow toolbox.ppy
```

A sanitizer instruments the IR with checks the program did not ask for:
every buffer index, every wrapping or proven integer operation. A check that
fails is not a fallback. The function returns a sanitizer status and the
boundary raises `SanitizerFailure` naming the kind and the function.

## Profile-guided optimization

```bash
ppy run --profile toolbox.ppy              # runs, then writes toolbox.ppyprof
ppy build --pgo toolbox.ppyprof toolbox.ppy --report-opt
ppy run --pgo toolbox.ppyprof toolbox.ppy
```

The profiling run counts every block and the taken edge of every branch,
and records what each native function was called with (`list[400]`, fifty
times). The guided build annotates what still matches: `compute` is hot,
`unused` is cold and never inlined, the branch on `x % 4` carries the
weights the run measured, and LLVM receives them as `!prof` metadata. The
report lists the profile first. A function edited since the profile was
recorded is named (`W2009`) and built as without one.

## Run it

```bash
python  toolbox.ppy
ppy run toolbox.ppy
ppy inspect toolbox.ppy --stage analysis
ppy emit ir toolbox.ppy
ppy emit c toolbox.ppy
ppy build toolbox.ppy --report-opt
ppy run --sanitize bounds,overflow toolbox.ppy
ppy run --profile toolbox.ppy
ppy build --pgo toolbox.ppyprof toolbox.ppy --report-opt
```

<!-- outputs:start -->
## What it prints

**`python  toolbox.ppy`**

```text
975000 798
```

**`ppy run toolbox.ppy`**

```text
975000 798
```

**`ppy inspect toolbox.ppy --stage analysis`**

```text
; ---- toolbox [analysis] ----
module toolbox
  toolbox.compute: (list[int]) -> int
    effects: MayRaise[ZeroDivisionError]
    native: eligible
    boundary: bound (takes a buffer)
  toolbox.pick: (list[int], int) -> int
    effects: MayRaise[IndexError]
    native: eligible
    boundary: bound (takes a buffer)
  toolbox.unused: (int) -> int
    effects: none
    native: eligible
    boundary: native callers only (the boundary crossing costs more than the body saves)
  toolbox.main: () -> NoneType
    effects: Alloc, IO, MayRaise[IndexError, KeyError, TypeError, ValueError, ZeroDivisionError], WriteObject
    native: stays in Python: has effects that must run on CPython: IO
```

**`ppy emit ir toolbox.ppy`**

```text
ppyir 1
module @toolbox
dialect core 1

func @toolbox_compute(%xs: buffer<i64> {ownership = "owned", ppy.kind = "list"}) -> i64 attrs {effects = ["may_raise", "read_memory"], ppy.abi = "ppy", ppy.qualname = "toolbox.compute", ppy.releases_gil = true, ppy.symbol = "ppy_toolbox_compute"} loc("examples/42_toolbox/toolbox.ppy":1:0) {
^entry:
    %0 = core.const 0 : i64 loc("examples/42_toolbox/toolbox.ppy":2:4)
    %total_addr = core.alloca : ptr<i64, stack>
    core.store %0, %total_addr
    %xs_i_addr = core.alloca : ptr<i64, stack> loc("examples/42_toolbox/toolbox.ppy":3:4)
    %1 = core.const 0 : i64
    core.store %1, %xs_i_addr
    %x_addr = core.alloca : ptr<i64, stack>
    %2 = core.buffer_len %xs : index
    %3 = core.cast %2 : i64
    %4 = core.const 4 : i64
    %5 = core.const 0 : i64
    %6 = core.const 1 : i64
    core.br ^each.head1 loc("examples/42_toolbox/toolbox.ppy":3:4)
^each.head1:
```

*82 lines in all — [full output](outputs/04-ppy-emit-ir-toolbox-ppy.txt).*

**`ppy emit c toolbox.ppy`**

```text
/* toolbox: generated by ppy, C11 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

static inline int ppy_ovf_add_i64(int64_t a, int64_t b, int64_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, out);
#else
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
        return 1;
    }
    *out = a + b;
    return 0;
#endif
}

static inline int ppy_ovf_sub_i64(int64_t a, int64_t b, int64_t *out) {
#if defined(__GNUC__) || defined(__clang__)
```

*190 lines in all — [full output](outputs/05-ppy-emit-c-toolbox-ppy.txt).*

**`ppy build toolbox.ppy --report-opt`**

```text
optimization report: PPy (O2, ir road)
module toolbox
  toolbox.compute: native, bound to Python
  toolbox.pick: native, bound to Python
  toolbox.unused: native, native callers only
  toolbox.main: Python -- has effects that must run on CPython: IO
  block merged: 1
  dead code removed: 2
  note: 2
    - @toolbox_compute: core.cmp: folded 4 and 0
    - @toolbox_compute: core.guard: condition always holds
    - simplify-cfg: ^each.latch3 merged into ^endif7
    - dce: unused core.const removed
    - dce: unused core.const removed
```

**`ppy run --sanitize bounds,overflow toolbox.ppy`**

```text
975000 798
```

**`ppy run --profile toolbox.ppy`**

```text
975000 798
```

**`ppy build --pgo toolbox.ppyprof toolbox.ppy --report-opt`**

```text
optimization report: PPy (O2, ir road)
profile: toolbox.ppyprof (2 runs, hot from 5 calls)
  toolbox.compute: hot, 100 calls; arguments 0: list[400] x100
  toolbox.pick: cold, 0 calls
  toolbox.unused: cold, 0 calls
module toolbox
  toolbox.compute: native, bound to Python
  toolbox.pick: native, bound to Python
  toolbox.unused: native, native callers only
  toolbox.main: Python -- has effects that must run on CPython: IO
  block merged: 1
  dead code removed: 2
  note: 2
  profile applied: 3
    - @toolbox_compute: core.cmp: folded 4 and 0
    - @toolbox_compute: core.guard: condition always holds
    - profile: `toolbox.compute` is hot (100 call(s)); 2 branch(es) weighted, 1 loop(s) with trip counts
    - profile: `toolbox.pick` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - profile: `toolbox.unused` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - simplify-cfg: ^each.latch3 merged into ^endif7
```

*22 lines in all — [full output](outputs/09-ppy-build-pgo-toolbox-ppyprof-toolbox-pp.txt).*

<!-- outputs:end -->

## Read on

- [CLI](../../docs/cli.md) — `emit`, `inspect`, `--report-opt`, `--sanitize`, `--profile`/`--pgo` in full.
- [The IR](../../docs/internals/ir.md) — sanitizers and profiles as passes.

`toolbox.ppy` is hand-written; there is no `.py` source and no conversion step.
