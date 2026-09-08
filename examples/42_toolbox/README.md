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
    %7 = core.load %xs_i_addr : i64 loc("examples/42_toolbox/toolbox.ppy":3:4)
    %8 = core.cmp.lt %7, %3 : bool
    core.cond_br %8, ^each.body2, ^each.end4
^each.body2:
    %9 = core.load %xs_i_addr : i64 loc("examples/42_toolbox/toolbox.ppy":3:4)
    %10 = core.buffer_load %xs, %9 : i64
    core.store %10, %x_addr
    %11 = core.load %x_addr : i64 loc("examples/42_toolbox/toolbox.ppy":4:8)
    %12 = core.mod %11, %4 {overflow = "python", rounding = "floor"} : i64
    %13 = core.cmp.eq %12, %5 : bool
    core.cond_br %13, ^then5, ^else6
^each.end4:
    %14 = core.load %total_addr : i64 loc("examples/42_toolbox/toolbox.ppy":8:4)
    core.ret %14
^then5:
    %15 = core.load %total_addr : i64 loc("examples/42_toolbox/toolbox.ppy":5:12)
    %16 = core.load %x_addr : i64
    %17 = core.add %15, %16 {overflow = "python"} : i64
    core.store %17, %total_addr
    core.br ^endif7
… 42 more lines
```

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
    return __builtin_sub_overflow(a, b, out);
#else
    if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) {
        return 1;
    }
    *out = a - b;
    return 0;
#endif
}

static inline int ppy_ovf_mul_i64(int64_t a, int64_t b, int64_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a > 0) {
        if (b > 0) { if (a > INT64_MAX / b) return 1; }
        else if (b < INT64_MIN / a) return 1;
    } else if (b > 0) {
        if (a < INT64_MIN / b) return 1;
    } else if (a != 0 && b < INT64_MAX / a) return 1;
… 150 more lines
```

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
profile: toolbox.ppyprof (1 run, hot from 2 calls)
  toolbox.compute: hot, 50 calls; arguments 0: list[400] x50
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
    - profile: `toolbox.compute` is hot (50 call(s)); 2 branch(es) weighted, 1 loop(s) with trip counts
    - profile: `toolbox.pick` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - profile: `toolbox.unused` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - simplify-cfg: ^each.latch3 merged into ^endif7
    - dce: unused core.const removed
    - dce: unused core.const removed
```

<!-- outputs:end -->
