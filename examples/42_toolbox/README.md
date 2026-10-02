# The toolbox

One small program, run through each tool that looks inside its build: the IR
at each stage, the source backends, the optimization report, the sanitizers,
and profile-guided optimization. The program is ordinary (a loop with a
branch, an indexed read, a function nobody calls), so the tools have
something to say.

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

**`python  toolbox.ppy`**, **`ppy run toolbox.ppy`**, **`ppy run --sanitize bounds,overflow toolbox.ppy`**, **`ppy run --profile toolbox.ppy`**

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

<details markdown="1">
<summary>83 lines</summary>

```text
ppyir 1
module @toolbox
dialect core 1
attrs {ppy.libraries = ["ppy_collections"]}

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
^else6:
    %18 = core.load %total_addr : i64 loc("examples/42_toolbox/toolbox.ppy":7:12)
    %19 = core.sub %18, %6 {overflow = "python"} : i64
    core.store %19, %total_addr
    core.br ^endif7
^endif7:
    %20 = core.const 1 : i64 loc("examples/42_toolbox/toolbox.ppy":7:12)
    %21 = core.load %xs_i_addr : i64
    %22 = core.add %21, %20 {overflow = "wrap"} : i64
    core.store %22, %xs_i_addr
    core.br ^each.head1
}

func @toolbox_pick(%xs: buffer<i64> {ownership = "owned", ppy.kind = "list"}, %i: i64) -> i64 attrs {effects = ["may_raise", "read_memory"], ppy.abi = "ppy", ppy.qualname = "toolbox.pick", ppy.releases_gil = false, ppy.symbol = "ppy_toolbox_pick"} loc("examples/42_toolbox/toolbox.ppy":11:0) {
^entry:
    %i_addr = core.alloca : ptr<i64, stack> loc("examples/42_toolbox/toolbox.ppy":11:0)
    core.store %i, %i_addr
    %0 = core.load %i_addr : i64 loc("examples/42_toolbox/toolbox.ppy":12:4)
    %i_entry = core.load %i_addr : i64
    %1 = core.const 0 : i64
    %2 = core.buffer_len %xs : index
    %3 = core.cast %2 : i64
    %4 = core.cmp.ge %0, %1 : bool
    %5 = core.cmp.lt %0, %3 : bool
    %6 = core.and %4, %5 : bool
    core.guard %6 {kind = "bounds", message = "index out of range", raises = "IndexError: list index out of range"}
    %7 = core.buffer_load %xs, %0 : i64
    %8 = core.const 2 : i64
    %9 = core.mul %7, %8 {overflow = "python"} : i64
    core.ret %9
}

func @toolbox_unused(%n: i64) -> i64 attrs {effects = [], ppy.abi = "ppy", ppy.qualname = "toolbox.unused", ppy.releases_gil = false, ppy.symbol = "ppy_toolbox_unused"} loc("examples/42_toolbox/toolbox.ppy":15:0) {
^entry:
    %n_addr = core.alloca : ptr<i64, stack> loc("examples/42_toolbox/toolbox.ppy":15:0)
    core.store %n, %n_addr
    %0 = core.load %n_addr : i64 loc("examples/42_toolbox/toolbox.ppy":16:4)
    %n_entry = core.load %n_addr : i64
    %1 = core.const 7 : i64
    %2 = core.mul %0, %1 {overflow = "python"} : i64
    core.ret %2
}
```

</details>

**`ppy emit c toolbox.ppy`**

<details markdown="1">
<summary>96 lines</summary>

```text
/* toolbox: generated by ppy, C11 */
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    *out = a * b;
    return 0;
#endif
}

static int64_t *ppy_coll_heap(void) {
#ifdef __cplusplus
    static thread_local int64_t heap[9];
#else
    static _Thread_local int64_t heap[9];
#endif
    return heap;
}

static void ppy_coll_failed(void) {
    ppy_coll_heap()[5] = 1;
}

int32_t ppy_toolbox_compute(int64_t *xs, int64_t xs_len, int64_t *out);
int32_t ppy_toolbox_pick(int64_t *xs, int64_t xs_len, int64_t i, int64_t *out);
int32_t ppy_toolbox_unused(int64_t n, int64_t *out);

int32_t ppy_toolbox_compute(int64_t *xs, int64_t xs_len, int64_t *out) {
    int64_t total = 0;
    int64_t xs_i = 0;
    while (xs_i < xs_len) {
        int64_t x = xs[xs_i];
        if ((x % 4 + 4) % 4 == 0) {
            if (ppy_ovf_add_i64(total, x, &total)) { ppy_coll_failed(); return 1; } /* arith.ok */
        } else {
            if (ppy_ovf_sub_i64(total, 1, &total)) { ppy_coll_failed(); return 1; } /* arith.ok */
        }
        xs_i = (int64_t)((uint64_t)xs_i + 1u);
    }
    *out = total;
    return 0;
}

int32_t ppy_toolbox_pick(int64_t *xs, int64_t xs_len, int64_t i, int64_t *out) {
    if (i < 0 || i >= xs_len) { ppy_coll_failed(); return 1; } /* bounds.ok */
    int64_t t1;
    if (ppy_ovf_mul_i64(xs[i], 2, &t1)) { ppy_coll_failed(); return 1; } /* arith.ok */
    *out = t1;
    return 0;
}

int32_t ppy_toolbox_unused(int64_t n, int64_t *out) {
    int64_t t1;
    if (ppy_ovf_mul_i64(n, 7, &t1)) { ppy_coll_failed(); return 1; } /* arith.ok */
    *out = t1;
    return 0;
}
```

</details>

**`ppy build toolbox.ppy --report-opt`**

<details markdown="1">
<summary>37 lines</summary>

```text
optimization report: ppy-docs (O2, ir road)
module toolbox
  toolbox.compute: native, bound to Python
  toolbox.main: native, native callers only
  toolbox.pick: native, bound to Python
  toolbox.unused: native, native callers only
  block merged: 7
  dead code removed: 17
  note: 4
    - @toolbox_compute: core.cmp: folded 4 and 0
    - @toolbox_compute: core.guard: condition always holds
    - @toolbox_main: core.sub: folded 50 and 1
    - @toolbox_main: core.sub: folded 400 and 1
    - simplify-cfg: ^each.latch3 merged into ^endif7
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^for.body4
    - simplify-cfg: ^for.guards7 merged into ^for.end6
    - simplify-cfg: ^for.setup8 merged into ^for.end6
    - simplify-cfg: ^for.latch11 merged into ^for.body10
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
```

</details>

**`ppy build --pgo toolbox.ppyprof toolbox.ppy --report-opt`**

<details markdown="1">
<summary>47 lines</summary>

```text
optimization report: ppy-docs (O2, ir road)
profile: toolbox.ppyprof (2 runs, hot from 5 calls)
  toolbox.compute: hot, 100 calls; arguments 0: list[400] x100
  toolbox.main: cold, 0 calls
  toolbox.pick: cold, 0 calls
  toolbox.unused: cold, 0 calls
module toolbox
  toolbox.compute: native, bound to Python
  toolbox.main: native, native callers only
  toolbox.pick: native, bound to Python
  toolbox.unused: native, native callers only
  block merged: 7
  dead code removed: 17
  note: 4
  profile applied: 4
    - @toolbox_compute: core.cmp: folded 4 and 0
    - @toolbox_compute: core.guard: condition always holds
    - @toolbox_main: core.sub: folded 50 and 1
    - @toolbox_main: core.sub: folded 400 and 1
    - profile: `toolbox.compute` is hot (100 call(s)); 2 branch(es) weighted, 1 loop(s) with trip counts
    - profile: `toolbox.pick` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - profile: `toolbox.unused` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - profile: `toolbox.main` is cold (0 call(s)); 2 branch(es) weighted, 2 loop(s) with trip counts
    - simplify-cfg: ^each.latch3 merged into ^endif7
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^for.body4
    - simplify-cfg: ^for.guards7 merged into ^for.end6
    - simplify-cfg: ^for.setup8 merged into ^for.end6
    - simplify-cfg: ^for.latch11 merged into ^for.body10
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
```

</details>

<!-- outputs:end -->

## The IR at each stage

```bash
ppy inspect toolbox.ppy --stage analysis   # what the checker knows of each function
ppy inspect toolbox.ppy --stage ir         # the frontend's module, before any pass
ppy inspect toolbox.ppy --stage optimized  # what the backend receives
ppy emit ir toolbox.ppy                    # the same module as .ppyir text
ppy emit c toolbox.ppy                     # one C11 translation unit; `cpp` for C++17
ppy emit llvm-ir toolbox.ppy               # what LLVM is handed
```

- `analysis` says of each function whether it is native and whether it gets
  a Python boundary. `unused` is native but "native callers only", because a
  two-instruction body is not worth a boundary crossing.
- `emit c` writes a translation unit that compiles alone with any C11
  compiler. It answers what the LLVM road answers, guards and fallbacks
  included.

## The optimization report

```bash
ppy build toolbox.ppy --report-opt
ppy build toolbox.ppy --report-opt-json report.json
```

The report first says whether each function is native, and why. Then it
lists every remark the passes left, by stable category (`block merged`,
`dead code removed`, `function inlined`, `bounds guard removed`), so a tool
can count them across versions.

## Sanitizers

```bash
ppy run --sanitize bounds,overflow toolbox.ppy
```

A sanitizer instruments the IR with checks the program did not ask for: every
buffer index, every wrapping or proven integer operation. A check that fails
is not a fallback. The function returns a sanitizer status, and the boundary
raises `SanitizerFailure` naming the kind and the function.

## Profile-guided optimization

```bash
ppy run --profile toolbox.ppy              # runs, then writes toolbox.ppyprof
ppy build --pgo toolbox.ppyprof toolbox.ppy --report-opt
ppy run --pgo toolbox.ppyprof toolbox.ppy
```

The profiling run counts every block and the taken edge of every branch. It
also records what each native function was called with (`list[400]`, fifty
times).

The guided build annotates what still matches:

- `compute` is hot.
- `unused` is cold and never inlined.
- The branch on `x % 4` carries the weights the run measured, and LLVM
  receives them as `!prof` metadata.

The report lists the profile first. A function edited since the profile was
recorded is named (`W2009`) and built as without one. A profile changes what
is fast, never what is computed.

Read on: [CLI](../../docs/cli.md) · [The IR](../../docs/internals/ir.md)

`toolbox.ppy` is hand-written; there is no `.py` source and no conversion step.
