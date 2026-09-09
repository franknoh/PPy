# The toolbox

One small program, and every way of looking inside its build: the IR at each
stage, the source backends, the optimization report, the sanitizers, and
profile-guided optimization. The program is ordinary — a loop with a
branch, an indexed read, a function nobody calls — so that the tools have
something to say, and it prints the same line under every command below.

## The IR at each stage

```bash
ppy inspect toolbox.ppy --stage analysis   # what the checker knows of each function
ppy inspect toolbox.ppy --stage ir         # the frontend's module, before any pass
ppy inspect toolbox.ppy --stage optimized  # what the backend receives
ppy emit ir toolbox.ppy                    # the same module as .ppyir text
ppy emit c toolbox.ppy                     # one C11 translation unit; `cpp` for C++17
ppy emit llvm-ir toolbox.ppy               # what LLVM is handed
```

`analysis` says of each function whether it is native and whether it gets a
Python boundary: `unused` is native but "native callers only", because a
two-instruction body is not worth a boundary crossing. `emit c` writes a
translation unit that compiles alone with any C11 compiler and answers what
the LLVM road answers, guards and fallbacks included.

## The report

```bash
ppy build toolbox.ppy --report-opt
ppy build toolbox.ppy --report-opt-json report.json
```

Native or not and why, then every remark the passes left, by stable
category — `block merged`, `dead code removed`, `function inlined`, `bounds
guard removed` — so a tool can count them across versions.

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
recorded is named (`W2009`) and built as without one. A profile changes what
is fast, never what is computed.

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

<details markdown="1">
<summary>82 lines</summary>

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

func @toolbox_pick(%xs: buffer<i64> {ownership = "owned", ppy.kind = "list"}, %i: i64) -> i64 attrs {effects = ["may_raise", "read_memory"], ppy.abi = "ppy", ppy.qualname = "toolbox.pick", ppy.releases_gil = true, ppy.symbol = "ppy_toolbox_pick"} loc("examples/42_toolbox/toolbox.ppy":11:0) {
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
    core.guard %6 {kind = "bounds", message = "index out of range"}
    %7 = core.buffer_load %xs, %0 : i64
    %8 = core.const 2 : i64
    %9 = core.mul %7, %8 {overflow = "python"} : i64
    core.ret %9
}

func @toolbox_unused(%n: i64) -> i64 attrs {effects = [], ppy.abi = "ppy", ppy.qualname = "toolbox.unused", ppy.releases_gil = true, ppy.symbol = "ppy_toolbox_unused"} loc("examples/42_toolbox/toolbox.ppy":15:0) {
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
<summary>190 lines</summary>

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
    *out = a * b;
    return 0;
#endif
}

int32_t ppy_toolbox_compute(int64_t *a0, int64_t a1, int64_t *out0);
int32_t ppy_toolbox_pick(int64_t *a0, int64_t a1, int64_t a2, int64_t *out0);
int32_t ppy_toolbox_unused(int64_t a0, int64_t *out0);

int32_t ppy_toolbox_compute(int64_t *a0, int64_t a1, int64_t *out0) {
    int64_t v1_v;
    int64_t v2_total_addr[1];
    int64_t v3_xs_i_addr[1];
    int64_t v4_v;
    int64_t v5_x_addr[1];
    int64_t v6_v;
    int64_t v7_v;
    int64_t v8_v;
    int64_t v9_v;
    int64_t v10_v;
    int64_t v11_v;
    bool v12_v;
    int64_t v13_v;
    int64_t v14_v;
    int64_t v15_v;
    int64_t v16_r;
    int64_t v17_v;
    bool v18_v;
    int64_t v19_v;
    int64_t v20_v;
    int64_t v21_v;
    int64_t v22_v;
    int64_t v23_v;
    int64_t v24_v;
    int64_t v25_v;
    int64_t v26_v;
    int64_t v27_v;
    v1_v = INT64_C(0);
    *v2_total_addr = v1_v;
    v4_v = INT64_C(0);
    *v3_xs_i_addr = v4_v;
    v6_v = a1;
    v7_v = ((int64_t)(v6_v));
    v8_v = INT64_C(4);
    v9_v = INT64_C(0);
    v10_v = INT64_C(1);
    goto L1_each_head1;
L1_each_head1:;
    v11_v = *v3_xs_i_addr;
    v12_v = v11_v < v7_v;
    if (v12_v) {
        goto L2_each_body2;
    } else {
        goto L3_each_end4;
    }
L2_each_body2:;
    v13_v = *v3_xs_i_addr;
    v14_v = a0[v13_v];
    *v5_x_addr = v14_v;
    v15_v = *v5_x_addr;
    if (!(!(v15_v == INT64_MIN && v8_v == -1))) goto fallback; /* div.ok */
    v16_r = v15_v % v8_v;
    v17_v = (v16_r != 0 && ((v15_v < 0) != (v8_v < 0))) ? v16_r + v8_v : v16_r;
    v18_v = v17_v == v9_v;
    if (v18_v) {
        goto L4_then5;
    } else {
        goto L5_else6;
    }
L3_each_end4:;
    v19_v = *v2_total_addr;
    *out0 = v19_v;
    return 0;
L4_then5:;
    v20_v = *v2_total_addr;
    v21_v = *v5_x_addr;
    v22_v = 0;
    if (!(!ppy_ovf_add_i64(v20_v, v21_v, &v22_v))) goto fallback; /* arith.ok */
    *v2_total_addr = v22_v;
    goto L6_endif7;
L5_else6:;
    v23_v = *v2_total_addr;
    v24_v = 0;
    if (!(!ppy_ovf_sub_i64(v23_v, v10_v, &v24_v))) goto fallback; /* arith.ok */
    *v2_total_addr = v24_v;
    goto L6_endif7;
L6_endif7:;
    v25_v = INT64_C(1);
    v26_v = *v3_xs_i_addr;
    v27_v = ((int64_t)(((uint64_t)(v26_v)) + ((uint64_t)(v25_v))));
    *v3_xs_i_addr = v27_v;
    goto L1_each_head1;
fallback:
    return 1;
}


int32_t ppy_toolbox_pick(int64_t *a0, int64_t a1, int64_t a2, int64_t *out0) {
    int64_t v1_i_addr[1];
    int64_t v2_v;
    int64_t v3_i_entry;
    int64_t v4_v;
    int64_t v5_v;
    int64_t v6_v;
    bool v7_v;
    bool v8_v;
    bool v9_v;
    int64_t v10_v;
    int64_t v11_v;
    int64_t v12_v;
    *v1_i_addr = a2;
    v2_v = *v1_i_addr;
    v3_i_entry = *v1_i_addr;
    (void)v3_i_entry;
    v4_v = INT64_C(0);
    v5_v = a1;
    v6_v = ((int64_t)(v5_v));
    v7_v = v2_v >= v4_v;
    v8_v = v2_v < v6_v;
    v9_v = v7_v & v8_v;
    if (!(v9_v)) goto fallback; /* bounds.ok */
    v10_v = a0[v2_v];
    v11_v = INT64_C(2);
    v12_v = 0;
    if (!(!ppy_ovf_mul_i64(v10_v, v11_v, &v12_v))) goto fallback; /* arith.ok */
    *out0 = v12_v;
    return 0;
fallback:
    return 1;
}


int32_t ppy_toolbox_unused(int64_t a0, int64_t *out0) {
    int64_t v1_n_addr[1];
    int64_t v2_v;
    int64_t v3_n_entry;
    int64_t v4_v;
    int64_t v5_v;
    *v1_n_addr = a0;
    v2_v = *v1_n_addr;
    v3_n_entry = *v1_n_addr;
    (void)v3_n_entry;
    v4_v = INT64_C(7);
    v5_v = 0;
    if (!(!ppy_ovf_mul_i64(v2_v, v4_v, &v5_v))) goto fallback; /* arith.ok */
    *out0 = v5_v;
    return 0;
fallback:
    return 1;
}
```

</details>

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

<details markdown="1">
<summary>22 lines</summary>

```text
optimization report: PPy (O2, ir road)
profile: toolbox.ppyprof (4 runs, hot from 10 calls)
  toolbox.compute: hot, 200 calls; arguments 0: list[400] x200
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
    - profile: `toolbox.compute` is hot (200 call(s)); 2 branch(es) weighted, 1 loop(s) with trip counts
    - profile: `toolbox.pick` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - profile: `toolbox.unused` is cold (0 call(s)); 0 branch(es) weighted, 0 loop(s) with trip counts
    - simplify-cfg: ^each.latch3 merged into ^endif7
    - dce: unused core.const removed
    - dce: unused core.const removed
```

</details>

<!-- outputs:end -->

Read on: [CLI](../../docs/cli.md) · [The IR](../../docs/internals/ir.md)

`toolbox.ppy` is hand-written; there is no `.py` source and no conversion step.
