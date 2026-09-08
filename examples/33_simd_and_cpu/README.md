# Lanes and the machine

`ppy.simd` is a few scalars operated on at once; `ppy.cpu` is the machine
as a facade with no instruction named. Both have a reference implementation
under CPython and a lowering to a dialect of the IR.

## Provenance

Hand-written. `lanes.ppy` is written directly; there is no `.py` source and
no conversion step involved.

## What it shows

- `simd.Vector[T, N]` is `N` lanes of `T`; `splat`, `load`, `store`,
  `insert`, `extract`, `shuffle`, `select`, and the reductions are the whole
  vocabulary, and `+ - * /`, `& | ^`, and the comparisons work lane by lane.
- Integer lanes wrap at their width: the ramp that starts three below the
  largest `int` wraps exactly as the machine would, on every path.
- A floating-point `reduce_add` folds in lane order, first to last, so the
  sum is the same number everywhere.
- `cpu.prefetch` and `cpu.pause` are hints and change no value;
  `cpu.vector_width[float]()` and `cpu.features()` are facts about the
  machine compiling, folded to constants natively. The two lines that print
  them start with `# `, which is how an example marks what is allowed to
  differ between machines.

## Run it

```bash
python  lanes.ppy
ppy run lanes.ppy
ppy emit ir lanes.ppy   # the simd and cpu dialects
```

<!-- outputs:start -->
## What it prints

**`python  lanes.ppy`**

```text
10.0
18446744073709551615 0
-6.0 2.0 2.0
28 1.5
# vector width for float: 4 lanes
# avx2 here: True
```

**`ppy run lanes.ppy`**

```text
10.0
18446744073709551615 0
-6.0 2.0 2.0
28 1.5
# vector width for float: 4 lanes
# avx2 here: True
```

**`ppy emit ir lanes.ppy`**

```text
ppyir 1
module @lanes
dialect core 1
dialect cpu 1
dialect simd 1

func @lanes_dot(%a: ptr<f64, generic, const>, %b: ptr<f64, generic, const>) -> f64 attrs {effects = ["read_memory"], ppy.abi = "ppy", ppy.qualname = "lanes.dot", ppy.releases_gil = true, ppy.symbol = "ppy_lanes_dot"} loc("examples/33_simd_and_cpu/lanes.ppy":5:0) {
^entry:
    %a_addr = core.alloca : ptr<ptr<f64, generic, const>, stack> loc("examples/33_simd_and_cpu/lanes.ppy":5:0)
    core.store %a, %a_addr
    %b_addr = core.alloca : ptr<ptr<f64, generic, const>, stack>
    core.store %b, %b_addr
    %0 = core.load %a_addr : ptr<f64, generic, const> loc("examples/33_simd_and_cpu/lanes.ppy":6:4)
    %1 = simd.load %0 : vector<f64, 4>
    %2 = core.load %b_addr : ptr<f64, generic, const>
    %3 = simd.load %2 : vector<f64, 4>
    %4 = core.mul %1, %3 : vector<f64, 4>
    %5 = simd.reduce_add %4 : f64
    core.ret %5
}

func @lanes_ramp(%p: ptr<i64>, %start: i64) -> i64 attrs {effects = ["write_memory"], ppy.abi = "ppy", ppy.qualname = "lanes.ramp", ppy.releases_gil = true, ppy.symbol = "ppy_lanes_ramp"} loc("examples/33_simd_and_cpu/lanes.ppy":9:0) {
^entry:
    %p_addr = core.alloca : ptr<ptr<i64>, stack> loc("examples/33_simd_and_cpu/lanes.ppy":9:0)
    core.store %p, %p_addr
    %start_addr = core.alloca : ptr<i64, stack>
    core.store %start, %start_addr
    %0 = core.load %start_addr : i64 loc("examples/33_simd_and_cpu/lanes.ppy":10:4)
    %start_entry = core.load %start_addr : i64
    %1 = simd.splat %0 : vector<i64, 4>
    %v_addr = core.alloca : ptr<vector<i64, 4>, stack>
    core.store %1, %v_addr
    %2 = core.load %v_addr : vector<i64, 4> loc("examples/33_simd_and_cpu/lanes.ppy":11:4)
    %3 = core.const 1 : i64
    %4 = core.load %start_addr : i64
    %5 = core.const 1 : i64
    %6 = core.add %4, %5 {overflow = "python"} : i64
    %7 = simd.insert %2, %6, %3 : vector<i64, 4>
    core.store %7, %v_addr
    %8 = core.load %v_addr : vector<i64, 4> loc("examples/33_simd_and_cpu/lanes.ppy":12:4)
… 116 more lines
```

<!-- outputs:end -->
