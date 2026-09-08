# Four lanes at a time, and the machine as a fact

`ppy.simd` is a few scalars operated on at once; `ppy.cpu` is the machine
as a facade with no instruction named. A `Vector[float, 4]` dot product,
an integer ramp that wraps exactly where the hardware would, a shuffle and
a select, a byte mask, a prefetch hint — each with a reference
implementation under CPython and a lowering to the simd and cpu dialects of
the IR, so the numbers match on every path down to the wrap.

## The whole vocabulary in four functions

```python
def dot(a: native.const_ptr[float], b: native.const_ptr[float]) -> float:
    return simd.reduce_add(simd.load[float, 4](a) * simd.load[float, 4](b))


def mix(p: native.ptr[float]) -> float:
    v = simd.load[float, 4](p)
    w = simd.shuffle(v, v, (3, 2, 1, 0))
    chosen = simd.select(v > w, v, w)
    simd.store(chosen / simd.splat[float, 4](2.0), p)
    return simd.extract(chosen, 0) + simd.reduce_add(-w)
```

`splat`, `load`, `store`, `insert`, `extract`, `shuffle`, `select`, and
the three reductions are the entire surface. `+ - * /`, `& | ^`, and the
comparisons work lane by lane; a comparison gives a `Vector[bool, N]` for
`select`. A floating-point `reduce_add` folds in lane order, first to last,
so the sum is one number everywhere, not a number close to it.

## Integer lanes wrap at their width

`ramp` starts three below the largest `int` and adds a lane-wise `v + v`.
Vector integer arithmetic carries `wrap` semantics, exactly as the machine
would — there is no unbounded integer in a register — and the reference
implementation wraps the same way, so `python` and `ppy run` print the same
wrapped value. `bytes_sum` masks eight `u8` lanes with `& 15` and reduces.

## The machine, folded to constants

`cpu.prefetch(p, locality=2)` and `cpu.pause()` are hints and change no
value. `cpu.vector_width[float]()` and `"avx2" in cpu.features()` are facts
about the machine compiling, folded to constants natively. The two lines
that print them start with `# `, which is how an example marks output that
is allowed to differ between machines.

## Run it

```bash
python  lanes.ppy
ppy run lanes.ppy
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

<!-- outputs:end -->

## Read on

- [Lanes and the machine](../../docs/guide/simd-cpu.md) — `ppy.simd` and `ppy.cpu` in full.
- [The IR: the simd and cpu dialects](../../docs/internals/ir.md) — what the operations lower to.

`lanes.ppy` is hand-written; there is no `.py` source and no conversion step.
