# A dataclass with no boxed form

`Vec3` is three floats. Native code treats it as three floats: a call to
`distance2(a, b)` passes six doubles in registers, not two object pointers,
and `a.norm2()` is a native function whose `self` is three scalars. Eight
float operations cost about 50 ns per call from Python here, most of it the
boundary.

## Flattened at the ABI, guarded on the class

```python
@dataclass
class Vec3:
    x: float
    y: float
    z: float

    @ppy.pure
    @ppy.opt(3)
    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z
```

An all-scalar `@dataclass` is a value class. The generated boundary reads
its fields, checks the class is exactly `Vec3`, and calls the native
function with scalars. `distance2(Tracked(1.0, 2.0, 3.0), b)` shows the
guard: `Tracked` is a subclass, the exact-class check refuses it, and the
Python body answers. A class too wide for the ABI stays boxed, and
`ppy explain` says so.

## Operators dispatch statically

Inside native code `a + b` on a value class calls the class's own
`__add__`, lowered like any other native function. There is no fallback to
Python's dynamic dispatch and no `__radd__` search; a class without a
native operator is refused with the reason. `steps` runs a 200-iteration
loop over a `Ray`'s two fields at about 200 ns per call — the loop is
native, the `Ray` never leaves the registers.

## Run it

```bash
python  value_classes.ppy
ppy     value_classes.ppy
ppy run value_classes.ppy
```

<!-- outputs:start -->
## What it prints

**`python  value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      61.5 ns/call
steps     (200 iters)      4644.5 ns/call
```

**`ppy     value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      63.7 ns/call
steps     (200 iters)      5009.5 ns/call
```

**`ppy run value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      63.6 ns/call
steps     (200 iters)       368.5 ns/call
```

<!-- outputs:end -->

## Read on

- [Generics](../40_generics/README.md) — generics over value classes, monomorphized.
- [Native data](../08_native_data/README.md) — which values have a native representation.

`value_classes.ppy` is hand-written; there is no `.py` source and no
conversion step.
