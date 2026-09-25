# Inheritance

An expression tree whose node classes derive from one another, and a 2D
vector class with arithmetic operators, all compiled to native code with
method calls dispatched on each object's class.

## Run it

```bash
python  shapes.ppy
ppy run shapes.ppy
ppy build --standalone shapes.ppy -o dist && ./dist/shapes
```

<!-- outputs:start -->
## What it prints

**`python  shapes.ppy`**, **`ppy run shapes.ppy`**, **`ppy build --standalone shapes.ppy -o dist && ./dist/shapes`**

```text
240837 1001001
```

<!-- outputs:end -->

## The program

- `Expr` is a dataclass with `eval` and `size`. `Num` derives from it and
  holds a value; `Add` holds a left and a right `Expr | None`; `Mul` derives
  from `Add` and overrides only `eval`.
- `trees` builds 2,000 trees ten levels deep and walks each one with
  `eval`, `size`, and `count_sums`. `tree.eval()` on an `Expr` runs `Num`'s,
  `Add`'s, or `Mul`'s method, whichever class the node was made as.
  `count_sums` asks `isinstance(e, Mul)` before `isinstance(e, Add)`, since
  a `Mul` is also an `Add`.
- `V2` defines `__add__`, `__sub__`, `__mul__`, and `__lt__`. `orbit` steps
  a body around a center two million times with `position + velocity * dt`
  and keeps the farthest point with `<`.

## How it runs natively

`ppy explain shapes.trees`, `shapes.orbit`, and the methods each say
`llvm backend: native`.

- A `Num`, `Add`, or `Mul` is one record: `Expr`'s field first, then each
  subclass's own. Its header holds a tag for the class it was made as.
- A call to a method some subclass overrides reads the tag and calls that
  class's method. `size` is overridden by `Add` only, so a call on a `Mul`
  runs `Add.size`.
- `isinstance` compares the tag with the tags of the class and the classes
  deriving from it.
- `V2`'s operators are calls to its methods. Each returns a new `V2`, so `V2`
  is held by handle like the tree nodes, and each intermediate vector is
  freed when its last reference goes.

## Timing

One machine, wall time for the whole program:

| | seconds |
|---|---:|
| `python shapes.ppy` | 5.5 |
| `ppy run shapes.ppy`, after the first run built the cache | 2.3 |
| `./dist/shapes`, the standalone binary | 0.61 |

`ppy run` includes the compiler's own start-up and checking the file, which
the standalone binary does not pay.

## Where the code comes from

`shapes.ppy` is hand-written.

Read on: [the classes guide](../../docs/guide/classes.md).
