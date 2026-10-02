# Objects and grids across the boundary

Python code owns the data here: a 128 by 128 grid of ints for the Game of
Life, and sixty `Body` objects for a gravity simulation. It hands them to
native functions that change them in place, and reads them back between
calls. Under `ppy run` the generated wrapper copies the grid and the
objects in, and after each call copies what the call wrote back into
Python's own lists and objects.

## Run it

```bash
python  world.ppy
ppy run world.ppy
ppy explain --summary world.ppy
```

## The program

- `Census` counts cells: `count(was, now)` adds one birth or death and one
  living cell. `Body` holds a position, a velocity, and a mass, and its
  `kick` and `drift` methods change the velocity and the position. Methods
  that change fields make both of them object classes: native code holds
  them by handle, as Python holds them by reference.
- `advance(grid, generations)` runs the Game of Life on a torus. Each
  generation builds the next grid, counts it into a `Census`, and copies it
  over `grid` cell by cell. It returns the `Census`.
- `orbit(bodies, dt, steps)` moves the bodies by `steps` steps of size
  `dt`: each body's acceleration from every other body, `kick`, then
  `drift`. It returns nothing; its result is what it did to the bodies.
- `energy(bodies)` adds up the kinetic and potential energy.
- `main` stays in Python (`sum(map(sum, grid))` has no native form). It
  fills the grid from `random.Random(3)` and calls `advance` five times,
  printing the census next to its own count of the living cells. Then it
  makes the bodies, keeps `first = bodies[0]`, and calls `orbit` five
  times, printing the energy and where `first` is.

## What crosses, and how

`ppy explain --summary world.ppy` counts `advance`, `orbit`, and `energy`
as native and called from Python, and the rest as native code they call.

- **A list of lists written in place.** `advance(grid, 40)` copies the
  128 rows into native memory, runs, and copies each row back into the
  same Python list. `grid` is still the caller's object, and so is each
  row, so `sum(map(sum, grid))` in `main` counts what native code left
  there. A row that a call did not change keeps its elements as they were.
- **An object made natively.** The `Census` that `advance` returns was
  made by native code. It comes back as a new `Census` instance with its
  fields set; `__init__` does not run a second time.
- **Objects written in place.** `orbit(bodies, ...)` copies each `Body`'s
  fields in, and after the call sets the fields it wrote on the caller's
  objects. `first` and `bodies[0]` are still one object, so the last line
  prints `True` on every path. An object reached twice crosses once.
- **The cost of the copy.** The wrapper copies in C, from type tables
  emitted for each signature. Python calls a function natively only where
  the body does enough with what crosses: here, nine reads per cell and a
  pass over every pair of bodies per step. `kick`, `drift`, `count`, and
  `neighbours` are too small for that, and Python calling them would run
  their Python bodies; native callers call them directly.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python world.ppy` | @@PY@@ |
| `ppy run world.ppy`, after the first run built the cache | @@RUN@@ |

Timed inside the program:

| part | CPython | `ppy run` |
|---|---:|---:|
| five `advance(grid, 40)` calls | @@P1@@ | @@N1@@ |
| five `orbit(bodies, 0.0005, 400)` calls | @@P2@@ | @@N2@@ |
| one `orbit(bodies, 0.0005, 1)` call, 60 bodies in and out | @@P3@@ | @@N3@@ |

There is no standalone build: `main` stays in Python, and the point of
the example is the crossing, which a standalone binary does not have.

## Where the code comes from

`world.ppy` is hand-written.

Read on: [Lists, dicts, and sets](../../docs/guide/containers.md) and
[classes at the boundary](../../docs/guide/classes.md).
