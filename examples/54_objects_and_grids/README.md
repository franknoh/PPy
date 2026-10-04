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

<!-- outputs:start -->
## What it prints

**`python  world.ppy`**, **`ppy run world.ppy`**

```text
generation  40: 2349 alive (2349 by Python's count), 53643 born, 56133 died
generation  80: 1739 alive (1739 by Python's count), 31597 born, 32207 died
generation 120: 1451 alive (1451 by Python's count), 23478 born, 23766 died
generation 160: 1372 alive (1372 by Python's count), 20657 born, 20736 died
generation 200: 1160 alive (1160 by Python's count), 19690 born, 19902 died
energy at the start: -3453.615764
step  400: energy -3439.215631, first body at (0.1334, -0.0402)
step  800: energy -3451.679512, first body at (0.4097, 0.4738)
step 1200: energy -3448.311153, first body at (-0.1150, -0.6762)
step 1600: energy -3454.178277, first body at (-0.0289, -0.6484)
step 2000: energy -3450.191775, first body at (0.0380, -0.6717)
the first body is the same object: True
```

**`ppy explain --summary world.ppy`**

```text
10 functions, 77 statements
  native, called from Python              3 functions ( 30%)       39 statements ( 51%)
  native, called from native code         6 functions ( 60%)       25 statements ( 32%)
  Python                                  1 functions ( 10%)       13 statements ( 17%)

what keeps functions in Python, by statements kept out (a function can count under more than one):
       13 statements      1 functions  `…` yields values with no native form
      examples/54_objects_and_grids/world.ppy:95 world.main

native, but Python calls the Python body (why its boundary is not used):
      6 functions  copying the collections in costs more than the body does with them
```

<!-- outputs:end -->

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
  128 rows into native memory, runs, and then sets back into each row only
  the cells whose value changed. `grid` is still the caller's object, and
  so is each row, so `sum(map(sum, grid))` in `main` counts what native
  code left there. A cell the call did not change keeps its object.
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
| `python world.ppy` | 2.20 |
| `ppy run world.ppy`, after the first run built the cache | 0.25 |

Timed inside the program, the mean of five runs:

| part | CPython | `ppy run` |
|---|---:|---:|
| five `advance(grid, 40)` calls | 1.51 s | 0.177 s |
| five `orbit(bodies, 0.0005, 400)` calls | 0.685 s | 0.032 s |
| one `orbit(bodies, 0.0005, 1)` call, 60 bodies in and out | 346 µs | 29 µs |

There is no standalone build: `main` stays in Python, and the point of
the example is the crossing, which a standalone binary does not have.

## Where the code comes from

`world.ppy` is hand-written.

Read on: [Lists, dicts, and sets](../../docs/guide/containers.md) and
[classes at the boundary](../../docs/guide/classes.md).
