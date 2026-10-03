# A clinic simulation

A seeded simulation of a clinic over 2,000 days, run three times with six,
seven, and eight doctors. Patients arrive at random, wait in order of
urgency, and are seen by the next free doctor. It uses `random`, `heapq`,
and `math`, a table of constants at module level, and two dataclasses that
cross between Python and native code under `ppy run`.

## Run it

```bash
python  clinic.ppy
ppy run clinic.ppy
ppy build --standalone clinic.ppy -o dist && ./dist/clinic
```

<!-- outputs:start -->
## What it prints

**`python  clinic.ppy`**, **`ppy run clinic.ppy`**, **`ppy build --standalone clinic.ppy -o dist && ./dist/clinic`**

```text
6 doctors:
  day   0: 395 seen, 202 late, longest wait 382.6 min
  day 500: 442 seen, 266 late, longest wait 486.3 min
  day 1000: 403 seen, 179 late, longest wait 362.4 min
  day 1500: 378 seen, 193 late, longest wait 382.4 min
  2000 days, 799475 patients, 55.67% late, mean wait 180.7 min, worst 632.7 min
7 doctors:
  day   0: 395 seen, 109 late, longest wait 201.3 min
  day 500: 442 seen, 254 late, longest wait 344.2 min
  day 1000: 403 seen, 29 late, longest wait 138.8 min
  day 1500: 378 seen, 78 late, longest wait 174.0 min
  2000 days, 799475 patients, 39.34% late, mean wait 97.0 min, worst 504.6 min
8 doctors:
  day   0: 395 seen, 0 late, longest wait 38.5 min
  day 500: 442 seen, 197 late, longest wait 201.4 min
  day 1000: 403 seen, 0 late, longest wait 80.2 min
  day 1500: 378 seen, 0 late, longest wait 60.9 min
  2000 days, 799475 patients, 10.75% late, mean wait 40.2 min, worst 364.4 min
```

<!-- outputs:end -->

## The program

- `SHARE`, `CARE`, and `LATE` are tuples of three floats at module level,
  one entry per triage level, most urgent first: the share of arrivals,
  the mean minutes a doctor takes, and how long a patient may wait before
  the wait counts as late.
- `Clinic` is a dataclass of three numbers: doctors, arrivals per hour, and
  the seed. `Report` is a dataclass that adds up the days; `add_day`
  writes its fields.
- `one_day` draws the day's arrivals with `random.expovariate`, gives each
  a level from `random.random()` through `level_of`, and pushes them onto
  a heap of events. It then pops events in time order. A waiting patient
  goes on a second heap ordered by level and arrival time. Each time a
  doctor is free, the most urgent patient is seen for a time drawn from
  `random.lognormvariate` around `math.log(CARE[level])`. The day goes into
  the report, and every 500th day prints a line.
- `run(clinic, days=30)` seeds the generator from the clinic, makes a
  `Report`, runs the days, and returns the report.
- `main` makes a `Clinic` for each number of doctors, calls
  `run(clinic, days=2000)`, and prints the share of late patients and the
  waits. Every run draws the same arrivals, since every clinic has the same
  seed, so the three lines differ only in the number of doctors.

## How it runs natively

`ppy explain clinic.level_of`, `clinic.one_day`, `clinic.run`, and
`clinic.Report.add_day` each say `llvm backend: native`.

- `SHARE[level]` reads a module-level tuple with an index computed at run
  time. A tuple of numbers bound once at module level is a constant: native
  code builds it once per call and picks the item by comparing the index
  with each position, after the same bounds check CPython makes. Being a
  constant, it also reaches the standalone binary.
- `random.expovariate`, `random.random`, and `random.lognormvariate` are
  CPython's algorithms on CPython's Mersenne Twister. Under `ppy run`,
  native code draws from the generator Python's `random` uses, so the
  seed in `run` gives the same numbers on every path.
- `heapq.heappush` and `heappop` on lists of tuples make the same
  comparisons as CPython, so the heaps pop in the same order and ties
  break the same way.
- `main` returns nothing, so it has no Python boundary of its own: under
  `ppy run`, Python runs `main`, and `run` is the native call it makes.
  The `Clinic` argument crosses as its three fields. `days=2000` is a
  keyword argument, which the boundary binds the way Python binds it,
  with the default for anything left out. The `Report` that native code
  made comes back as a new `Report` instance with its fields set.
- `one_day(clinic, report, day)` leaves `hours` to its default. Native
  callers bind keywords and constant defaults when the module is compiled.
- The lines `one_day` prints are held while `run` runs and written when it
  returns, in the order they were printed.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python clinic.ppy` | 2.76 |
| `ppy run clinic.ppy`, after the first run built the cache | 1.14 |
| `./dist/clinic`, the standalone binary | 1.12 |

`ppy run` also starts CPython, names the cached build from the project's
sources, and loads its native library, which the standalone binary does
not pay.

## Where the code comes from

`clinic.ppy` is hand-written.

Read on: [the standard library guide](../../docs/guide/stdlib.md) and
[classes at the boundary](../../docs/guide/classes.md).
