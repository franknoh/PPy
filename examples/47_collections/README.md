# Collections

Five problems written with `ppy.Vec`, `ppy.Deque`, `ppy.Heap`,
`ppy.LinkedList`, `ppy.HashMap`, and `ppy.TreeSet`, with no pointers in sight.
Every function goes native, and the program prints the same thing on all
three paths.

## Run it

```bash
python  graphs.ppy
ppy run graphs.ppy
ppy build --standalone graphs.ppy -o dist && ./dist/graphs
```

<!-- outputs:start -->
## What it prints

**`python  graphs.ppy`**, **`ppy run graphs.ppy`**, **`ppy build --standalone graphs.ppy -o dist && ./dist/graphs`**

```text
68871776 18
27151 3276962 2
```

<!-- outputs:end -->

## The problems

- `shortest_paths`: Dijkstra over 200,000 nodes with four weighted edges
  each. The frontier is a `Heap[int]` holding `distance * NODES + node`, so
  one integer carries both.
- `hops`: breadth-first search over the same graph with a `Deque[int]`,
  reporting the farthest node in edges.
- `josephus`: 100,000 people in a circle, every seventh removed. The circle
  is a `LinkedList[int]`, walked by node id with `next` and cut with
  `remove`.
- `repeats`: counts 2,000,000 keys in a `HashMap[int, int]` with
  `get(key, 0) + 1`, then walks the keys in insertion order.
- `closest_gaps`: the smallest gap between 200,000 values as they arrive,
  using a `TreeSet[int]`'s `floor` and `ceiling` for each value's neighbors.

`dist` and `seen` are `Vec[int](NODES)`, which start with `NODES` zeros, as
`[0] * NODES` would.

## How it runs natively

`ppy explain graphs.shortest_paths` and the others each say
`llvm backend: native`. A collection that a function makes is a handle into
the C runtime in `ppy_runtime/collections.py`, freed when the function
returns. A check CPython would raise for, such as an empty `pop` or a
missing key, is a guard: under `ppy run` it hands the call back to Python,
which raises, and a standalone binary stops.

## Timing

One machine, five runs each, wall time for the whole program:

| | seconds |
|---|---:|
| `python graphs.ppy` (the reference classes under CPython) | 4.3 |
| `idiomatic.py`: the same algorithms with `list`, `deque`, `heapq`, `dict`, `bisect` | 2.3 |
| `ppy run graphs.ppy`, after the first run built the cache | 0.39 |
| `./dist/graphs`, the standalone binary | 0.19 |

The reference classes are slower than Python's built-ins because every
method is a Python call. `idiomatic.py` is there so the comparison is with
code a Python programmer would write. Its `josephus` pops from a list rather
than walking a linked list, which is the usual way to write it in Python.

## Where the code comes from

`graphs.ppy` and `idiomatic.py` are hand-written.

Read on: [the collections guide](../../docs/guide/collections.md).
