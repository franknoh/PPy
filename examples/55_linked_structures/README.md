# Linked structures

A search tree with parent links and a singly linked list, written the way
TheAlgorithms/Python writes its data structures: classes with no
annotations, fields set to `None` in `__init__` and linked up by methods,
and the work in a function the main block calls. The folder's
`pyproject.toml` turns strict mode off. `ppy run` types every field from
what the program stores into it, and runs the methods that relink the
nodes as native code.

## Run it

```bash
python  trees.ppy
ppy run trees.ppy
ppy explain trees.SearchTree.rotate_up
ppy explain --summary trees.ppy
```

<!-- outputs:start -->
## What it prints

**`python  trees.ppy`**

```text
found, depth of 0, total: (207200, 1, 1249975000)
# workload: 0.62 s
reversed in place: 10 5
```

**`ppy run trees.ppy`**

```text
found, depth of 0, total: (207200, 1, 1249975000)
# workload: 0.29 s
reversed in place: 10 5
```

**`ppy explain trees.SearchTree.rotate_up`**

```text
function: rotate_up
qualname: trees.SearchTree.rotate_up
semantic type: (trees.SearchTree, trees.Node) -> NoneType
effects: ReadObject, WriteObject
purity: impure
optimization: O2
python backend: optimized
llvm backend: native; called from Python, its Python body runs: copying the collections in costs more than the body does with them
jit: not requested
parallel: rejected
reason: the function contains no parallelizable loop
representation:
  self: trees.SearchTree -> LLVM aggregate (guarded)
  node: trees.Node -> LLVM aggregate (guarded)
  return: NoneType -> void
inferred (not annotated):
  node: trees.Node, from 1 call (trees.ppy:80)
  (the Python boundary checks these at each call, and runs the Python body otherwise)
  return: NoneType, from the body's return statements
```

**`ppy explain --summary trees.ppy`**

```text
12 functions, 94 statements
  native, called from Python              1 functions (  8%)       16 statements ( 17%)
  native, called from native code        11 functions ( 92%)       78 statements ( 83%)
  Python                                  0 functions (  0%)        0 statements (  0%)

native, but Python calls the Python body (why its boundary is not used):
     11 functions  copying the collections in costs more than the body does with them
```

<!-- outputs:end -->

## The program

- `Node(key)` has a `key` and three links, `left`, `right`, and `parent`,
  all `None` at first.
- `SearchTree.insert(key)` walks down from the root, hangs a new node on
  the first empty side, and links it back to its parent.
- `SearchTree.find(key)` walks down and returns the node or `None`.
- `SearchTree.rotate_up(node)` rotates `node` above its parent: it moves
  one subtree across, swaps the two nodes' places, and fixes the three
  parent links and the root or the grandparent's child.
- `SearchTree.access(key)` finds a key and rotates its node up to the
  root, one rotation at a time, so keys that are asked for often stay near
  the top (the "move to root" heuristic).
- `LinkedList` reuses `Node`: `push` links a new node in front of the head
  through `right`, `reverse` relinks every node in place, and `total` walks
  the list.
- `workload(count, seed)` inserts `count` keys from a linear congruential
  generator, accesses `20 * count` keys of a small range, pushes `count`
  values onto a list and reverses it 100 times, and returns three numbers.
- The main block runs `workload(50_000, 7)`, then builds a five-node list
  in Python, reverses it, and prints its total.

## Where the field types come from

No field is annotated, and every link starts as `None`. In strict mode
`ppy check trees.ppy` reports 15 errors: nine `E1201` for the parameters,
and six `E1202` for reads of `key` through a node whose type it could not
settle. With `strict = false` the compiler reads every store into a field
anywhere in the project, not only `__init__`'s:

- `here.left = node` in `insert`, with `node = Node(key)`, adds `Node` to
  `left`'s `None`, so `Node.left` is a `Node | None`. `right` and `parent`
  get the same from `insert`, `rotate_up`, and `push`.
- `self.root = node` makes `SearchTree.root` a `Node | None`, and
  `self.head = node` does the same for `LinkedList.head`.
- `self.size += 1` and `self.length += 1` from `0` make `int` fields.
- `key` is an `int` from the calls that make nodes: `Node(key)` in
  `insert`, whose `key` comes from `tree.insert(state % (count * 4))`.

With the fields typed, every method is an ordinary native function over
`Node` handles. `rotate_up`'s `node` is a `Node` from its one call in
`access`, and `ppy explain` (below) lists it.

Two shapes common in such code do not settle, and the
[Classes](../../docs/guide/classes.md#fields-without-annotations) guide
says why: a recursive `insert(node, key)` that stores its own result
(`node.left = insert(node.left, key)`) and is not annotated, and a local
that starts as `None` and later holds a node (`prev = None` in a list
reversal). `reverse` here starts from `self.head = None` instead, which is
a field, so it settles.

## What crosses

`workload` takes two ints and returns a tuple of three, so the call from
Python crosses three numbers. The tree of 50,000 nodes and the list are
made, relinked, and freed inside native code, and every method call is a
native call between native functions.

The methods themselves are native, but when Python calls one, as the main
block does with `small.reverse()`, it runs the Python body: an object
crosses by copy with every object it reaches, which costs more than one
pass over five nodes saves. `ppy explain --summary` counts them as native
and called from native code, and says so.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python trees.ppy` | 0.65 |
| `ppy run trees.ppy`, after the first run built the cache | 0.30 |

`workload(50_000, 7)` alone, as the program prints it: 0.63 s under
CPython and 0.27 s under `ppy run`.

There is no standalone build: the program has no `main()`, and a
standalone binary needs a native `main` to start from.

## Where the code comes from

`trees.ppy` is hand-written.

Read on: [Classes](../../docs/guide/classes.md#fields-without-annotations)
and [Types from call sites](../../docs/guide/subset.md#types-from-call-sites).
