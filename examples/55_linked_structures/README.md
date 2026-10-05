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
ppy explain trees.Node
ppy explain --summary trees.ppy
```

<!-- outputs:start -->
## What it prints

**`python  trees.ppy`**

```text
found, depth of 0, total: (207200, 1, 1249975000)
# workload: 0.78 s
reversed in place: 10 5
```

**`ppy run trees.ppy`**

```text
found, depth of 0, total: (207200, 1, 1249975000)
# workload: 0.37 s
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
llvm backend: native; called from Python, its Python body runs: the boundary crossing costs more than the body saves
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

**`ppy explain trees.Node`**

<details markdown="1">
<summary>27 lines</summary>

```text
class: Node
qualname: trees.Node
fields:
  left: NoneType | trees.Node, from what the program stores into it:
    NoneType at trees.ppy:7 in `__init__`
    NoneType | trees.Node at trees.ppy:57
    trees.Node at trees.ppy:65
    trees.Node at trees.ppy:71
    trees.Node at trees.ppy:35
  right: NoneType | trees.Node, from what the program stores into it:
    NoneType at trees.ppy:8 in `__init__`
    NoneType | trees.Node at trees.ppy:99
    trees.Node at trees.ppy:60
    NoneType | trees.Node at trees.ppy:62
    NoneType | trees.Node at trees.ppy:108
    trees.Node at trees.ppy:73
    trees.Node at trees.ppy:40
  parent: NoneType | trees.Node, from what the program stores into it:
    NoneType at trees.ppy:9 in `__init__`
    trees.Node at trees.ppy:43
    trees.Node at trees.ppy:66
    NoneType | trees.Node at trees.ppy:67
    trees.Node at trees.ppy:59
    trees.Node at trees.ppy:64
  key: int, from what the program stores into it:
    int at trees.ppy:6 in `__init__`
methods: __init__
```

</details>

**`ppy explain --summary trees.ppy`**

```text
12 functions, 94 statements
  native, called from Python              6 functions ( 50%)       45 statements ( 48%)
  native, called from native code         6 functions ( 50%)       49 statements ( 52%)
  Python                                  0 functions (  0%)        0 statements (  0%)

native, but Python calls the Python body (why its boundary is not used):
      4 functions  the boundary crossing costs more than the body saves
      2 functions  the objects it makes cost more to hand to Python than its loops save
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

`ppy explain trees.Node` (below) lists each field with the stores that
typed it and their lines.

One shape common in such code does not settle, and the
[Classes](../../docs/guide/classes.md#fields-without-annotations) guide
says why: a recursive `insert(node, key)` that stores its own result
(`node.left = insert(node.left, key)`) and is not annotated. A local that
starts as `None` and later holds a node (`prev = None` in a list reversal)
does settle: it takes the type of all its bindings, `Node | None`.

## What crosses

`workload` takes two ints and returns a tuple of three, so the call from
Python crosses three numbers. The tree of 50,000 nodes and the list are
made, relinked, and freed inside native code, and every method call is a
native call between native functions.

The methods are native too. When Python calls one, as the main block does
after the workload, the cost model decides whether the crossing pays:

- `small.reverse()` and `small.total()` loop over the list, so Python
  calls them natively. The first of them copies the list's five nodes into
  native memory; the second finds them resident and hands native code the
  same records (`PPY_RESIDENT_REPORT=1` prints `6 admitted, 1 calls`: the
  list and its five nodes, one resident call). `reverse` relinks the nodes
  natively, and the changed `right` links are set on the Python nodes when
  the call answers.
- `push` makes a `Node`. An object native code makes has to be made in
  Python too when the call answers, which costs more than `push` saves, so
  Python runs its Python body ("the objects it makes cost more to hand to
  Python than its loops save"). `insert` is the same.
- `rotate_up` and the three `__init__` methods have no loop, and the
  crossing costs more than their bodies.

`ppy explain --summary` counts the six that Python calls through their
Python bodies as native and called from native code, and gives each
reason.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python trees.ppy` | 0.67 |
| `ppy run trees.ppy`, after the first run built the cache | 0.34 |

`workload(50_000, 7)` alone, as the program prints it, the mean of five runs: 0.65 s under
CPython and 0.30 s under `ppy run`.

There is no standalone build: the program has no `main()`, and a
standalone binary needs a native `main` to start from.

## Where the code comes from

`trees.ppy` is hand-written.

Read on: [Classes](../../docs/guide/classes.md#fields-without-annotations)
and [Types from call sites](../../docs/guide/subset.md#types-from-call-sites).
