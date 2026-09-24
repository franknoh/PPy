# Classes

Native code holds an instance of a class in one of two ways, decided by the
class:

- A **value class** is copied, like a number. It is a dataclass or plain class
  whose fields are all numbers, and whose methods only read them. Native code
  keeps it as a struct of its fields.
- An **object class** is shared, like any Python object. It has fields that
  hold other objects or collections, or methods that change its fields.
  Native code keeps a handle to it on the heap, counts references to it, and
  frees it when the last one goes.

You don't choose between them. `ppy explain` shows which one a class is by
how its functions lower.

```python
from dataclasses import dataclass


@dataclass
class Tree:
    key: int
    left: "Tree | None" = None
    right: "Tree | None" = None


def insert(root: Tree | None, key: int) -> Tree:
    if root is None:
        return Tree(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    return root


def height(root: Tree | None) -> int:
    if root is None:
        return 0
    return 1 + max(height(root.left), height(root.right))
```

`insert` and `height` both lower to native code: `Tree` is an object class,
`Tree | None` is a handle that may be null, and `root is None` compares it
with null.

## Object classes

An object class has no base class and no `__getattr__` or similar hook. Each
field holds a value native code can represent:

- a number, a tuple of numbers, or a value class
- a collection: `Vec[int]`, `HashMap[int, Vec[int]]`
- another object, or `None` where the field is `Node | None`

Native code builds an instance by running the class's `__init__`, or for a
dataclass by setting each field from the arguments and the defaults. A
default natively has to be a constant, `None` included.

Reading a field of `None` is Python's `AttributeError`. Under `ppy run` it
falls back to Python, which raises it; a standalone binary stops.

Methods lower like functions, with `self` as a handle. `len(obj)` calls
`__len__`, and `if obj:` calls `__bool__` or `__len__` where the class has
one and otherwise tests that `obj` is not `None`.

## Generic classes

A class can declare type parameters, and its fields and methods use them:

```python
from ppy import Vec


class Stack[T]:
    def __init__(self) -> None:
        self.items: Vec[T] = Vec[T]()

    def push(self, value: T) -> None:
        self.items.push(value)

    def pop(self) -> T:
        return self.items.pop()

    def __len__(self) -> int:
        return len(self.items)
```

`Stack[int]()` and `Stack[float]()` are two classes to the checker, which
holds each method to its type argument: `push(2.5)` on a `Stack[int]` is
`E1301`. Native code instantiates the class and its methods once per type
argument, as a C++ template is instantiated. Write the type argument at
construction: `Stack[int]()`.

## Memory

A handle counts its references. A name holding an object keeps one, a field
or a collection holding it keeps one, and it is freed when the last goes,
together with what it holds. The tests run the emitted C of object programs
under AddressSanitizer with leak detection.

Reference counting does not free cycles. A doubly linked list, or a tree
whose nodes point back at their parents, is never freed natively: a
standalone binary lets the process exit free it, and under `ppy run` it stays
allocated. Singly linked structures, trees without parent links, and graphs
kept as adjacency lists are freed as usual.

## The Python boundary

An object never crosses between Python and native code. A native function
that takes or returns one is called by native code only. When Python code
calls it, its Python body runs. The functions Python calls natively take and
return numbers, and make their objects inside.

## Limitations

- A class with a base class, or a field native code cannot represent (a
  `str`, say), keeps the functions that use it in Python.
- A dataclass built natively takes constant defaults only.
- A generic class's type arguments are written at construction; `Stack()`
  without them stays in Python.
- Reading a field that holds a reference, off an object made in the same
  expression (`make().child`), is not lowered.
- Reference cycles are not freed, as described above.

Examples: [Collections](../howto/47_collections.md).
