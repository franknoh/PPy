# Classes

Native code holds an instance of a class in one of two ways, decided by the
class:

- A **value class** is copied, like a number. It is a dataclass or plain class
  whose fields are all numbers, and whose methods only read them. Native code
  keeps it as a struct of its fields.
- An **object class** is shared, like any Python object. It has fields that
  hold other objects or collections, methods that change its fields or
  return a new instance, its own `__lt__`, `__hash__`, or `__eq__` (which
  a collection calls with the object), a base class, or subclasses. Native
  code keeps a handle to it on the heap, counts references to it, and frees
  it when the last one goes.

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

An object class has at most one base, itself an object class of the same
module, and no `__getattr__` or similar hook. Each field holds a value
native code can represent:

- a number, a string, a tuple of numbers, or a value class
- a collection: `Vec[int]`, `HashMap[int, Vec[int]]`
- another object, or `None` where the field is `Node | None`

Native code builds an instance by running the class's `__init__`, or for a
dataclass by setting each field from the arguments and the defaults. A
default is a constant (`None` included), `field(default=<constant>)`, or
`field(default_factory=...)` naming a collection (`Vec[int]`,
`HashMap[int, int]`) or a class built with no arguments, which is made anew
for each instance.

Reading a field of `None` is Python's `AttributeError`. Under `ppy run` it
falls back to Python, which raises it; a standalone binary stops.

Methods lower like functions, with `self` as a handle. `len(obj)` calls
`__len__`, and `if obj:` calls `__bool__` or `__len__` where the class has
one and otherwise tests that `obj` is not `None`.

## Inheritance

A class may derive from one object class of the same module. Its instance is
one record: the root class's fields first, then each subclass's own, so a
`Square` is a `Shape` with more words after it.

```python
from ppy import Vec


class Shape:
    def __init__(self, scale: int) -> None:
        self.scale: int = scale

    def area(self) -> int:
        return 0

    def describe(self) -> int:
        return self.area() * 10 + self.scale


class Square(Shape):
    def __init__(self, scale: int, side: int) -> None:
        super().__init__(scale)
        self.side: int = side

    def area(self) -> int:
        return self.side * self.side


def total(shapes: Vec[Shape]) -> int:
    acc: int = 0
    for s in shapes:
        acc += s.describe()
        if isinstance(s, Square):
            acc += s.side
    return acc
```

- A method a subclass does not define is its base's.
- A call to a method some subclass overrides goes by the class the object was
  made as. Each object carries a tag for its class in its header, and the
  call compares the tag and calls that class's method. The overriding method
  takes and returns the same types as the one it overrides; one that does
  not keeps the caller in Python.
- `super().__init__(...)` and `super().method(...)` call the next class's
  method along the bases, directly.
- A parameter, a field, or an element typed `Shape` takes a `Square`.
- `isinstance(obj, Cls)`, or with a tuple of classes, reads the tag. The
  checker narrows on it as usual, so `s.side` after `isinstance(s, Square)`
  reads the `Square`'s field.

A class that others derive from is an object class even when its fields are
all numbers, since a copy of its fields would cut a subclass's short.

## Operators

An object class's operator methods lower as calls:

| written | calls |
|---|---|
| `a + b`, `a - b`, `a * k`, and the other arithmetic operators | `__add__`, `__sub__`, `__mul__`, ..., or the right operand's `__radd__`, ... |
| `-a`, `+a`, `~a` | `__neg__`, `__pos__`, `__invert__` |
| `a == b`, `a != b` | `__eq__`, `__ne__` (or `not __eq__`); a class with neither compares identity |
| `a < b`, `a <= b`, `a > b`, `a >= b` | `__lt__`, ..., or the reflected method of the right operand |
| `obj[key]`, `obj[key] = value` | `__getitem__`, `__setitem__` |
| `x in obj` | `__contains__` |
| `len(obj)`, `if obj:` | `__len__`, `__bool__` |

A method that returns a new instance, as `__add__` usually does, makes the
class an object class: each result is a new object, freed when its last
reference goes.

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
argument, as a C++ template is instantiated.

The type arguments may be left out where the checker can tell them:

- from where the instance goes: `s: Stack[int] = Stack()`, a return from a
  function declared `-> Stack[int]`, or an assignment to a field declared
  `Stack[int]`
- from the constructor's arguments: `Pair(1, 2.5)` is a `Pair[int, float]`
  when `__init__` (or the dataclass fields) takes an `A` and a `B`

The collections work the same way: `v: Vec[int] = Vec()` and
`m: HashMap[int, int] = HashMap()`. A collection that stores floats is the
exception, since CPython runs `Vec()` without knowing its type and would
keep a stored `3` an `int`: write `Vec[float]()`.

### Generic bases

A generic class may derive from another, and a class may derive from a
generic one with its arguments given:

```python
class Counted[T](Stack[T]):
    def __init__(self) -> None:
        super().__init__()
        self.pushes: int = 0

    def push(self, value: T) -> None:
        self.pushes += 1
        super().push(value)


class IntStack(Stack[int]):
    def total(self) -> int:
        s = 0
        for v in self.items:
            s += v
        return s
```

The checker follows what each class gives its base: in a `Counted[int]`,
`Stack`'s `T` is `int` as well, so `pop()` returns an `int`; in an
`IntStack` it is `int` with nothing written at the use. `super()` is the
base with those arguments. A `Counted[int]` or an `IntStack` goes where a
`Stack[int]` is expected, and one given to a `Stack[int]` parameter runs its
own `push`.

Native code lays the object out and instantiates each inherited method by
the same bindings. A call through a `Stack[int]` dispatches to a
subclass's override by the object's class, with the subclass's own
arguments worked out from the base's. Where they cannot be, as for a
`class Tagged[T, U](Stack[T])` whose `U` the `Stack[int]` says nothing of,
the function that makes such a call stays in Python.

## Memory

A handle counts its references. A name holding an object keeps one, a field
or a collection holding it keeps one, and it is freed when the last goes,
together with what it holds. The tests run the emitted C of object programs
under AddressSanitizer with leak detection.

Reference counting alone does not free a cycle: a doubly linked list, or a
tree whose nodes point back at their parents. A collector does. Every
object and collection is on a list of what its thread made. When the
objects made since the last collection that can hold others number 700 more
than those that survived it, the collector counts the references each one
gets from the others on the list. What still has a reference from
outside (a name, a native frame) is kept, with everything it reaches. What
is left is held only by itself, and is freed. This is the method CPython's
`gc` uses.

`gc.collect()` in native code runs the collector at once. CPython's count
is of its own objects, so native code does not return it: the call lowers as
a statement, and `n = gc.collect()` keeps the function in Python. A
standalone binary also collects once before it exits, which is why its
leak check passes.

## The Python boundary

An object never crosses between Python and native code. A native function
that takes or returns one is called by native code only. When Python code
calls it, its Python body runs. The functions Python calls natively take and
return numbers, and make their objects inside.

## Limitations

- A class with more than one base, a base from another module or a
  library, or a field native code cannot represent (a `dict`, say), keeps
  the functions that use it in Python.
- A call through a generic base whose subclass has type parameters the
  base's arguments do not decide stays in Python, as above.
- A dataclass object compared with `==` keeps the function in Python unless
  the class defines `__eq__`, since the generated one compares fields.
- A generic class whose type arguments nothing tells stays in Python.

Examples: [Inheritance](../howto/49_inheritance.md),
[Collections](../howto/47_collections.md).
