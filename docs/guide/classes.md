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
- a `list`, `dict`, or `set` of what native code holds, such as
  `dict[str, int]` or `list[list[int]]`
- a `ppy` collection: `Vec[int]`, `HashMap[int, Vec[int]]`
- another object, or `None` where the field is `Node | None`
- a number or a string that may be `None` (`int | None`, `str | None`;
  see [Numbers and strings that may be `None`](native-lowering.md#numbers-and-strings-that-may-be-none))

Native code builds an instance by running the class's `__init__`, or for a
dataclass by setting each field from the arguments and the defaults. A
default is a constant (`None` included), `field(default=<constant>)`, or
`field(default_factory=...)` naming a collection (`Vec[int]`,
`HashMap[int, int]`) or a class built with no arguments, which is made anew
for each instance.

Reading a field of `None` is Python's `AttributeError`. Under `ppy run` it
falls back to Python, which raises it; a standalone binary stops.

Fields are assigned as Python assigns them, tuple assignment included:
`node.left, node.right = node.right, node.left` evaluates both values
before it stores either. A conditional expression chooses an object as it
chooses a number, evaluating only the side the test picks:
`node = node.left if key < node.key else node.right`.

## Fields without annotations

A field the class does not annotate has the type of everything the program
stores into it. That is the class's own `self.x = ...` in any method, and an
assignment to the field of an instance anywhere in the project:

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class Tree:
    def __init__(self):
        self.root = None

    def insert(self, key: int) -> None:
        if self.root is None:
            self.root = Node(key)
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = Node(key)
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(key)
                    return
                node = node.right
```

`self.left = None` in `__init__` and `node.left = Node(key)` in
`Tree.insert` make `Node.left` a `Node | None`, and `Tree.root` is one too.
Without strict mode, `Node.__init__`'s `key` is an `int` from the calls that
make nodes ([types from call sites](subset.md#types-from-call-sites)), and
`insert` lowers like the annotated `Tree` above.

- Values of several classes join. A subclass's instance where the base's is
  stored gives the base: `Shape | None` for a field set to `None`, a
  `Shape`, and a `Square`.
- An empty container a field starts as takes its element type from what is
  stored into it: `append`, `insert`, `add`, `heapq.heappush`, and
  `d[key] = value`, also one level in (`self.adj[u].append(v)` for
  `self.adj = [[] for _ in range(n)]`). `self.queue = []` and
  `self.queue.append(job)` make a `list` of what `job` is.
- A value the checker cannot type says nothing. A field stored only from
  unannotated parameters no call types stays unknown, and the functions that
  use it stay in Python.
- A field annotated anywhere (`self.count: int = 0` in `__init__`, or in the
  class body) keeps its annotation.
- A store that would add a field the class never sets itself is not
  counted, so such a field stays Python's.
- Evidence that goes round in a circle settles nothing. In
  `node.left = insert(node.left, key)` with an unannotated recursive
  `insert`, the field's type waits on `insert`'s result, and `insert`'s
  parameter on the field, so both stay unknown and the functions run on
  CPython. Annotating `insert` (`node: Node | None, key: int -> Node`)
  settles the field too. An `insert` method that walks down and stores
  `Node(key)`, as above, needs no annotation.
- A local that starts as `None` and later holds an object
  (`prev = None` ... `prev = node`) keeps its function in Python; declare
  it (`prev: Node | None = None`).

[`55_linked_structures`](../howto/55_linked_structures.md) is a search tree
with parent links and a linked list written this way, without annotations,
edited in place by native code.

The type is what the program shows, not a promise about every caller. Code
outside the project can store anything. When an object crosses into native
code, its fields are checked against these types, and a field holding
something else makes Python run the function's body instead.

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
| `a == b`, `a != b` | `__eq__`, `__ne__` (or `not __eq__`); a dataclass compares its fields; any other class compares identity |
| `a < b`, `a <= b`, `a > b`, `a >= b` | `__lt__`, ..., or the reflected method of the right operand; `@dataclass(order=True)` compares its fields as tuples |
| `max(a, b, ...)`, `min(a, b, ...)` of names of one class | `later > best` (`later < best` for `min`) for each later argument, as above; the first of equals wins |
| `str(a)`, `repr(a)`, `print(a)`, `f"{a}"`, `f"{a!r}"` | `__str__`, `__repr__`, or a dataclass's generated `__repr__`, `Point(x=1, y=2.5)` |
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

Under `ppy run`, Python can call a native function that takes or returns
objects of the project's classes. The object is copied into native memory
whole, together with the objects and containers its fields hold. An object
reached twice becomes one native object, so a shared object and a cycle
stay what they were.

After the call:

- if the function writes a field of any object that crossed, the new values
  are set on the caller's objects, which stay the same objects. A write
  through a local that holds a field (`node = self.head`, then
  `node.value = 0`), or through an object a call hands back
  (`self.last().value += 1`, `tail(head).next = Node(k)`), counts as a write
  through the parameter it was reached from;
- an object the function returns is the caller's own object when it came
  from one;
- an object native code made becomes a new instance of its class, with its
  fields set. `__init__` does not run again, because native code already
  ran it.

```python
import ppy


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None


@ppy.native
def bump(head: Node | None) -> None:
    while head is not None:
        head.value += 1
        head = head.next
```

Called from Python, `bump(a)` runs natively and leaves each node's `value`
one higher. `bump(None)` is native too, since the parameter allows `None`.

The copy costs time in proportion to what crosses, about 80 ns an object
in and as much back after a write, where CPython reads a field in a few
nanoseconds. So Python calls the native body only when the function does
work in proportion to it, and more than a few operations of it per object:
a loop that follows a field (`head = head.next`), a loop over a container
of objects, or a call to itself on a field (`height(node.left)`), each
doing six operations or more on what it reaches. Without `@ppy.native`,
`bump` above, which adds one to each node, would run its Python body when
Python calls it; the directive asks for the crossing whatever it costs.
Native callers pass objects by handle and copy nothing. `ppy explain` gives the reason for each function.

So a method Python calls on a large structure, such as `tree.insert(key)`
on a tree of a thousand nodes, usually runs its Python body: the whole tree
would cross for one descent. Where the structure is built and worked on by
a native function that Python calls with numbers and that returns numbers,
nothing but the numbers crosses, and every method it calls runs natively
([`55_linked_structures`](../howto/55_linked_structures.md)). Unlike
containers, objects are always copied, also by a call that only reads them.

A method of a class that crosses this way is bound like a method: `node.f(x)`
passes `node` to the native code, and a `@staticmethod` stays static.

Python runs the function's own body instead when an argument's class is not
one the signature describes. That covers a class made at run time, a
subclass from another module, and an object missing a field.

## Class attributes

An attribute the class body sets (`capacity: int = 10`, `LIMIT = 3`) may be
assigned through the class, as `Cache.capacity = n` or
`setattr(Cache, "capacity", n)`. The checker holds the value to the
attribute's type, widened from a literal. Such an attribute is state the
whole program shares: functions that read or write it run on CPython, so
every reader sees the latest value. An attribute the class body does not set
is still refused when assigned through the class (`E1506`).

A method can be called through its class with the receiver written out:
`Cache.add(b, 5)` is `b.add(5)`.

`typing.Self` names the class in its own signatures, fields, and method
bodies: `def link(self, other: Self) -> Self`. The checker reads it as the
class that writes it, so a subclass inherits a method that returns the base
class.

## Named tuples

`class P(NamedTuple)`, `P = NamedTuple("P", [("x", int), ("y", int)])`, and
`P = namedtuple("P", "x y")` declare a class whose instance is also a tuple:
`p.x`, `p[0]`, `x, y = p`, and `for v in p` read its fields, and
`_replace`, `_asdict`, and `_fields` are typed. A `namedtuple` field has no
type, so what is read from it is `Any`. Functions that take named tuples run
on CPython.

## Limitations

- A class with more than one base, a base from another module or a
  library, or a field native code cannot represent (a NumPy array, a
  `list` with no element type), keeps the functions that use it in Python. A
  Protocol the class only satisfies, without naming it as a base, is not a
  base.
- A call through a generic base whose subclass has type parameters the
  base's arguments do not decide stays in Python, as above.
- A dataclass's generated `==` and order compare fields that are numbers,
  bools, and strings; a field holding anything else keeps the comparison in
  Python. A float field that is NaN falls back, since whether it equals itself
  depends on CPython's version.
- A dataclass with subclasses keeps its generated methods in Python, since
  they read the instance's own class.
- `sorted`, and `min` and `max` of a collection, of `order=True` dataclass
  objects stay in Python; comparing two of them is native, and so is
  `max(a, b)` of them.
- A class with neither `__str__`, `__repr__`, nor a generated one prints its
  address, which only Python has. A value class with its own `__repr__` or
  `__str__` is shown by Python.
- A generic class whose type arguments nothing tells stays in Python.
- An exception object, an instance of a generic class, and an object with a
  field holding a function do not cross the Python boundary. A function that
  takes one is called natively by native code only.

Examples: [Inheritance](../howto/49_inheritance.md),
[Collections](../howto/47_collections.md).
