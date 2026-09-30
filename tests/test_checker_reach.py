"""Valid Python the checker used to refuse.

Each program here is ordinary Python that CPython runs and a type checker
accepts: an old-style `TypeVar` with `Generic[T]`, `typing.Self`, a
`queue.Queue[int]`, `date - date`, `Decimal / 3`, `Counter & Counter`. The
checker accepts each in strict mode, still refuses the matching mistake, and
`ppy run` prints what `python` prints.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

OLD_STYLE_GENERICS = """
from typing import Generic, TypeVar

T = TypeVar("T")
N = TypeVar("N", bound=float)
S = TypeVar("S", int, str)


class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: list[T] = []

    def push(self, value: T) -> None:
        self.items.append(value)

    def pop(self) -> T:
        return self.items.pop()


def first(xs: list[T]) -> T:
    return xs[0]


def biggest(a: N, b: N) -> N:
    return a if a > b else b


def twice(x: S) -> S:
    return x + x


def main() -> None:
    s: Stack[int] = Stack()
    s.push(3)
    s.push(4)
    print(s.pop(), first([1, 2]), biggest(1.5, 2.5), twice("ab"), twice(3))


main()
"""

SELF = """
from typing import Self


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Self | None = None

    def link(self, other: Self) -> Self:
        self.next = other
        return self

    def copy(self) -> Self:
        return type(self)(self.value)


def main() -> None:
    a = Node(1).link(Node(2))
    nxt = a.next
    print(a.value, nxt.value if nxt is not None else -1, a.copy().value)


main()
"""

LIBRARY_VALUES = """
from collections import Counter, OrderedDict, deque
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from queue import Queue


def later(d: datetime) -> datetime:
    return d + timedelta(days=1)


def span(a: date, b: date) -> int:
    return (b - a).days


def money(x: Decimal) -> Decimal:
    return x / 4 + 1


def part(f: Fraction) -> Fraction:
    return f * 2 - Fraction(1, 3)


def common(a: Counter[int], b: Counter[int]) -> Counter[int]:
    return a & b


def drain(q: Queue[int]) -> int:
    total = 0
    while not q.empty():
        total += q.get()
    return total


def main() -> None:
    q: Queue[int] = Queue()
    q.put(1)
    q.put(2)
    d: deque[int] = deque([1, 2])
    d.appendleft(0)
    c: Counter[str] = Counter("abca")
    o: OrderedDict[str, int] = OrderedDict()
    o["a"] = 1
    o["b"] = 2
    o.move_to_end("a")
    print(later(datetime(2024, 1, 31)), span(date(2024, 1, 1), date(2024, 3, 1)))
    print(money(Decimal(10)), part(Fraction(1, 2)), drain(q))
    print(common(Counter([1, 1, 2]), Counter([1, 2, 2])))
    print(d.popleft() + d.pop(), c.most_common(1)[0][0].upper(), c.total(), list(o.keys()))


main()
"""

CLASS_STATE = """
class Cache:
    capacity: int = 10
    LIMIT = 3

    def __init__(self, n: int) -> None:
        if n:
            Cache.capacity = n
        self.items: list[int] = []

    def full(self) -> bool:
        return len(self.items) >= Cache.capacity

    def add(self, x: int) -> int:
        self.items.append(x)
        return len(self.items)

    def scaled(self, k: int) -> int:
        return self.capacity * k


def total(xs: list[int]) -> int:
    s = 0
    for x in xs:
        s += x * Cache.capacity
    return s


def main() -> None:
    a = Cache(0)
    print(a.full(), total([1, 2]), a.scaled(2))
    b = Cache(2)
    Cache.add(b, 5)
    print(Cache.add(b, 6), b.full(), a.full(), total([1, 2]), a.scaled(2))
    Cache.LIMIT += 1
    print(Cache.LIMIT, __import__("math").floor(2.5))


main()
"""

NAMED_TUPLES = """
from collections import namedtuple
from typing import NamedTuple

Particle = namedtuple("Particle", "x y z mass")
Pair = namedtuple("Pair", ["a", "b"], defaults=[0])
Point = NamedTuple("Point", [("x", int), ("y", int)])


def weight(ps: list[Particle]) -> float:
    return sum(p.mass for p in ps)


def dist(p: Point) -> int:
    x, y = p
    return abs(x) + abs(p[1])


def main() -> None:
    ps = [Particle(1, 2, 3, 4.5), Particle(0, 0, 0, 1)]
    print(weight(ps), Pair(1), Pair(1, 2).b, dist(Point(3, -4)), ps[0]._replace(mass=1))


main()
"""

NAMED_TUPLE_CLASS = """
from typing import NamedTuple


class P(NamedTuple):
    x: int
    y: float = 0.0


def f(p: P) -> float:
    a, b = p
    s = 0.0
    for v in p:
        s += v
    return p.x + p.y + p[0] + a + b + s


def main() -> None:
    p = P(1, 2.5)
    print(f(p), p, p._replace(x=3), len(p), p == P(1, 2.5), p._asdict(), P._fields)


main()
"""

SETATTR = """
from typing import Callable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Config:
    level: int = 1

    def __init__(self) -> None:
        self.name = "a"


def memo(func: Callable[[T], U]) -> Callable[[T], U]:
    seen: dict[T, U] = {}

    def wrapper(*args: T) -> U:
        if args[0] not in seen:
            seen[args[0]] = func(*args)
        return seen[args[0]]

    def size() -> int:
        return len(seen)

    setattr(wrapper, "size", size)
    return wrapper


def square(x: int) -> int:
    return x * x


def main() -> None:
    c = Config()
    fast = memo(square)
    setattr(c, "name", "b")
    setattr(Config, "level", 5)
    print(c.name, Config.level, c.level, fast(3), fast(3))


main()
"""

NARROWING = """
def f(x: int | float, y: list[int] | str, z: object, w: list[int]) -> str:
    out = []
    if isinstance(x, float):
        out.append(str(x / 2))
    if isinstance(y, (list, tuple)):
        out.append(str(sum(y)))
    if isinstance(z, int):
        out.append(str(z + 1))
    if not isinstance(w, (list, tuple)):
        raise ValueError
    out.append(str(len(w)))
    return ",".join(out)


print(f(3.0, [1, 2], 4, [1]), f(3, "a", "b", [1, 2]))
"""

GENERIC_STATIC = """
from __future__ import annotations


class HeapNode[T: float]:
    def __init__(self, value: T) -> None:
        self.value = value
        self.left: HeapNode[T] | None = None
        self.right: HeapNode[T] | None = None

    @staticmethod
    def merge(a: HeapNode[T] | None, b: HeapNode[T] | None) -> HeapNode[T] | None:
        if a is None:
            return b
        if b is None:
            return a
        if a.value > b.value:
            a, b = b, a
        a.left, a.right = a.right, a.left
        a.left = HeapNode.merge(a.left, b)
        return a


class Heap[T: float]:
    def __init__(self) -> None:
        self.root: HeapNode[T] | None = None

    def push(self, value: T) -> None:
        self.root = HeapNode.merge(self.root, HeapNode(value))

    def pop(self) -> T:
        root = self.root
        if root is None:
            raise IndexError("empty")
        self.root = HeapNode.merge(root.left, root.right)
        return root.value


def main() -> None:
    h: Heap[int] = Heap()
    for x in [5, 3, 8, 1, 9, 2]:
        h.push(x)
    print([h.pop() for _ in range(6)])


main()
"""

GENERATORS = """
from collections.abc import Generator, Iterator, Iterable


def squares(n: int) -> Generator[int, None, None]:
    return (i * i for i in range(n))


def evens(xs: list[int]) -> Generator[int]:
    return (x for x in xs if x % 2 == 0)


def take(g: Generator[int, None, None]) -> int:
    return sum(g)


def main() -> None:
    g: Generator[int, None, None] = (i for i in range(3))
    print(list(squares(4)), list(evens([1, 2, 4])), take(i for i in range(5)), next(g))


main()
"""

PROGRAMS = {
    "generators": GENERATORS,
    "generic_static": GENERIC_STATIC,
    "setattr": SETATTR,
    "narrowing": NARROWING,
    "named_tuples": NAMED_TUPLES,
    "named_tuple_class": NAMED_TUPLE_CLASS,
    "class_state": CLASS_STATE,
    "old_style_generics": OLD_STYLE_GENERICS,
    "self": SELF,
    "library_values": LIBRARY_VALUES,
}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1] / "src"), env.get("PYTHONPATH", "")]
    )
    return env


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_the_checker_accepts_it_in_strict_mode(write, codes, name: str):
    path = write("prog.py", PROGRAMS[name])
    assert [c for c in codes(path) if c.startswith("E")] == []


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_ppy_run_prints_what_python_prints(project_dir: Path, write, name: str):
    path = write("prog.py", PROGRAMS[name])
    expected = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, check=True
    ).stdout
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", "prog.py"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout == expected


@pytest.mark.parametrize(
    "source",
    [
        # A bound is still a bound.
        """
        from typing import TypeVar

        N = TypeVar("N", bound=float)


        def biggest(a: N, b: N) -> N:
            return a if a > b else b


        def wrong() -> None:
            biggest("a", "b")
        """,
        # So is a `Generic[T]` class's argument.
        """
        from typing import Generic, TypeVar

        T = TypeVar("T")


        class Box(Generic[T]):
            def __init__(self, value: T) -> None:
                self.value = value

            def put(self, value: T) -> None:
                self.value = value


        def wrong(b: Box[int]) -> None:
            b.put("no")
        """,
        # And a queue's element type.
        """
        from queue import Queue


        def wrong(q: Queue[int]) -> str:
            return q.get()
        """,
        # A module name `__import__` cannot know.
        """
        def load(name: str) -> None:
            __import__(name)
        """,
        # An attribute the class body never set is still a monkey-patch.
        """
        class Box:
            size: int = 1


        def patch() -> None:
            Box.colour = "red"
        """,
        # And one it did set keeps its type.
        """
        class Box:
            size: int = 1


        def patch() -> None:
            Box.size = "big"
        """,
        # `setattr` with a constant name is the assignment it spells.
        """
        class C:
            def __init__(self) -> None:
                self.n = 1


        def f(c: C) -> None:
            setattr(c, "n", "x")
        """,
        # `Self` outside a class says what it is.
        """
        from typing import Self


        def wrong(x: Self) -> None:
            pass
        """,
    ],
)
def test_the_matching_mistake_is_still_refused(write, codes, source: str):
    path = write("wrong.py", textwrap.dedent(source))
    assert any(c.startswith("E") for c in codes(path))


MAYBE_NONE = """
class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None


def value_of(node: Node) -> int:
    return node.value


def second(node: Node) -> int:
    return node.next.value


def total(head: Node, n: int) -> int:
    s = 0
    for _ in range(n):
        s += value_of(head.next) + second(head)
    return s


def main() -> None:
    a = Node(1)
    a.next = Node(2)
    print(total(a, 1000))
    for f in (lambda: value_of(a.next.next), lambda: second(a.next), lambda: total(a.next, 3)):
        try:
            print(f())
        except AttributeError as e:
            print("AttributeError", e)


main()
"""


def test_a_value_that_may_be_none_is_a_warning_without_strict(write, codes):
    """`W2011` under `--no-strict`: `node.next.value` and `value_of(head.next)`
    run, and raise `AttributeError` where CPython does. Strict mode keeps
    `E1206` and `E1301`."""
    path = write("prog.py", MAYBE_NONE)
    loose = codes(path, strict=False)
    assert "W2011" in loose
    assert [c for c in loose if c.startswith("E")] == []
    strict = codes(path)
    assert "E1206" in strict and "E1301" in strict


def test_native_code_raises_where_cpython_does_on_none(project_dir: Path, write):
    (project_dir / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    path = write("prog.py", MAYBE_NONE)
    expected = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, check=True
    ).stdout
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", "prog.py"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout == expected
