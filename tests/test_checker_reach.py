"""Valid Python the checker used to refuse.

Each program here is ordinary Python that CPython runs and a type checker
accepts: an old-style `TypeVar` with `Generic[T]`, `typing.Self`, a
`queue.Queue[int]`, `date - date`, `Decimal / 3`, `Counter & Counter`. The
checker accepts each in strict mode, still refuses the matching mistake, and
`ppy run` prints what `python` prints.
"""

from __future__ import annotations

import json
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

MUTABLE_SEQUENCE = """
from collections.abc import MutableSequence


def bump(xs: MutableSequence[int]) -> MutableSequence[int]:
    for i in range(len(xs)):
        xs[i] += 1
    xs.append(0)
    xs.insert(0, 7)
    xs.reverse()
    return xs


def total(xs: MutableSequence[int]) -> int:
    s = 0
    for x in xs:
        s += x
    return s + xs.count(0) + xs[0] + xs.pop()


def main() -> None:
    a = [1, 2, 3]
    print(bump(a), total(a), a)


main()
"""

BARE_ITERATORS = """
from collections.abc import Generator, Iterable, Iterator


def main() -> None:
    g: Generator = (i * i for i in range(5))
    it: Iterator = (i for i in range(3))
    ib: Iterable = (i for i in range(3))
    print(sum(g), list(it), list(ib))


main()
"""

EMPTY_RETURNS = """
def firsts(limit: int) -> list[int]:
    if limit <= 0:
        return []
    out = []
    for i in range(limit):
        out.append(i * i)
    return out


def table(n: int) -> dict[int, int]:
    if n == 0:
        return {}
    d: dict[int, int] = {}
    for i in range(n):
        d[i] = i + 1
    return d


def main() -> None:
    print(firsts(0), firsts(4), table(0), table(3))


main()
"""

PROGRAMS = {
    "bare_iterators": BARE_ITERATORS,
    "mutable_sequence": MUTABLE_SEQUENCE,
    "empty_returns": EMPTY_RETURNS,
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
        # A tuple is not a `MutableSequence`.
        """
        from collections.abc import MutableSequence


        def bump(xs: MutableSequence[int]) -> None:
            xs.append(1)


        def wrong() -> None:
            bump((1, 2))
        """,
        # Nor is a `list[str]` a `MutableSequence[int]`.
        """
        from collections.abc import MutableSequence


        def wrong(xs: list[str]) -> MutableSequence[int]:
            return xs
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


#: Programs CPython runs -- to the end, or to an error of its own -- that the
#: checker cannot fully read. Strict mode refuses each; `--no-strict` warns
#: (`W2010`), keeps what it cannot read on the Python path, and runs it.
CPYTHON_RUNS = {
    # A sibling module that is not there: CPython raises at the import.
    "relative_import": """
from .stack import Stack


def evaluate(text: str) -> int:
    operands: Stack[int] = Stack()
    for ch in text:
        operands.push(int(ch))
    return operands.pop()


print(evaluate("12"))
""",
    "missing_package": """
from data_structures.heap.heap import Heap


def top(xs: list[int]) -> Heap:
    heap = Heap()
    heap.build_max_heap(xs)
    return heap


print(top([3, 1, 2]))
""",
    # Annotations CPython never evaluates, naming what nothing defines.
    "unreadable_annotations": """
def scale(x: "Missing", k: int) -> int:
    held: Unknowable[int] = x
    return held * k


print(scale(4, 3))
""",
    # `globals()`, `eval` and a computed `getattr`: CPython runs them all.
    "dynamic_features": """
from timeit import timeit

LIMIT = 3


class Box:
    size = 5


def lookup(name: str) -> int:
    return globals()[name]


def attribute(box: Box, name: str) -> int:
    return getattr(box, name)


def total(n: int) -> int:
    s = 0
    for i in range(n):
        s += i
    return s


print(lookup("LIMIT"), eval("1 + 2"), attribute(Box(), "size"), total(10))
print(timeit("total(10)", globals=globals(), number=2) >= 0)
""",
    # A star import may rebind any name -- here `pow`, to `math.pow`, which
    # returns a float -- so nothing in the module is lowered.
    "star_import": """
from math import *


def squares(n: int) -> int:
    s = 0
    for i in range(n):
        s += pow(i, 2)
    return s


print(squares(4), floor(3.7))
""",
    # A base computed at runtime may give the class its operators.
    "computed_base": """
def pick(flag: bool) -> type:
    return int if flag else float


class Num(pick(True)):
    pass


def bump(n: Num) -> int:
    return n + 1


print(bump(Num(5)))
""",
    # A name nothing defines: CPython raises `NameError` when it gets there.
    "undefined_name": """
def area(r: float) -> float:
    return PI * r * r


print("before")
print(area(2.0))
""",
}


def _last_raised(err: str) -> str:
    lines = [line for line in err.strip().splitlines() if line and not line[0].isspace()]
    raised = [line for line in lines if line.split(":")[0].endswith(("Error", "Exception"))]
    return raised[-1] if raised else ""


@pytest.mark.parametrize("name", sorted(CPYTHON_RUNS))
def test_strict_mode_refuses_what_it_cannot_read(write, codes, name: str):
    path = write("prog.py", CPYTHON_RUNS[name])
    assert any(c.startswith("E") for c in codes(path))
    loose = codes(path, strict=False)
    assert [c for c in loose if c.startswith("E")] == [], loose
    assert "W2010" in loose


@pytest.mark.parametrize("name", sorted(CPYTHON_RUNS))
def test_no_strict_runs_what_cpython_runs(project_dir: Path, write, name: str):
    """Same stdout, same exit code, and the same exception last, as CPython."""
    (project_dir / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    path = write("prog.py", CPYTHON_RUNS[name])
    expected = subprocess.run(
        [sys.executable, path.name],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", "prog.py"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=600,
        check=False,
    )
    assert "error[" not in done.stderr, done.stderr
    assert done.returncode == expected.returncode, done.stdout + done.stderr
    assert done.stdout == expected.stdout
    assert _last_raised(done.stderr) == _last_raised(expected.stderr)


PRIVATE_HELPERS = """
def _scale(xs, k):
    return [x * k for x in xs]


def _count(words):
    seen = {}
    for w in words:
        if w in seen:
            seen[w] += 1
        else:
            seen[w] = 1
    return seen


def evens(n: int) -> list[int]:
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(i)
    return out


def pairs(n: int) -> int:
    acc = []
    i = 0
    while i < n:
        acc.append((i, i * i))
        i += 1
    return sum(b for _, b in acc)


def main() -> None:
    print(_scale([1, 2, 3], 2), _count(["a", "b", "a"]), evens(7), pairs(5))


main()
"""


def test_private_helpers_take_their_parameter_types_from_their_calls(write, codes, analyze):
    """Without strict mode, `_scale(xs, k)` called only as `_scale([1, 2, 3], 2)`
    has `xs: list[int]` and `k: int`; strict mode still asks for annotations."""
    path = write("prog.py", PRIVATE_HELPERS)
    bundle = analyze(path, strict=False)
    params = {p.name: str(p.type) for p in bundle.symbols.functions["prog._scale"].params}
    assert params == {"xs": "list[int]", "k": "int"}
    assert [d.code for d in bundle.diagnostics.sorted() if d.code in {"W2010", "E1201"}] == []
    assert "E1201" in codes(path)


def test_a_helper_used_as_a_value_is_not_inferred(write, analyze):
    path = write(
        "prog.py",
        """
        def _double(x):
            return x * 2


        def main() -> None:
            print(_double(3), list(map(_double, [1, 2])))
        """,
    )
    bundle = analyze(path, strict=False)
    assert not bundle.symbols.functions["prog._double"].params[0].inferred


def test_an_empty_display_returned_in_place_goes_native(project_dir: Path, write):
    """`return []` from a `-> list[int]` is a `list[int]`: the function is
    native, not held back for an element type nobody told."""
    write("prog.py", EMPTY_RETURNS)
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "explain", "--summary", "--json", "prog.py"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=600,
        check=True,
    )
    tiers = {f["qualname"]: f["tier"] for f in json.loads(done.stdout)["functions"]}
    assert tiers["prog.firsts"] != "python"
    assert tiers["prog.table"] != "python"
