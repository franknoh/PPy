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

PROGRAMS = {
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
