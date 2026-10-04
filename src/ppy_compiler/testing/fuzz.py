"""A differential fuzzer over the natively lowered subset, on every path.

`generate_program(seed)` writes a small, well-typed, strict PPy program: a
fixed prelude of classes, then functions over ints (past 64 bits too),
floats (NaN, the infinities, -0.0, `int()` of them), bools, strings and
f-strings, tuples, the `ppy` collections, a value class, an object class,
and a generic function, and a `main()` that calls each with constant
arguments and prints what it returns. The same seed gives the same program
on every machine.

`run_program` runs it under CPython (the reference), the Python backend,
`ppy run`, a standalone binary, and emitted C and C++ built with
AddressSanitizer and UBSan, one path at a time, each under a timeout and a
memory cap. `compare` holds every path to the reference: its output, its
exit status, and the last line of what it wrote to stderr. The one
difference allowed is documented: where CPython computes an integer past 64
bits, a standalone binary and emitted C stop with
`OverflowError: the result does not fit in a 64-bit integer`, having
printed what CPython printed before it.

`minimize` deletes statements from a failing program while the failure
persists; `scripts/fuzz.py` drives all of it and saves what it finds under
`tests/fuzz_regressions/`.
"""

from __future__ import annotations

import ast
import contextlib
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ALL_PATHS",
    "OVERFLOW_64",
    "STRUCTURES_MARK",
    "TIMED_OUT",
    "UNANNOTATED_MARK",
    "Mismatch",
    "Result",
    "compare",
    "generate_program",
    "minimize",
    "printed_twice",
    "run_program",
]

#: Every path a program can take, the reference first.
ALL_PATHS = ("python", "ppy", "run", "standalone", "c", "cpp")
#: The paths a program with module state runs on (`generate_program(state=True)`).
STATE_PATHS = ("python", "ppy", "run")

#: What native code says where CPython would compute an integer no word holds.
OVERFLOW_64 = "OverflowError: the result does not fit in a 64-bit integer"

#: The status a path is given when it ran past its timeout: a finding like
#: any other, since CPython answered in that time.
TIMED_OUT = 124

#: The paths that are native code with no Python to fall back to.
_NATIVE_ONLY = frozenset({"standalone", "c", "cpp"})

_PRELUDE = """\
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ppy import Deque, HashMap, Heap, TreeMap, Vec


@dataclass
class Point:
    x: int
    y: int


class Counter:
    def __init__(self, start: int) -> None:
        self.count: int = start

    def bump(self, by: int) -> int:
        self.count += by
        return self.count


def biggest[T: int | float](values: Vec[T]) -> T:
    best = values[0]
    for value in values:
        if value > best:
            best = value
    return best


class Broken(ValueError):
    pass


class Failed(ValueError):
    def __init__(self, code: int, where: str) -> None:
        super().__init__(f"{where}: {code}")
        self.code: int = code
        self.where: str = where


@dataclass(order=True)
class Tag:
    rank: int
    label: str


def countdown(n: int) -> Iterator[int]:
    while n > 0:
        yield n
        n -= 2


def pairs(n: int) -> Iterator[int]:
    for i in range(n):
        if i % 3 == 0:
            continue
        yield i * i
    yield from countdown(n)


def drain(it: Iterator[int], most: int) -> int:
    total = 0
    for value in it:
        if most <= 0:
            break
        total = total * 5 + value
        most -= 1
    return total


class Ring:
    def __init__(self, size: int) -> None:
        self.size: int = size
        self.names: list[str] = []

    def add(self, name: str) -> None:
        self.names.append(name)

    def __iter__(self) -> Iterator[str]:
        seen: list[str] = []
        for name in self.names:
            if name not in seen:
                seen.append(name)
                yield name + str(len(seen))


"""

#: What a program that calls the standard library adds: a cached function,
#: whose recursion goes through its cache.
_STDLIB_PRELUDE = """\
@functools.lru_cache(maxsize=None)
def _cached(n: int) -> int:
    if n < 2:
        return n
    return (_cached(n - 1) + _cached(n - 2)) % 1000003


"""

#: What a program with module state adds: objects Python makes and native
#: code walks. `ppy run` passes both across its boundary; a standalone build
#: has no Python to hold them, so those programs run on the paths that do.
_STATE_PRELUDE = """\
class Link:
    def __init__(self, value: int, label: str) -> None:
        self.value: int = value
        self.label: str = label
        self.next: Link | None = None


def chain(n: int) -> Link:
    head = Link(n, "0")
    for i in range(1, n):
        made = Link(n - i, str(i))
        made.next = head
        head = made
    return head


def values(head: Link | None) -> list[int]:
    out: list[int] = []
    while head is not None:
        out.append(head.value)
        head = head.next
    return out


"""

#: What a program with boundary writes adds: an object class whose instances
#: cross by copy, and the directive that asks for the crossing.
_BOUNDARY_PRELUDE = """\
import ppy


class Box:
    def __init__(self, value: int) -> None:
        self.value: int = value
        self.items: list[int] = []
        self.peer: Box | None = None


"""

#: What a program with structure edits adds: classes whose fields have no
#: annotations, typed from what the program stores into them (`None` at
#: first, then objects), and helpers that read a structure back.
_STRUCTURES_PRELUDE = """\
import ppy


class SNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class DNode:
    def __init__(self, key):
        self.key = key
        self.prev = None
        self.next = None


def sshape(n):
    if n is None:
        return "."
    return "(" + sshape(n.left) + str(n.key) + sshape(n.right) + ")"


def slinked(t):
    ok = t.root is None or t.root.parent is None
    pending = [t.root]
    while pending:
        n = pending.pop()
        if n is None:
            continue
        for c in (n.left, n.right):
            if c is not None:
                ok = ok and c.parent is n
                pending.append(c)
    return ok


def dkeys(d):
    ahead = []
    n = d.head
    while n is not None and len(ahead) < 50:
        ahead.append(n.key)
        n = n.next
    back = []
    n = d.tail
    while n is not None and len(back) < 50:
        back.append(n.key)
        n = n.prev
    return ahead, back == ahead[::-1]


"""

_BIG = (2**62, -(2**62), 2**63 - 1, -(2**63), 3037000499, 4611686018427387903)
_FLOATS = ("0.5", "-0.0", "1e308", "-2.5", "3.0", "1e-300", "7.25")
_SPECIAL_FLOATS = ('float("inf")', 'float("-inf")', 'float("nan")')
_WORDS = ("", "a", "ab", "Hello", " spaced out ", "x-y-z", "MiXeD", "123", "two  gaps")


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 0

    def put(self, text: str) -> None:
        self.lines.append("    " * self.depth + text)


@dataclass
class _Scope:
    """The names a block may read, by type."""

    ints: list[str] = field(default_factory=list)
    floats: list[str] = field(default_factory=list)
    strs: list[str] = field(default_factory=list)
    bools: list[str] = field(default_factory=list)
    vecs: list[str] = field(default_factory=list)

    def copy(self) -> _Scope:
        return _Scope(
            list(self.ints), list(self.floats), list(self.strs), list(self.bools), list(self.vecs)
        )


class _Generator:
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        seed: int,
        prints: bool = False,
        state: bool = False,
        stdlib: bool = False,
        calls: bool = False,
        unannotated: bool = False,
        boundary: bool = False,
        structures: bool = False,
    ) -> None:
        self.rng = random.Random(seed)
        #: Whether functions take defaults and keyword-only parameters, and
        #: `main` calls them by keyword and leaves defaults out, drawn from a
        #: sequence of their own so the rest of the program stays the same.
        self.calls = calls
        self.call_rng = random.Random(seed ^ 0xCA11)
        #: Per function: its parameters, their kinds, their defaults, and
        #: where the keyword-only ones start.
        self.signatures: dict[str, tuple[list[str], list[str], dict[str, str], int]] = {}
        #: Functions without annotations, whose types `--no-strict` infers
        #: from the calls `main` makes; Python then calls them with others.
        self.unannotated = unannotated
        self.foreign = random.Random(seed ^ 0xA77)
        self.fresh = 0
        self.seed = seed
        #: Whether functions print too, between checks that may fall back.
        self.prints = prints
        #: Also draw seeded random numbers and call `math`, `heapq`, and `bisect`.
        self.stdlib = stdlib
        #: Whether the program also has module state and objects crossing
        #: `ppy run`'s boundary, drawn from a sequence of their own.
        self.with_state = state
        self.state = random.Random(seed ^ 0x5EED)
        #: Whether the program also writes through containers and objects
        #: Python passes, shared and nested, drawn from a sequence of their own.
        self.with_boundary = boundary
        self.crossing = random.Random(seed ^ 0xB0DE)
        #: Whether the program also edits linked structures in place through
        #: methods Python calls natively, on classes with unannotated fields.
        self.with_structures = structures
        self.structure = random.Random(seed ^ 0x57C7)

    def name(self, prefix: str) -> str:
        self.fresh += 1
        return f"{prefix}{self.fresh}"

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    # -- expressions ---------------------------------------------------------

    def int_literal(self) -> str:
        if self.chance(0.12):
            return str(self.rng.choice(_BIG))
        return str(self.rng.randint(-12, 12))

    def int_expr(self, scope: _Scope, depth: int = 0) -> str:
        rng = self.rng
        if self.stdlib and depth < 2 and self.chance(0.15):
            return self.stdlib_int(scope, depth + 1)
        if depth > 2 or self.chance(0.3):
            if scope.ints and self.chance(0.65):
                return rng.choice(scope.ints)
            return self.int_literal()
        roll = rng.random()
        left = self.int_expr(scope, depth + 1)
        if roll < 0.45:
            op = rng.choice(("+", "-", "*"))
            return f"({left} {op} {self.int_expr(scope, depth + 1)})"
        if roll < 0.6:
            op = rng.choice(("//", "%"))
            divisor = self.int_expr(scope, depth + 1)
            # Mostly nonzero; a zero divisor now and then is CPython's error.
            if self.chance(0.9):
                fallback = rng.choice((-7, -3, 3, 5))
                divisor = f"({divisor} if {divisor} != 0 else {fallback})"
            return f"({left} {op} {divisor})"
        if roll < 0.68:
            op = rng.choice(("<<", ">>"))
            return f"({left} {op} ({self.int_expr(scope, depth + 1)} % {rng.randint(1, 70)}))"
        if roll < 0.74:
            op = rng.choice(("&", "|", "^"))
            return f"({left} {op} {self.int_expr(scope, depth + 1)})"
        if roll < 0.8:
            return f"{rng.choice(('abs', '-'))}({left})"
        if roll < 0.86:
            return f"{rng.choice(('min', 'max'))}({left}, {self.int_expr(scope, depth + 1)})"
        if roll < 0.91:
            return f"len({self.str_expr(scope, depth + 1)})"
        if roll < 0.93:
            return f"int({self.float_expr(scope, depth + 1)})"
        if roll < 0.95:
            # A small exponent, and one past a word now and then.
            exponent = rng.choice(("0", "1", "2", "3", "7", "40", "63", "64"))
            if self.chance(0.5):
                modulus = rng.choice(("7", "1000000007", "-5", "1"))
                return f"pow({left}, {exponent}, {modulus})"
            return f"({left} ** {exponent})"
        return f"({left} if {self.bool_expr(scope, depth + 1)} else {self.int_expr(scope, 3)})"

    def float_expr(self, scope: _Scope, depth: int = 0) -> str:
        rng = self.rng
        if self.stdlib and depth < 2 and self.chance(0.15):
            return self.stdlib_float(scope, depth + 1)
        if depth > 2 or self.chance(0.3):
            if scope.floats and self.chance(0.6):
                return rng.choice(scope.floats)
            if self.chance(0.03):
                return rng.choice(_SPECIAL_FLOATS)
            return rng.choice(_FLOATS)
        roll = rng.random()
        left = self.float_expr(scope, depth + 1)
        if roll < 0.5:
            op = rng.choice(("+", "-", "*"))
            return f"({left} {op} {self.float_expr(scope, depth + 1)})"
        if roll < 0.62:
            divisor = self.float_expr(scope, depth + 1)
            if self.chance(0.9):
                divisor = f"({divisor} if {divisor} != 0.0 else 2.0)"
            return f"({left} / {divisor})"
        if roll < 0.74:
            return f"({left} {rng.choice(('+', '*'))} {self.int_expr(scope, depth + 1)})"
        if roll < 0.82:
            return f"math.sqrt(abs({left}))"
        if roll < 0.9:
            return f"float({self.int_expr(scope, depth + 1)})"
        return f"{rng.choice(('min', 'max'))}({left}, {self.float_expr(scope, depth + 1)})"

    def small(self, scope: _Scope, depth: int, top: int) -> str:
        """A nonnegative int below `top`, from an expression."""
        return f"(abs({self.int_expr(scope, depth + 1)}) % {top})"

    # A list display holds one item at least: `[]` says nothing of what it
    # holds, which a native build needs told.
    def int_list(self, scope: _Scope, depth: int) -> str:
        items = ", ".join(self.int_expr(scope, 3) for _ in range(self.rng.randint(1, 5)))
        return f"[{items}]"

    def float_list(self, scope: _Scope, depth: int) -> str:
        items = ", ".join(self.float_expr(scope, 3) for _ in range(self.rng.randint(1, 5)))
        return f"[{items}]"

    def stdlib_int(self, scope: _Scope, depth: int) -> str:
        rng = self.rng
        roll = rng.random()
        if roll < 0.15:
            low = self.int_expr(scope, depth + 1)
            return f"random.randint({low}, {low} + {self.small(scope, depth, 50)})"
        if roll < 0.25:
            return f"random.randrange({self.int_expr(scope, depth + 1)}, {rng.randint(-5, 60)})"
        if roll < 0.32:
            return f"random.getrandbits({self.small(scope, depth, 64)})"
        if roll < 0.4:
            return f"random.choice({self.int_list(scope, depth)})"
        if roll < 0.48:
            return f"sum(random.sample({self.int_list(scope, depth)}, {rng.randint(0, 3)}))"
        if roll < 0.56:
            return f"math.gcd({self.int_expr(scope, depth + 1)}, {self.int_expr(scope, depth + 1)})"
        if roll < 0.62:
            return f"math.lcm({self.small(scope, depth, 1000)}, {self.small(scope, depth, 1000)})"
        if roll < 0.7:
            return f"math.comb({self.small(scope, depth, 40)}, {self.small(scope, depth, 40)})"
        if roll < 0.76:
            return f"math.factorial({self.small(scope, depth, 22)})"
        if roll < 0.82:
            return f"math.isqrt({self.int_expr(scope, depth + 1)})"
        if roll < 0.86:
            return (
                f"bisect.bisect_left(sorted({self.int_list(scope, depth)}), {self.int_literal()})"
            )
        if roll < 0.88:
            return f"len(list(itertools.combinations(range({self.small(scope, depth, 8)}), 2)))"
        return self.library_int(scope, depth)

    def library_int(self, scope: _Scope, depth: int) -> str:
        """`collections`, `functools`, `operator`, and a `random.Random` of its own."""
        rng = self.rng
        roll = rng.random()
        items = self.int_list(scope, depth)
        if roll < 0.12:
            start = self.int_expr(scope, depth + 1)
            return (
                f"functools.reduce(operator.{rng.choice(('add', 'sub', 'xor'))}, {items}, {start})"
            )
        if roll < 0.2:
            return f"functools.reduce({rng.choice(('max', 'min'))}, {items})"
        if roll < 0.3:
            return f"collections.Counter({items})[{self.int_expr(scope, depth + 1)}]"
        if roll < 0.38:
            return f"collections.Counter({items}).most_common(1)[0][{rng.randint(0, 1)}]"
        if roll < 0.46:
            return f"collections.Counter({items}).total()"
        if roll < 0.56:
            low = self.int_expr(scope, depth + 1)
            return (
                f"random.Random({self.seed}).randint({low}, {low} + {self.small(scope, depth, 30)})"
            )
        if roll < 0.64:
            return (
                f"random.Random({self.int_expr(scope, depth + 1)}).randrange({rng.randint(1, 90)})"
            )
        if roll < 0.74:
            return f"_cached({self.small(scope, depth, 60)})"
        if roll < 0.84:
            return f"collections.deque({items})[{rng.randint(-1, 0)}]"
        if roll < 0.92:
            return f"max({items}, key=operator.neg)"
        return f"sorted({items}, key=functools.cmp_to_key(lambda a, b: b - a))[0]"

    def stdlib_float(self, scope: _Scope, depth: int) -> str:
        rng = self.rng
        roll = rng.random()
        if roll < 0.2:
            return "random.random()"
        if roll < 0.35:
            left = self.float_expr(scope, depth + 1)
            return f"random.uniform({left}, {self.float_expr(scope, depth + 1)})"
        if roll < 0.45:
            return f"random.gauss({self.float_expr(scope, depth + 1)}, 1.0)"
        if roll < 0.55:
            return f"random.expovariate({rng.choice(('0.5', '2.0', '1.0'))})"
        if roll < 0.65:
            return f"math.fsum({self.float_list(scope, depth)})"
        if roll < 0.75:
            left = self.float_expr(scope, depth + 1)
            return f"math.hypot({left}, {self.float_expr(scope, depth + 1)})"
        if roll < 0.85:
            name = rng.choice(("atan", "tanh", "erf", "cbrt", "asinh", "log1p", "exp2"))
            return f"math.{name}({self.float_expr(scope, depth + 1)})"
        left = self.float_expr(scope, depth + 1)
        return f"math.copysign({left}, {self.float_expr(scope, depth + 1)})"

    def stdlib_statements(self, w: _Writer, scope: _Scope) -> None:
        rng = self.rng
        heap = self.name("h")
        w.put(f"{heap}: list[int] = {self.int_list(scope, 2)}")
        w.put(f"heapq.heapify({heap})")
        for _ in range(rng.randint(1, 4)):
            w.put(f"heapq.heappush({heap}, {self.int_expr(scope, 2)})")
        if self.chance(0.5):
            w.put(f"random.shuffle({heap})")
            w.put(f"heapq.heapify({heap})")
        total = self.name("n")
        w.put(f"{total}: int = heapq.heappop({heap}) + len({heap})")
        w.put(f"bisect.insort({heap}, {self.int_expr(scope, 2)})")
        w.put(f"{total} += sum({heap}) + {heap}[0]")
        scope.ints.append(total)
        if self.chance(0.6):
            self.library_statements(w, scope, total)
        if self.chance(0.4):
            self.drawn_text(w, scope)

    def drawn_text(self, w: _Writer, scope: _Scope) -> None:
        """Strings a `random.Random` of its own draws from a list of strings
        the function made, concatenated: each is the list's, not the draw's."""
        rng = self.rng
        parts, drawn, text = self.name("parts"), self.name("r"), self.name("t")
        width, count = rng.randint(1, 4), rng.randint(1, 6)
        w.put(f"{parts}: list[str] = [str(x) * {width} for x in range({count})]")
        w.put(f"{drawn} = random.Random({self.int_expr(scope, 2)})")
        w.put(f'{text}: str = ""')
        w.put(f"for _ in range({rng.randint(1, 5)}):")
        w.put(f"    {text} += {drawn}.choice({parts})")
        if self.chance(0.5):
            w.put(f"{parts}.append({drawn}.choice({parts}) + {text})")
            w.put(f"{text} += {parts}[-1]")
        scope.strs.append(text)

    def library_statements(self, w: _Writer, scope: _Scope, total: str) -> None:
        """A `defaultdict`, a `Counter`, an `OrderedDict`, and a `deque` filled
        and read, and shown, whose text CPython's `repr` decides."""
        rng = self.rng
        table = self.name("d")
        factory = rng.choice(("int", "lambda: -1"))
        w.put(f"{table}: collections.defaultdict[int, int] = collections.defaultdict({factory})")
        for _ in range(rng.randint(1, 4)):
            w.put(f"{table}[{self.small(scope, 2, 6)}] += {self.int_expr(scope, 2)}")
        w.put(f"{total} += {table}[{self.small(scope, 2, 8)}] + len({table})")
        counts = self.name("c")
        w.put(f"{counts} = collections.Counter({self.int_list(scope, 2)})")
        w.put(f"{counts}.update({self.int_list(scope, 2)})")
        if self.chance(0.5):
            w.put(f"{counts}.subtract({self.int_list(scope, 2)})")
        ordered = self.name("od")
        w.put(f"{ordered}: collections.OrderedDict[int, int] = collections.OrderedDict()")
        moved = self.name("k")
        w.put(f"{moved}: int = {self.small(scope, 2, 5)}")
        w.put(f"{ordered}[{moved}] = {self.int_expr(scope, 2)}")
        for _ in range(rng.randint(1, 4)):
            w.put(f"{ordered}[{self.small(scope, 2, 5)}] = {self.int_expr(scope, 2)}")
        w.put(f"{ordered}.move_to_end({moved}, last={rng.choice(('True', 'False'))})")
        queue = self.name("q")
        w.put(f"{queue} = collections.deque({self.int_list(scope, 2)})")
        w.put(f"{queue}.rotate({rng.randint(-3, 3)})")
        w.put(f"{queue}.appendleft({self.int_expr(scope, 2)})")
        if factory == "int":
            shown = self.name("s")
            w.put(
                f"{shown}: str = str({table}) + str({counts}) + str({counts}.most_common(2))"
                f" + str({ordered}) + str({queue})"
            )
            scope.strs.append(shown)
        else:
            w.put(f"{total} += sum({counts}.values()) + sum({ordered}.values()) + {queue}[0]")

    def bool_expr(self, scope: _Scope, depth: int = 0) -> str:
        rng = self.rng
        if scope.bools and self.chance(0.2):
            return rng.choice(scope.bools)
        roll = rng.random()
        compare = rng.choice(("<", "<=", ">", ">=", "==", "!="))
        if roll < 0.45 or depth > 2:
            return f"({self.int_expr(scope, 2)} {compare} {self.int_expr(scope, 2)})"
        if roll < 0.65:
            return f"({self.float_expr(scope, 2)} {compare} {self.float_expr(scope, 2)})"
        if roll < 0.75:
            return f"({self.str_expr(scope, 2)} {compare} {self.str_expr(scope, 2)})"
        if roll < 0.8:
            return f"({self.str_expr(scope, 2)} in {self.str_expr(scope, 2)})"
        if roll < 0.84:
            items = ", ".join(self.int_expr(scope, 2) for _ in range(rng.randint(1, 4)))
            test = rng.choice(("in", "not in"))
            container = rng.choice((f"({items},)", f"{{{items}}}"))
            if self.chance(0.4):
                step = rng.choice(("1", "3", "-2", self.int_expr(scope, 2) + " % 5 + 1"))
                container = f"range({self.int_expr(scope, 2)}, {self.int_expr(scope, 2)}, {step})"
            return f"({self.int_expr(scope, 2)} {test} {container})"
        if roll < 0.87:
            ops = [rng.choice(("<", "<=", ">", ">=", "==", "!=")) for _ in range(rng.randint(2, 3))]
            spelled = self.int_expr(scope, 2)
            for op in ops:
                spelled += f" {op} {self.int_expr(scope, 2)}"
            return f"({spelled})"
        joiner = rng.choice(("and", "or"))
        left = self.bool_expr(scope, depth + 1)
        return f"({left} {joiner} not {self.bool_expr(scope, depth + 1)})"

    def str_expr(self, scope: _Scope, depth: int = 0) -> str:
        rng = self.rng
        if depth > 2 or self.chance(0.3):
            if scope.strs and self.chance(0.6):
                return rng.choice(scope.strs)
            return repr(rng.choice(_WORDS))
        roll = rng.random()
        text = self.str_expr(scope, depth + 1)
        if roll < 0.2:
            return f"({text} + {self.str_expr(scope, depth + 1)})"
        if roll < 0.28:
            return f"({text} * {rng.randint(0, 3)})"
        if roll < 0.44:
            method = rng.choice(("upper", "lower", "strip", "title", "swapcase", "capitalize"))
            return f"{text}.{method}()"
        if roll < 0.52:
            return f"{text}.replace({rng.choice(_WORDS)!r}, {rng.choice(_WORDS)!r})"
        if roll < 0.6:
            low, high = sorted((rng.randint(-4, 6), rng.randint(-4, 6)))
            step = rng.choice(("", ":2", ":-1"))
            return f"{text}[{low}:{high}{step}]"
        if roll < 0.68:
            return f"str({self.int_expr(scope, depth + 1)})"
        if roll < 0.74:
            return f"str({self.float_expr(scope, depth + 1)})"
        if roll < 0.82:
            return f'"-".join({text}.split())'
        if roll < 0.92:
            spec = rng.choice(("", ":>6", ":<4", ":^7", ":08.3f", ":,", ":x", ":+d"))
            if spec == ":08.3f":
                return f'f"<{{{self.float_expr(scope, depth + 1)}{spec}}}>"'
            if spec in {":,", ":x", ":+d"}:
                return f'f"[{{{self.int_expr(scope, depth + 1)}{spec}}}]"'
            return f'f"{{{text}{spec}}}|{{{self.int_expr(scope, depth + 1)}}}"'
        return f"({text} if {self.bool_expr(scope, depth + 1)} else {self.str_expr(scope, 3)})"

    # -- statements ----------------------------------------------------------

    def statement(self, w: _Writer, scope: _Scope, budget: int, ret: str) -> None:
        rng = self.rng
        if self.stdlib and self.chance(0.08):
            self.stdlib_statements(w, scope)
            return
        if self.prints and self.chance(0.15):
            self.print_statement(w, scope)
            return
        roll = rng.random()
        if roll < 0.14:
            name = self.name("n")
            w.put(f"{name}: int = {self.int_expr(scope)}")
            scope.ints.append(name)
        elif roll < 0.22:
            name = self.name("f")
            w.put(f"{name}: float = {self.float_expr(scope)}")
            scope.floats.append(name)
        elif roll < 0.3:
            name = self.name("s")
            w.put(f"{name}: str = {self.str_expr(scope)}")
            scope.strs.append(name)
        elif roll < 0.34:
            name = self.name("b")
            w.put(f"{name}: bool = {self.bool_expr(scope)}")
            scope.bools.append(name)
        elif roll < 0.42 and scope.ints:
            op = rng.choice(("+=", "-=", "*=", "//=", "%="))
            right = self.int_expr(scope, 2)
            if op in {"//=", "%="}:
                right = f"({right} if {right} != 0 else 3)"
            w.put(f"{rng.choice(scope.ints)} {op} {right}")
        elif roll < 0.5:
            w.put(f"if {self.bool_expr(scope)}:")
            self.block(w, scope, min(budget, 2), ret)
            w.put("else:")
            self.block(w, scope, 1, ret)
        elif roll < 0.58:
            index = self.name("i")
            w.put(f"for {index} in range({rng.randint(0, 5)}):")
            inner = scope.copy()
            inner.ints.append(index)
            w.depth += 1
            for _ in range(max(1, min(budget, 2))):
                self.statement(w, inner, 0, ret)
            w.depth -= 1
        elif roll < 0.6:
            self.vec_statements(w, scope)
        elif roll < 0.68:
            self.container_statements(w, scope)
        elif roll < 0.71:
            self.map_statements(w, scope)
        elif roll < 0.74:
            self.heap_statements(w, scope)
        elif roll < 0.78:
            self.tree_statements(w, scope)
        elif roll < 0.82:
            name = self.name("p")
            w.put(f"{name} = Point({self.int_expr(scope, 2)}, {self.int_expr(scope, 2)})")
            total = self.name("n")
            w.put(f"{total}: int = {name}.x * 3 - {name}.y")
            scope.ints.append(total)
        elif roll < 0.86:
            name = self.name("c")
            w.put(f"{name} = Counter({self.int_expr(scope, 2)})")
            total = self.name("n")
            w.put(f"{total}: int = {name}.bump({self.int_expr(scope, 2)})")
            w.put(f"{total} += {name}.bump({self.int_expr(scope, 2)})")
            scope.ints.append(total)
        elif roll < 0.9:
            name = self.name("t")
            w.put(f"{name} = ({self.int_expr(scope, 2)}, {self.float_expr(scope, 2)})")
            first, second = self.name("n"), self.name("f")
            w.put(f"{first}, {second} = {name}")
            scope.ints.append(first)
            scope.floats.append(second)
        elif roll < 0.92:
            w.put(f"if {self.bool_expr(scope)}:")
            w.depth += 1
            w.put(f"return {self.value(ret, scope)}")
            w.depth -= 1
        elif roll < 0.925:
            self.try_statement(w, scope, ret)
        elif roll < 0.945:
            self.closure_statement(w, scope)
        elif roll < 0.96:
            self.generator_statement(w, scope)
        elif roll < 0.975:
            if self.chance(0.5):
                self.lifted_statement(w, scope)
            else:
                self.expression_statement(w, scope)
        elif roll < 0.985:
            # An assert that holds more often than not, so programs run on.
            holds = self.chance(0.7)
            w.put(f"assert {self.bool_expr(scope, 2)} or {holds}, {self.str_expr(scope, 2)}")
        else:
            count = self.name("k")
            w.put(f"{count}: int = 0")
            w.put(f"while {count} < {rng.randint(0, 4)}:")
            w.depth += 1
            w.put(f"{count} += 1")
            self.statement(w, scope.copy(), 0, ret)
            w.depth -= 1
            scope.ints.append(count)

    def print_statement(self, w: _Writer, scope: _Scope) -> None:
        """A print between two checks: one before it that may fall back, and one
        after it that may too, often past 64 bits. The line is tagged, so a line
        printed twice is plain to see."""
        rng = self.rng
        before = self.name("n")
        w.put(f"{before}: int = {self.int_expr(scope, 1)}")
        scope.ints.append(before)
        tag = f"<{self.name('p')}>"
        kind = rng.choice(("int", "float", "str", "bool"))
        options = ""
        if self.chance(0.3):
            options += f", sep={rng.choice(('', '|', ', '))!r}"
        if self.chance(0.2):
            options += f", end={rng.choice(('', ';', '!\n'))!r}"
        if self.chance(0.15):
            options += ", flush=True"
        w.put(f"print({self.value(kind, scope)}, {tag!r}{options})")
        after = self.name("n")
        big = rng.choice(_BIG)
        w.put(f"{after}: int = {before} * {big} + {self.int_expr(scope, 2)}")
        scope.ints.append(after)

    def block(self, w: _Writer, scope: _Scope, budget: int, ret: str) -> None:
        w.depth += 1
        inner = scope.copy()
        for _ in range(max(1, budget)):
            self.statement(w, inner, 0, ret)
        w.depth -= 1

    def try_statement(self, w: _Writer, scope: _Scope, ret: str) -> None:
        """`try` over something that may raise, caught by class, with `else`
        and `finally` now and then; a caught exception's text is kept."""
        rng = self.rng
        name = self.name("n")
        w.put(f"{name}: int = 0")
        w.put("try:")
        w.depth += 1
        roll = rng.random()
        if roll < 0.3:
            w.put(f"{name} = {self.int_expr(scope, 1)} // {self.int_expr(scope, 2)}")
        elif roll < 0.5 and scope.vecs:
            w.put(f"{name} = {rng.choice(scope.vecs)}[{self.int_expr(scope, 2)}]")
        elif roll < 0.65:
            w.put(f"{name} = int({self.float_expr(scope, 1)})")
        elif roll < 0.8:
            w.put(f"if {self.bool_expr(scope, 2)}:")
            w.put(f"    raise Broken({self.str_expr(scope, 2)})")
            w.put(f"{name} = {self.int_expr(scope, 2)}")
        else:
            w.put(f"assert {self.bool_expr(scope, 2)}, {self.str_expr(scope, 2)}")
            w.put(f"{name} = {self.int_expr(scope, 2)}")
        if self.chance(0.2):
            w.put(f"if {self.bool_expr(scope, 2)}:")
            w.put(f"    return {self.value(ret, scope)}")
        w.depth -= 1
        caught = rng.choice(
            (
                "ZeroDivisionError",
                "IndexError",
                "(ValueError, OverflowError)",
                "Broken",
                "ValueError",
                "AssertionError",
                "ArithmeticError",
                "Exception",
            )
        )
        text = self.name("s")
        w.put(f"except {caught} as e:")
        w.depth += 1
        w.put(f"{text}: str = str(e)")
        w.put(f"{name} = len({text}) + {rng.randint(1, 9)}")
        if self.chance(0.1):
            w.put("raise")
        w.depth -= 1
        if self.chance(0.3):
            w.put("else:")
            w.put(f"    {name} += 1")
        if self.chance(0.3):
            w.put("finally:")
            w.put(f"    {name} = {name} * 2")
        scope.ints.append(name)

    def generator_statement(self, w: _Writer, scope: _Scope) -> None:
        """A generator consumed where it lowers: a loop, a reduction, `next`,
        a collection built from it."""
        rng = self.rng
        name = self.name("n")
        count = f"({self.int_expr(scope, 2)} % 9)"
        source = rng.choice(
            (f"countdown({count})", f"pairs({count})", f"(x * 3 for x in range({count}))")
        )
        roll = rng.random()
        if roll < 0.3:
            w.put(f"{name}: int = 0")
            item = self.name("x")
            w.put(f"for {item} in {source}:")
            w.put(f"    if {item} > {rng.randint(10, 40)}:")
            w.put("        break")
            w.put(f"    {name} = {name} * 3 + {item}")
        elif roll < 0.5:
            w.put(f"{name}: int = sum(Vec[int]({source}))")
        elif roll < 0.62:
            w.put(f"{name}: int = sum({source})")
        elif roll < 0.74:
            # Never empty: `min` of nothing is CPython's ValueError, which the
            # `try` forms already reach.
            nonempty = source.replace("% 9)", "% 9 + 1)")
            w.put(f"{name}: int = {rng.choice(('min', 'max'))}({nonempty})")
        elif roll < 0.84:
            w.put(f"{name}: int = next({source}, {self.int_expr(scope, 2)})")
        elif roll < 0.92:
            test = rng.choice(("any", "all"))
            w.put(f"{name}: int = 1 if {test}(x > 4 for x in {source}) else 0")
        else:
            w.put(f"{name}: int = 0")
            item = self.name("x")
            w.put(f"for {item} in sorted({source}):")
            w.put(f"    {name} = {name} * 7 + {item}")
        scope.ints.append(name)

    def expression_statement(self, w: _Writer, scope: _Scope) -> None:
        """What 0.6.0 took native: `isinstance`, loops over ranges with a
        step, strings, and tuples under `enumerate`, `zip`, and `reversed`,
        chained assignment, `e.args`, and generators held, passed on, and
        made by `__iter__`."""
        rng = self.rng
        roll = rng.random()
        name = self.name("n")
        if roll < 0.12:
            subject = rng.choice([*scope.ints, *scope.floats, *scope.strs, *scope.bools, "None"])
            classes = rng.choice(("int", "float", "(int, float)", "str", "bool", "(str, list)"))
            w.put(f"{name}: int = 1 if isinstance({subject}, {classes}) else 0")
        elif roll < 0.3:
            step = rng.choice(("1", "2", "-1", "-3", f"({self.int_expr(scope, 2)} % 4 or 1)"))
            source = rng.choice(
                (
                    f"range({self.int_expr(scope, 2)} % 9, {self.int_expr(scope, 2)} % 9, {step})",
                    f"reversed(range({self.int_expr(scope, 2)} % 7))",
                    f"({self.int_expr(scope, 2)}, {self.int_expr(scope, 2)}, 5)",
                )
            )
            wrapped = rng.choice(("plain", "enumerate", "zip"))
            w.put(f"{name}: int = 0")
            if wrapped == "plain":
                w.put(f"for x in {source}:")
                w.put(f"    {name} = {name} * 3 + x")
            elif wrapped == "enumerate":
                w.put(f"for i, x in enumerate({source}, {rng.randint(0, 2)}):")
                w.put(f"    {name} = {name} * 3 + i * x")
            else:
                w.put(f"for x, c in zip({source}, {self.str_expr(scope, 2)}):")
                w.put(f"    {name} = {name} * 3 + x + ord(c)")
        elif roll < 0.42:
            other = self.name("n")
            w.put(f"{name} = {other} = {self.int_expr(scope, 2)}")
            w.put(f"{other} += 1")
            scope.ints.append(other)
        elif roll < 0.55:
            w.put(f"{name}: int = 0")
            w.put("try:")
            w.put(f"    if {self.bool_expr(scope, 2)}:")
            w.put(f"        raise ValueError({self.str_expr(scope, 2)})")
            w.put("    raise KeyError()")
            w.put("except ValueError as e:")
            w.put(f"    {name} = len(str(e.args)) * 10 + len(e.args)")
            w.put("except KeyError as e:")
            w.put(f"    {name} = len(e.args) - 1")
        elif roll < 0.8:
            count = f"({self.int_expr(scope, 2)} % 9)"
            it = self.name("it")
            w.put(f"{it} = countdown({count})")
            w.put(f"{name}: int = 0")
            w.put(f"for k in range({rng.randint(0, 3)}):")
            w.put(f"    {name} += next({it}, -1) * (k + 2)")
            w.put(f"{name} += drain({it}, {rng.randint(0, 4)})")
            w.put(f"{name} += drain(pairs({count}), {rng.randint(0, 4)})")
        else:
            ring = self.name("r")
            w.put(f"{ring} = Ring({rng.randint(0, 3)})")
            for _ in range(rng.randint(0, 4)):
                w.put(f"{ring}.add({self.str_expr(scope, 2)})")
            w.put(f"{name}: int = 0")
            w.put(f"for word in {ring}:")
            w.put(f"    {name} = {name} * 7 + len(word)")
            w.put(f"    if {name} > {rng.randint(5, 60)}:")
            w.put("        break")
        scope.ints.append(name)

    def lifted_statement(self, w: _Writer, scope: _Scope) -> None:
        """What 0.5.0 took native: a set's order shown, float and bool keys, a
        dataclass's `==`, order, and repr, an exception's fields, a generator
        stepped by hand, and `str.format` and `%`."""
        rng = self.rng
        roll = rng.random()
        if roll < 0.2:
            a, b = self.name("sa"), self.name("sb")
            members = ", ".join(self.int_expr(scope, 2) for _ in range(rng.randint(0, 6)))
            w.put(f"{a}: set[int] = {{{members}}}" if members else f"{a}: set[int] = set()")
            step, count = rng.randint(1, 9), rng.randint(0, 12)
            w.put(f"{b}: set[int] = {{x * {step} % 17 for x in range({count})}}")
            for _ in range(rng.randint(1, 4)):
                op = rng.random()
                if op < 0.3:
                    w.put(f"{a}.add({self.int_expr(scope, 2)})")
                elif op < 0.45:
                    w.put(f"{a}.discard({rng.randint(-2, 16)})")
                elif op < 0.7:
                    w.put(f"{a} {rng.choice(('|=', '&=', '-=', '^='))} {b}")
                elif op < 0.85:
                    left, right = rng.choice((a, b)), rng.choice((a, b))
                    w.put(f"{a} = {left} {rng.choice(('|', '&', '-', '^'))} {right}")
                else:
                    w.put(f"if {a}:")
                    w.put(f"    {a}.pop()")
            text = self.name("s")
            w.put(f"{text}: str = str({a}) + str(list({b}))")
            scope.strs.append(text)
        elif roll < 0.35:
            d = self.name("d")
            keyed = rng.choice(("float", "bool"))
            w.put(f"{d}: dict[{keyed}, int] = {{}}")
            for _ in range(rng.randint(1, 4)):
                if keyed == "float":
                    key = rng.choice((*_FLOATS, "0.0", self.float_expr(scope, 2)))
                else:
                    key = self.bool_expr(scope, 2)
                w.put(f"if {key} == {key}:")
                w.put(f"    {d}[{key}] = {d}.get({key}, 0) + {self.int_expr(scope, 2)}")
            text = self.name("s")
            w.put(f"{text}: str = str({d})")
            scope.strs.append(text)
        elif roll < 0.55:
            a, b = self.name("g"), self.name("g")
            w.put(f"{a} = Tag({self.int_expr(scope, 2)} % 4, {self.str_expr(scope, 2)})")
            w.put(f"{b} = Tag({rng.randint(0, 3)}, {rng.choice(_WORDS)!r})")
            op = rng.choice(("==", "!=", "<", "<=", ">", ">="))
            text = self.name("s")
            w.put(f'{text}: str = f"{{{a}}} {{{b}!r}} {{{a} {op} {b}}}"')
            scope.strs.append(text)
        elif roll < 0.7:
            name = self.name("n")
            w.put(f"{name}: int = 0")
            w.put("try:")
            w.put(f"    if {self.bool_expr(scope, 2)}:")
            w.put(f"        raise Failed({self.int_expr(scope, 2)}, {self.str_expr(scope, 2)})")
            w.put(f"    {name} = {self.int_expr(scope, 2)}")
            w.put("except Failed as e:")
            w.put(f"    {name} = e.code % 1000 + len(e.where) + len(str(e))")
            scope.ints.append(name)
        elif roll < 0.85:
            name, it = self.name("n"), self.name("it")
            w.put(f"{it} = countdown(({self.int_expr(scope, 2)}) % 9)")
            w.put(f"{name}: int = next({it}, -1)")
            w.put(f"{name} *= 100")
            item = self.name("x")
            w.put(f"for {item} in {it}:")
            w.put(f"    {name} += {item}")
            last = self.name("n")
            w.put(f"{last}: int = next({it}, {rng.randint(0, 9)})")
            w.put(f"{name} += {last}")
            scope.ints.append(name)
        else:
            text = self.name("s")
            if self.chance(0.5):
                w.put(
                    f'{text}: str = "{{:>6}}|{{:.3f}}|{{!r}}|{{:+d}}".format('
                    f"{self.str_expr(scope, 2)}, {self.float_expr(scope, 2)}, "
                    f"{self.str_expr(scope, 2)}, {self.int_expr(scope, 2)})"
                )
            else:
                w.put(
                    f'{text}: str = "%-5s|%05d|%.2e|%x" % ('
                    f"{self.str_expr(scope, 2)}, {self.int_expr(scope, 2)}, "
                    f"{self.float_expr(scope, 2)}, {self.int_expr(scope, 2)})"
                )
            scope.strs.append(text)

    def closure_statement(self, w: _Writer, scope: _Scope) -> None:
        """A nested function or a lambda over the block's names: called, keeping
        a count through `nonlocal`, a sort or reduction key, a `Callable` local
        mapped over a range, or a closure a nested function makes."""
        rng = self.rng
        name = self.name("n")
        roll = rng.random()
        if roll < 0.25:
            inner = self.name("g")
            w.put(f"def {inner}(x: int) -> int:")
            w.put(f"    return x * {rng.randint(-3, 3)} + {self.int_expr(scope, 2)}")
            w.put(
                f"{name}: int = {inner}({self.int_expr(scope, 2)}) - {inner}({self.int_literal()})"
            )
        elif roll < 0.45:
            bump = self.name("bump")
            w.put(f"{name}: int = {self.int_expr(scope, 2)}")
            w.put(f"def {bump}(by: int) -> None:")
            w.put(f"    nonlocal {name}")
            w.put(f"    {name} = {name} * 2 + by")
            w.put(f"{bump}({self.int_expr(scope, 2)})")
            w.put(f"{bump}({self.int_literal()})")
        elif roll < 0.65:
            items = self.name("xs")
            values = ", ".join(self.int_expr(scope, 2) for _ in range(rng.randint(1, 4)))
            w.put(f"{items}: list[int] = [{values}]")
            key = rng.choice(
                (
                    f"lambda v: v % {rng.randint(2, 5)}",
                    f"lambda v: (v % {rng.randint(2, 5)}, -v)",
                    f"lambda v: v * {self.int_expr(scope, 2)}",
                )
            )
            reverse = rng.choice(("", ", reverse=True"))
            w.put(f"{items}.sort(key={key}{reverse})")
            which = rng.choice(("min", "max"))
            w.put(f"{name}: int = {items}[0] * 7 + {which}({items}, key={key})")
        elif roll < 0.85:
            function = self.name("h")
            w.put(f"{function}: Callable[[int], int] = lambda v: v * 3 + {self.int_expr(scope, 2)}")
            count = rng.randint(0, 5)
            if self.chance(0.5):
                w.put(f"{name}: int = sum(map({function}, range({count})))")
            else:
                w.put(f"{name}: int = 0")
                item = self.name("x")
                w.put(f"for {item} in filter(lambda v: v % 2 == 0, range({count})):")
                w.put(f"    {name} += {function}({item})")
        else:
            make = self.name("make")
            w.put(f"def {make}(k: int) -> Callable[[int], int]:")
            w.put(f"    return lambda v: v * k + {self.int_expr(scope, 2)}")
            made = self.name("h")
            w.put(f"{made} = {make}({self.int_expr(scope, 2)})")
            w.put(f"{name}: int = {made}({self.int_literal()}) + {made}({self.int_literal()})")
        scope.ints.append(name)

    def container_statements(self, w: _Writer, scope: _Scope) -> None:
        """Python's own `list`, `dict`, and `set`: displays, comprehensions,
        methods, negative indices, `in`, `del`, and reductions over them."""
        rng = self.rng
        roll = rng.random()
        total = self.name("n")
        if roll < 0.45:
            xs = self.name("xs")
            items = ", ".join(self.int_expr(scope, 2) for _ in range(rng.randint(1, 4)))
            if self.chance(0.3):
                w.put(
                    f"{xs}: list[int] = [x * 2 - 1 for x in range({rng.randint(1, 6)}) if x != 2]"
                )
                w.put(f"{xs}.append({self.int_expr(scope, 2)})")
            else:
                w.put(f"{xs}: list[int] = [{items}]")
            for _ in range(rng.randint(1, 3)):
                op = rng.random()
                if op < 0.25:
                    w.put(f"{xs}.append({self.int_expr(scope, 2)})")
                elif op < 0.4:
                    w.put(f"{xs}.insert({rng.randint(-2, 2)}, {self.int_expr(scope, 2)})")
                elif op < 0.55:
                    w.put(f"{xs}[-1] = {xs}[0] + len({xs})")
                elif op < 0.7:
                    w.put(f"{xs}.sort(reverse={rng.choice(('True', 'False'))})")
                elif op < 0.8:
                    w.put(f"if len({xs}) > 1:")
                    w.put(f"    del {xs}[{rng.choice(('0', '-1'))}]")
                elif op < 0.9:
                    w.put(f"{xs}.extend([{self.int_expr(scope, 2)}, {rng.randint(-5, 5)}])")
                else:
                    w.put(f"{xs} = {xs}[::-1] + {xs}[1:]")
            pick = rng.random()
            if pick < 0.2:
                w.put(f"{total}: int = {xs}[{self.int_expr(scope, 2)} % len({xs})] + {xs}[-1]")
            elif pick < 0.4:
                w.put(f"{total}: int = sum({xs}) + {rng.choice(('min', 'max'))}({xs})")
            elif pick < 0.55:
                w.put(f"{total}: int = sorted({xs})[0] * 3 + len({xs})")
            elif pick < 0.7:
                w.put(f"{total}: int = 1 if {self.int_expr(scope, 2)} in {xs} else 0")
            elif pick < 0.85:
                w.put(f"{total}: int = {xs}.count({xs}[0]) + {xs}.index({xs}[-1])")
            else:
                w.put(f"{total}: int = {xs}.pop() + len({xs})")
        elif roll < 0.8:
            d = self.name("d")
            keyed = rng.choice(("int", "str"))
            w.put(f"{d}: dict[{keyed}, int] = {{}}")
            for _ in range(rng.randint(1, 4)):
                key = self.int_expr(scope, 2) if keyed == "int" else self.str_expr(scope, 2)
                w.put(f"{d}[{key}] = {d}.get({key}, 0) + {self.int_expr(scope, 2)}")
            if self.chance(0.3):
                key = self.int_expr(scope, 2) if keyed == "int" else self.str_expr(scope, 2)
                w.put(f"{d}.pop({key}, 0)")
            if self.chance(0.2):
                # `None` on a miss: no number holds that natively.
                key = self.int_expr(scope, 2) if keyed == "int" else self.str_expr(scope, 2)
                found, got = self.name("v"), self.name("n")
                w.put(f"{found} = {d}.get({key})")
                w.put(f"{got}: int = -1 if {found} is None else {found}")
                scope.ints.append(got)
            if self.chance(0.2) and keyed == "int":
                w.put(f"{d} = {{k: v * 2 for k, v in {d}.items() if v != 0}}")
            w.put(f"{total}: int = len({d}) * 100")
            key = self.name("k")
            if self.chance(0.5):
                w.put(f"for {key}, value in {d}.items():")
                w.put(f"    {total} = {total} * 3 + value")
            else:
                w.put(f"{total} += sum({d}.values())")
            if self.chance(0.3) and keyed == "int":
                probe = rng.randint(-3, 3)
                w.put(f"if {probe} in {d}:")
                w.put(f"    del {d}[{probe}]")
                w.put(f"    {total} += 1")
        else:
            a, b = self.name("sa"), self.name("sb")
            members = ", ".join(self.int_expr(scope, 2) for _ in range(rng.randint(1, 4)))
            w.put(f"{a}: set[int] = {{{members}}}")
            w.put(f"{b}: set[int] = {{x % 5 for x in range({rng.randint(1, 8)})}}")
            w.put(f"{a}.add({self.int_expr(scope, 2)})")
            w.put(f"{a}.discard({rng.randint(-2, 4)})")
            op = rng.choice(("|", "&", "-", "^"))
            w.put(
                f"{total}: int = len({a} {op} {b}) * 10 + (1 if {rng.randint(0, 4)} in {b} else 0)"
            )
            if self.chance(0.5):
                w.put(f"{total} += sum({a}) + (max({b}) if {b} else 0)")
        scope.ints.append(total)

    def vec_statements(self, w: _Writer, scope: _Scope) -> None:
        rng = self.rng
        name = self.name("v")
        w.put(f"{name} = Vec[int]()")
        for _ in range(rng.randint(1, 4)):
            w.put(f"{name}.push({self.int_expr(scope, 2)})")
        roll = rng.random()
        if roll < 0.3:
            w.put(f"{name}.sort()")
        elif roll < 0.45:
            w.put(f"{name}.reverse()")
        elif roll < 0.55:
            w.put(f"{name}.sort(reverse=True)")
        total = self.name("n")
        pick = rng.random()
        if pick < 0.3:
            w.put(f"{total}: int = {name}[{self.int_expr(scope, 2)} % len({name})]")
        elif pick < 0.5:
            w.put(f"{total}: int = {name}.pop() + len({name})")
        elif pick < 0.7:
            w.put(f"{total}: int = biggest({name})")
        elif pick < 0.85:
            w.put(f"{total}: int = sum({name})")
        else:
            w.put(f"{total}: int = 0")
            item = self.name("x")
            w.put(f"for {item} in {name}:")
            w.put(f"    {total} = {total} * 7 + {item}")
        scope.ints.append(total)
        scope.vecs.append(name)

    def map_statements(self, w: _Writer, scope: _Scope) -> None:
        rng = self.rng
        name = self.name("m")
        keyed = rng.choice(("int", "str"))
        w.put(f"{name} = HashMap[{keyed}, int]()")
        for _ in range(rng.randint(1, 4)):
            key = self.int_expr(scope, 2) if keyed == "int" else self.str_expr(scope, 2)
            w.put(f"{name}[{key}] = {name}.get({key}, 0) + {self.int_expr(scope, 2)}")
        total = self.name("n")
        w.put(f"{total}: int = len({name}) * 100")
        key = self.name("key")
        w.put(f"for {key} in {name}:")
        w.put(f"    {total} = {total} * 3 + {name}[{key}]")
        scope.ints.append(total)

    def heap_statements(self, w: _Writer, scope: _Scope) -> None:
        rng = self.rng
        name = self.name("h")
        w.put(f"{name} = Heap[int]()")
        for _ in range(rng.randint(1, 5)):
            w.put(f"{name}.push({self.int_expr(scope, 2)})")
        total = self.name("n")
        w.put(f"{total}: int = 0")
        w.put(f"while {name}:")
        w.put(f"    {total} = {total} * 5 + {name}.pop()")
        scope.ints.append(total)

    def tree_statements(self, w: _Writer, scope: _Scope) -> None:
        rng = self.rng
        name = self.name("tm")
        w.put(f"{name} = TreeMap[int, int]()")
        for _ in range(rng.randint(1, 4)):
            w.put(f"{name}[{self.int_expr(scope, 2)}] = {self.int_expr(scope, 2)}")
        total = self.name("n")
        w.put(f"{total}: int = {name}.min() * 10 + {name}.max() + len({name})")
        if self.chance(0.4):
            queue = self.name("q")
            w.put(f"{queue} = Deque[int]()")
            w.put(f"{queue}.push_front({total})")
            w.put(f"{queue}.push_back({self.int_expr(scope, 2)})")
            w.put(f"{total} = {queue}.pop_back() - {queue}.pop_front()")
        scope.ints.append(total)

    def value(self, kind: str, scope: _Scope) -> str:
        if kind == "int":
            return self.int_expr(scope)
        if kind == "float":
            return self.float_expr(scope)
        if kind == "str":
            return self.str_expr(scope)
        return self.bool_expr(scope)

    # -- functions -----------------------------------------------------------

    def function(self, w: _Writer, name: str) -> tuple[list[str], str]:
        rng = self.rng
        kinds = [
            rng.choice(("int", "int", "float", "str", "bool")) for _ in range(rng.randint(1, 3))
        ]
        ret = rng.choice(("int", "int", "float", "str", "bool"))
        params = [self.name("a") for _ in kinds]
        signature = ", ".join(f"{p}: {k}" for p, k in zip(params, kinds, strict=True))
        if self.calls:
            signature = self.call_signature(name, params, kinds)
        if self.unannotated:
            w.put(f"def {name}({_unannotated(signature)}):")
        else:
            w.put(f"def {name}({signature}) -> {ret}:")
        w.depth += 1
        scope = _Scope()
        for param, kind in zip(params, kinds, strict=True):
            {"int": scope.ints, "float": scope.floats, "str": scope.strs, "bool": scope.bools}[
                kind
            ].append(param)
        for _ in range(rng.randint(2, 6)):
            self.statement(w, scope, 2, ret)
        w.put(f"return {self.value(ret, scope)}")
        w.depth -= 1
        w.put("")
        w.put("")
        return kinds, ret

    def call_signature(self, name: str, params: list[str], kinds: list[str]) -> str:
        """Parameters with constant defaults on the last few, and the last ones
        keyword-only now and then."""
        rng = self.call_rng
        defaulted = rng.randint(0, len(params))
        defaults = {
            p: self.constant(k)
            for p, k in list(zip(params, kinds, strict=True))[len(params) - defaulted :]
        }
        keyword_only = rng.randint(1, len(params)) if rng.random() < 0.3 else len(params)
        self.signatures[name] = (params, kinds, defaults, keyword_only)
        parts = []
        for index, (param, kind) in enumerate(zip(params, kinds, strict=True)):
            if index == keyword_only:
                parts.append("*")
            default = defaults.get(param)
            parts.append(f"{param}: {kind}" + (f" = {default}" if default is not None else ""))
        return ", ".join(parts)

    def constant(self, kind: str) -> str:
        """A literal of `kind`, as a default may be."""
        rng = self.call_rng
        if kind == "int":
            return str(rng.randint(-9, 9))
        if kind == "float":
            return rng.choice(_FLOATS)
        if kind == "str":
            return repr(rng.choice(_WORDS))
        return rng.choice(("True", "False"))

    def call(self, name: str, kinds: list[str]) -> str:
        """A call of `name`: by position, or with keywords and defaults left out."""
        found = self.signatures.get(name)
        if found is None:
            return f"{name}({', '.join(self.argument(k) for k in kinds)})"
        params, kinds, defaults, keyword_only = found
        rng = self.call_rng
        positional: list[str] = []
        named: list[str] = []
        by_name = False
        for index, (param, kind) in enumerate(zip(params, kinds, strict=True)):
            if param in defaults and rng.random() < 0.4:
                by_name = True  # left out: whatever follows is named
                continue
            if index >= keyword_only or by_name or rng.random() < 0.3:
                by_name = True
                named.append(f"{param}={self.argument(kind)}")
            else:
                positional.append(self.argument(kind))
        rng.shuffle(named)
        return f"{name}({', '.join([*positional, *named])})"

    def argument(self, kind: str) -> str:
        rng = self.rng
        if kind == "int":
            return self.int_literal()
        if kind == "float":
            return (
                rng.choice((*_FLOATS, *_SPECIAL_FLOATS))
                if self.chance(0.2)
                else rng.choice(_FLOATS)
            )
        if kind == "str":
            return repr(rng.choice(_WORDS))
        return rng.choice(("True", "False"))

    # -- module state and objects ----------------------------------------------

    def state_globals(self, w: _Writer) -> None:
        """Module globals bound once and never rebound (settled), which native
        code is passed at the call: a list and a dict it reads and writes, a
        number and a string made by a call, so none is a folded constant."""
        rng = self.state
        w.put(
            f"G_TABLE: list[int] = [k * {rng.randint(1, 9)} - {rng.randint(0, 5)} "
            f"for k in range({rng.randint(3, 9)})]"
        )
        w.put("G_SEEN: dict[int, int] = {}")
        w.put(f'G_SCALE = int("{rng.randint(-3, 7)}")')
        w.put(f"G_WORD = str({rng.randint(0, 99)}) + {rng.choice(_WORDS)!r}")
        w.put("")
        w.put("")

    def state_function(self, w: _Writer, name: str, callee: str | None) -> None:
        """A function over the settled globals: reads them, writes the list and
        the dict, and, when `callee` is given, calls another that does, which
        passes the globals on."""
        rng = self.state
        w.put(f"def {name}(a: int, b: int) -> int:")
        w.depth += 1
        scope = _Scope(ints=["a", "b", "total"])
        w.put(f"total = {'0' if callee is None else f'{callee}(b, a)'}")
        w.put(f"for i in range(b % {rng.randint(2, 7)} + 1):")
        w.depth += 1
        inner = scope.copy()
        inner.ints.append("i")
        w.put("total += G_TABLE[(a + i) % len(G_TABLE)] * G_SCALE")
        if rng.random() < 0.7:
            w.put(f"G_SEEN[(a * i + {rng.randint(0, 9)}) % 11] = {self.int_expr(inner, 1)}")
        if rng.random() < 0.5:
            w.put(f"if len(G_TABLE) < 30 and {self.bool_expr(inner, 2)}:")
            w.put(f"    G_TABLE.append({self.int_expr(inner, 1)})")
        if rng.random() < 0.4:
            w.put(f"total += G_SEEN.get(i, {rng.randint(-2, 2)})")
        w.depth -= 1
        w.put(f"return total + len(G_WORD) + len(G_SEEN) + {self.int_expr(scope, 1)}")
        w.depth -= 1
        w.put("")
        w.put("")

    def object_function(self, w: _Writer, name: str) -> None:
        """A function that walks linked objects Python made, writing their
        fields: the objects cross whole, and the writes come back."""
        rng = self.state
        w.put(f"def {name}(head: Link, k: int) -> int:")
        w.depth += 1
        w.put("total = 0")
        w.put("node: Link | None = head")
        w.put("while node is not None:")
        w.depth += 1
        scope = _Scope(ints=["k", "total", "node.value"], strs=["node.label"])
        w.put(f"node.value = ({self.int_expr(scope, 1)}) % {rng.randint(50, 999)}")
        if rng.random() < 0.5:
            w.put(f"node.label = {self.str_expr(scope, 2)}")
        w.put("total += node.value + len(node.label)")
        if rng.random() < 0.3:
            w.put("if node.next is None and k > 0:")
            w.put('    node.next = Link(k, "new")')
            w.put("    k = 0")
        w.put("node = node.next")
        w.depth -= 1
        w.put("return total")
        w.depth -= 1
        w.put("")
        w.put("")

    def program(self) -> str:
        w = _Writer()
        if self.with_structures:
            w.lines.append(STRUCTURES_MARK)
        elif self.unannotated:
            w.lines.append(UNANNOTATED_MARK)
        if self.stdlib:
            w.lines.extend(["import bisect", "import heapq", "import itertools", "import random"])
            w.lines.extend(["import functools", "import operator"])
            w.lines.append("import collections")
        w.lines.extend(_PRELUDE.splitlines())
        if self.stdlib:
            w.lines.extend(_STDLIB_PRELUDE.splitlines())
        if self.with_state:
            w.lines.extend(_STATE_PRELUDE.splitlines())
            self.state_globals(w)
        if self.with_boundary:
            w.lines.extend(_BOUNDARY_PRELUDE.splitlines())
        if self.with_structures:
            w.lines.extend(_STRUCTURES_PRELUDE.splitlines())
        calls: list[str] = []
        foreign: list[tuple[str, str]] = []
        for _ in range(self.rng.randint(3, 6)):
            name = self.name("fn")
            kinds, _ret = self.function(w, name)
            calls.extend(self.call(name, kinds) for _ in range(self.rng.randint(1, 3)))
            if self.unannotated:
                foreign.append(self.foreign_call(name, kinds))
        after = self.state_part(w) if self.with_state else []
        if self.with_boundary:
            after.extend(self.boundary_part(w))
        if self.with_structures:
            after.extend(self.structures_part(w))
        w.put("def main() -> None:")
        if self.stdlib:
            w.put(f"    random.seed({self.seed})")
        for call in calls:
            w.put(f"    print({call})")
        for line in after:
            w.put(f"    {line}")
        w.put("")
        w.put("")
        w.put("main()")
        if foreign:
            # Python calls each function by a name the analysis cannot follow,
            # with other types: the native entry must refuse them and run the
            # Python body, which prints what CPython prints.
            w.put('if __name__ == "__main__":')
            w.put("    import sys")
            w.put("")
            w.put("    here = sys.modules[__name__]")
            for call in foreign:
                w.put("    try:")
                w.put(f"        print(getattr(here, {call[0]!r})({call[1]}))")
                w.put("    except Exception as e:")
                w.put("        print(type(e).__name__)")
        return "\n".join(w.lines) + "\n"

    def foreign_call(self, name: str, kinds: list[str]) -> tuple[str, str]:
        """A call of `name` with arguments of other types than `main` passes."""
        rng = self.foreign
        others = {
            "int": ("True", "2.5", "'7'", "-3"),
            "float": ("3", "True", "'x'", "-0.5"),
            "str": ("4", "b'ab'", "'ok'"),
            "bool": ("1", "0", "'y'", "False"),
        }
        return name, ", ".join(rng.choice(others[k]) for k in kinds)

    def state_part(self, w: _Writer) -> list[str]:
        """The functions over module state and objects, and what `main` does with them."""
        after: list[str] = []
        first = self.name("st")
        self.state_function(w, first, None)
        second = self.name("st")
        self.state_function(w, second, first)
        for _ in range(self.state.randint(1, 3)):
            a, b = self.state.randint(-5, 9), self.state.randint(0, 9)
            after.append(f"print({self.state.choice((first, second))}({a}, {b}))")
        after.append("print(G_TABLE, sorted(G_SEEN.items()))")
        walker = self.name("walk")
        self.object_function(w, walker)
        after.append(f"h = chain({self.state.randint(1, 6)})")
        after.extend(
            f"print({walker}(h, {self.state.randint(-3, 9)}), values(h), h.label)"
            for _ in range(self.state.randint(1, 2))
        )
        return after

    # -- writes across the boundary ----------------------------------------------

    def boundary_function(self, w: _Writer, name: str) -> None:
        """A function Python calls natively with containers and objects it
        writes through: a list of lists, a dict of lists, a set, and objects in
        a list, each of which the generated wrapper copies in and back."""
        rng = self.crossing
        w.put("@ppy.native")
        w.put(
            f"def {name}(g: list[list[int]], d: dict[int, list[int]], s: set[int], "
            "boxes: list[Box], k: int) -> int:"
        )
        w.depth += 1
        w.put("total = 0")
        modulus = rng.randint(7, 97)
        menu = [
            f"row[j] = (row[j] * {rng.randint(1, 5)} + k + i) % {modulus}",
            "total += row[j] * (i + 1)",
            (
                f"if len(row) < 8 and (row[j] + k) % {rng.randint(2, 4)} == 0:\n"
                f"    row.append((k + j) % {modulus})"
            ),
            f"if len(row) > 1 and row[j] > {rng.randint(10, 60)}:\n    row.pop()\n    break",
            "s.add((row[j] + k) % 13)",
            f"if row[j] % {rng.randint(2, 5)} == 1:\n    s.discard(row[j] % 13)",
            (
                "d.setdefault((row[j] + i) % 5, []).append(k)\nif len(d[(row[j] + i) % 5]) > 6:\n"
                "    d[(row[j] + i) % 5].pop(0)"
            ),
        ]
        w.put("for i in range(len(g)):")
        w.depth += 1
        w.put("row = g[i]")
        w.put("for j in range(len(row)):")
        w.depth += 1
        for statement in rng.sample(menu, rng.randint(2, 5)):
            for line in statement.split("\n"):
                w.put(line)
        w.depth -= 2
        tails = [
            "g.reverse()",
            "if len(g) > 0:\n    g[0].sort()",
            "if len(g) < 5 and len(g) > 0:\n    g.append(g[len(g) - 1])",
            "if len(g) < 5:\n    g.append([k % 7, k % 3])",
            "if len(g) > 2:\n    g.pop(0)",
            "for key in d:\n    if len(d[key]) < 6:\n        d[key].append(len(d[key]) + k)",
            "if k in d:\n    d[k].reverse()",
            (
                f"for b in boxes:\n    b.value = (b.value * {rng.randint(2, 5)} + k) % {modulus}\n"
                "    if len(b.items) < 6:\n        b.items.append(b.value)"
            ),
            "if len(boxes) > 0 and boxes[0].peer is not None:\n    boxes[0].peer.value += k",
            "if len(boxes) < 4:\n    boxes.append(Box(k))",
            "if len(boxes) > 1:\n    boxes[1].peer = boxes[0]",
        ]
        for statement in rng.sample(tails, rng.randint(2, 5)):
            for line in statement.split("\n"):
                w.put(line)
        w.put("return total + len(s) + len(d) + len(boxes)")
        w.depth -= 1
        w.put("")
        w.put("")

    def lent_function(self, w: _Writer, name: str) -> None:
        """A function Python calls natively with a list it only reads, lent
        for the call, that a nested function or a lambda reads too."""
        rng = self.crossing
        w.put("@ppy.native")
        w.put(f"def {name}(xs: list[int], k: int) -> int:")
        if rng.random() < 0.5:
            w.put("    def at(i: int) -> int:")
            w.put(f"        return xs[i] * {rng.randint(1, 5)} + k")
        else:
            w.put(f"    at: Callable[[int], int] = lambda i: xs[i] - k * {rng.randint(1, 5)}")
        w.put("    total = len(xs)")
        w.put("    for i in range(len(xs)):")
        w.put("        total = total * 3 + at(i)")
        w.put("    return total")
        w.put("")
        w.put("")

    def reader_function(self, w: _Writer, name: str) -> None:
        """A function Python calls natively with containers it only reads, which
        the generated wrapper reads in place (its lists in an arena, its strings
        borrowed), and a row of them it hands back."""
        rng = self.crossing
        w.put("@ppy.native")
        w.put(
            f"def {name}(g: list[list[int]], words: list[str], d: dict[str, int], "
            "s: set[int], k: int) -> list[int]:"
        )
        w.depth += 1
        w.put("total = k")
        w.put("best = g[0] if len(g) > 0 else []")
        menu = [
            "for row in g:\n    for x in row:\n        total += x * (k + 1)",
            (
                "for row in g:\n"
                f"    if len(row) > len(best) or sum(row) % {rng.randint(2, 5)} == 1:\n"
                "        best = row"
            ),
            "for w in words:\n    total += len(w)\n    if w in d:\n        total += d[w]",
            'for w in words:\n    for c in w:\n        if c in "ae\u00e9":\n            total += 1',
            "for key, v in d.items():\n    total += v * len(key)",
            "for x in s:\n    total += x % 5",
            f"for i in range({rng.randint(1, 9)}):\n    if i in s:\n        total += i",
        ]
        for statement in rng.sample(menu, rng.randint(2, 5)):
            for line in statement.split("\n"):
                w.put(line)
        w.put("if total % 3 == 0 or len(best) == 0:")
        w.put("    return [total]")
        w.put("return best")
        w.depth -= 1
        w.put("")
        w.put("")

    def boundary_part(self, w: _Writer) -> list[str]:
        """The functions with boundary writes, and what `main` does with them:
        arguments that share rows, a row both in the list and in the dict, the
        same object twice, and objects that point at each other."""
        rng = self.crossing
        name = self.name("bw")
        self.boundary_function(w, name)
        lent = self.name("lent")
        self.lent_function(w, lent)
        after = [
            f"row = [{', '.join(str(rng.randint(0, 9)) for _ in range(rng.randint(1, 4)))}]",
            rng.choice(("g = [row, [4, 5], row]", "g = [row] * 3", "g = [[1, 2], row, []]")),
            rng.choice(("d = {1: row, 2: [7]}", "d = {0: g[0], 3: []}", "d = {}")),
            f"s = {{{', '.join(str(rng.randint(0, 12)) for _ in range(rng.randint(1, 4)))}}}",
            "b1 = Box(1)",
            "b2 = Box(2)",
            "b1.peer = b2",
            rng.choice(("b2.peer = b1", "b2.peer = b2", "b2.peer = None")),
            rng.choice(("boxes = [b1, b2, b1]", "boxes = [b2]", "boxes = []")),
        ]
        after.append(f"print({lent}(row, {rng.randint(-3, 9)}), {lent}([], 1), row)")
        reader = self.name("rd")
        self.reader_function(w, reader)
        after.append(
            "words = "
            + rng.choice(('["ab", "é", ""]', '["x", "x", "naïve", "日本"]', "[]", '["bad\\ud800"]'))
        )
        after.append(rng.choice(('wd = {"ab": 2, "x": 5}', "wd = {}", 'wd = {"é": -1}')))
        for _ in range(rng.randint(1, 2)):
            after.append(f"got = {reader}(g, words, wd, s, {rng.randint(-3, 9)})")
            after.append("print(got, got is row, any(got is r for r in g))")
        for _ in range(rng.randint(1, 3)):
            after.append(f"print({name}(g, d, s, boxes, {rng.randint(-3, 9)}))")
            after.append(
                "print(g, sorted(d.items()), sorted(s), [b.value for b in boxes], "
                "b1.items, b2.items, row in g, len(g) > 1 and g[0] is g[len(g) - 1])"
            )
        return after

    # -- linked structures edited in place ---------------------------------------

    def tree_class(self, w: _Writer) -> list[str]:
        """A search tree with parent links, its fields unannotated, and methods
        Python calls natively that relink it: inserts, rotations, mirroring by
        tuple assignment, unlinking, and a new root grafted on top. Each walks
        it through a local alias of a field (`node = self.root`). The names
        of the methods it has, which `main` may call."""
        rng = self.structure
        w.put("class STree:")
        w.depth += 1
        w.put("def __init__(self):")
        w.put("    self.root = None")
        w.put("    self.size = 0")
        w.put("")
        w.put("@ppy.native")
        w.put("def insert(self, key: int) -> None:")
        w.put("    self.size += 1")
        w.put("    made = SNode(key)")
        w.put("    if self.root is None:")
        w.put("        self.root = made")
        w.put("        return")
        w.put("    node = self.root")
        w.put("    while True:")
        w.put("        if key < node.key:")
        w.put("            if node.left is None:")
        w.put("                node.left = made")
        w.put("                made.parent = node")
        w.put("                return")
        w.put("            node = node.left")
        w.put("        else:")
        w.put("            if node.right is None:")
        w.put("                node.right = made")
        w.put("                made.parent = node")
        w.put("                return")
        w.put("            node = node.right")
        w.put("")
        w.put("def find(self, key: int):")
        w.put("    node = self.root")
        w.put("    while node is not None and node.key != key:")
        w.put("        node = node.left if key < node.key else node.right")
        w.put("    return node")
        w.put("")
        methods: list[str] = []
        for side, other in (("left", "right"), ("right", "left")):
            if rng.random() < 0.8:
                methods.append(f"rotate_{side}")
                w.put("@ppy.native")
                w.put(f"def rotate_{side}(self, key: int) -> int:")
                w.put("    x = self.find(key)")
                w.put(f"    if x is None or x.{other} is None:")
                w.put("        return 0")
                w.put(f"    y = x.{other}")
                w.put(f"    x.{other} = y.{side}")
                w.put(f"    if y.{side} is not None:")
                w.put(f"        y.{side}.parent = x")
                w.put("    y.parent = x.parent")
                w.put("    if x.parent is None:")
                w.put("        self.root = y")
                w.put("    elif x is x.parent.left:")
                w.put("        x.parent.left = y")
                w.put("    else:")
                w.put("        x.parent.right = y")
                w.put(f"    y.{side} = x")
                w.put("    x.parent = y")
                w.put("    return 1")
                w.put("")
        if rng.random() < 0.6:
            methods.append("mirror")
            w.put("@ppy.native")
            w.put("def mirror(self, k: int) -> int:")
            w.put("    count = 0")
            w.put("    pending = [self.root]")
            w.put("    while len(pending) > 0:")
            w.put("        node = pending.pop()")
            w.put("        if node is None:")
            w.put("            continue")
            w.put("        node.left, node.right = node.right, node.left")
            if rng.random() < 0.5:
                w.put(f"        node.key = node.key * {rng.randint(-2, 3)} + k")
            w.put("        count += 1")
            w.put("        pending.append(node.left)")
            w.put("        pending.append(node.right)")
            w.put("    return count")
            w.put("")
        if rng.random() < 0.6:
            methods.append("pop_min")
            w.put("@ppy.native")
            w.put("def pop_min(self, k: int) -> int:")
            w.put("    node = self.root")
            w.put("    if node is None:")
            w.put("        return k")
            w.put("    while node.left is not None:")
            w.put("        node = node.left")
            w.put("    if node.parent is None:")
            w.put("        self.root = node.right")
            w.put("    else:")
            w.put("        node.parent.left = node.right")
            w.put("    if node.right is not None:")
            w.put("        node.right.parent = node.parent")
            w.put("    node.parent = None")
            w.put("    node.right = None")
            w.put("    self.size -= 1")
            w.put("    return node.key")
            w.put("")
        if rng.random() < 0.5:
            methods.append("graft")
            w.put("@ppy.native")
            w.put("def graft(self, k: int) -> int:")
            w.put("    made = SNode(k)")
            w.put(f"    made.{rng.choice(('left', 'right'))} = self.root")
            w.put("    if self.root is not None:")
            w.put("        self.root.parent = made")
            w.put("    self.root = made")
            w.put("    self.size += 1")
            w.put("    return self.size")
            w.put("")
        w.depth -= 1
        w.put("")
        return methods

    def dlist_class(self, w: _Writer) -> list[str]:
        """A doubly linked list, unannotated, edited in place natively: pushes
        at both ends, reversal by tuple assignment, rotation, and dropping
        nodes by key. Its links make cycles every crossing has to keep."""
        rng = self.structure
        w.put("class DList:")
        w.depth += 1
        w.put("def __init__(self):")
        w.put("    self.head = None")
        w.put("    self.tail = None")
        w.put("    self.count = 0")
        w.put("")
        w.put("@ppy.native")
        w.put("def push(self, key: int) -> None:")
        w.put("    made = DNode(key)")
        w.put("    made.prev = self.tail")
        w.put("    if self.tail is None:")
        w.put("        self.head = made")
        w.put("    else:")
        w.put("        self.tail.next = made")
        w.put("    self.tail = made")
        w.put("    self.count += 1")
        w.put("")
        methods: list[str] = []
        if rng.random() < 0.7:
            methods.append("push_front")
            w.put("@ppy.native")
            w.put("def push_front(self, key: int) -> int:")
            w.put("    made = DNode(key)")
            w.put("    made.next = self.head")
            w.put("    if self.head is None:")
            w.put("        self.tail = made")
            w.put("    else:")
            w.put("        self.head.prev = made")
            w.put("    self.head = made")
            w.put("    self.count += 1")
            w.put("    return self.count")
            w.put("")
        if rng.random() < 0.7:
            methods.append("reverse")
            w.put("@ppy.native")
            w.put("def reverse(self, k: int) -> int:")
            w.put("    node = self.head")
            w.put("    self.head, self.tail = self.tail, self.head")
            w.put("    while node is not None:")
            w.put("        node.prev, node.next = node.next, node.prev")
            if rng.random() < 0.5:
                w.put(f"        node.key += k * {rng.randint(1, 3)}")
            w.put("        node = node.prev")
            w.put("    return self.count")
            w.put("")
        if rng.random() < 0.6:
            methods.append("rotate")
            w.put("@ppy.native")
            w.put("def rotate(self, k: int) -> int:")
            w.put("    moved = 0")
            w.put(f"    while moved < k % {rng.randint(2, 5)} and self.head is not self.tail:")
            w.put("        first = self.head")
            w.put("        self.head = first.next")
            w.put("        self.head.prev = None")
            w.put("        first.next = None")
            w.put("        first.prev = self.tail")
            w.put("        self.tail.next = first")
            w.put("        self.tail = first")
            w.put("        moved += 1")
            w.put("    return moved")
            w.put("")
        if rng.random() < 0.6:
            modulus = rng.randint(2, 4)
            methods.append("drop")
            w.put("@ppy.native")
            w.put("def drop(self, k: int) -> int:")
            w.put("    node = self.head")
            w.put("    gone = 0")
            w.put("    while node is not None:")
            w.put("        after = node.next")
            w.put(f"        if (node.key + k) % {modulus} == 0:")
            w.put("            if node.prev is None:")
            w.put("                self.head = after")
            w.put("            else:")
            w.put("                node.prev.next = after")
            w.put("            if after is None:")
            w.put("                self.tail = node.prev")
            w.put("            else:")
            w.put("                after.prev = node.prev")
            w.put("            gone += 1")
            w.put("            self.count -= 1")
            w.put("        node = after")
            w.put("    return gone")
            w.put("")
        w.depth -= 1
        w.put("")
        return methods

    def structures_part(self, w: _Writer) -> list[str]:
        """The structure classes, and what `main` does with them: builds a tree
        and a list, holds on to nodes, edits them through the methods in a
        drawn order, and prints shapes, links, and identities after each."""
        rng = self.structure
        tree = self.tree_class(w)
        listed = self.dlist_class(w)
        w.put("")
        keys = rng.sample(range(-20, 40), rng.randint(3, 9))
        after = ["t = STree()"]
        after.extend(f"t.insert({key})" for key in keys)
        after.append("first = t.root")
        after.append("print(sshape(t.root), t.size, slinked(t))")
        for _ in range(rng.randint(2, 6)):
            if not tree:
                break
            method = rng.choice(tree)
            argument = rng.choice(keys) if method.startswith("rotate") else rng.randint(-3, 9)
            after.append(f"print(t.{method}({argument}), sshape(t.root), t.size, slinked(t))")
        after.append(
            "print(first is t.root, first.parent is None, t.find(" + str(keys[0]) + ") is not None)"
        )
        after.append("d = DList()")
        after.extend(f"d.push({rng.randint(-9, 9)})" for _ in range(rng.randint(0, 6)))
        after.append("ends = (d.head, d.tail)")
        after.append("print(dkeys(d), d.count)")
        for _ in range(rng.randint(2, 6)):
            if not listed:
                break
            method = rng.choice(listed)
            after.append(f"print(d.{method}({rng.randint(-3, 9)}), dkeys(d), d.count)")
        after.append("print(ends[0] is d.head, ends[1] is d.tail, ends[0] is d.tail)")
        return after


#: The first line of a program whose functions have no annotations; it runs
#: without strict mode.
UNANNOTATED_MARK = "# fuzz: unannotated"
#: The first line of a program whose structure classes have no annotations on
#: their fields; it runs without strict mode too (it starts with the mark above).
STRUCTURES_MARK = "# fuzz: unannotated fields"


def _unannotated(signature: str) -> str:
    """A parameter list with its annotations taken out, defaults kept."""
    tree = ast.parse(f"def f({signature}): pass")
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        argument.annotation = None
    return ast.unparse(function.args)


def generate_program(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    seed: int,
    prints: bool = False,
    state: bool = False,
    stdlib: bool = False,
    calls: bool = False,
    unannotated: bool = False,
    *,
    boundary: bool = False,
    structures: bool = False,
) -> str:
    """The program for `seed`: identical on every machine and every run. With
    `prints`, functions print between checks that may fall back. With `state`,
    it also reads and writes module globals and walks objects Python made,
    which only the paths with a Python boundary run (`STATE_PATHS`). With
    `stdlib`, functions also draw seeded random numbers and call `math`,
    `heapq`, `bisect`, `itertools`, `functools`, `operator`, and
    `collections`' containers, and a `random.Random` of their own. With
    `calls`, functions take defaults and keyword-only parameters, and `main`
    calls them by keyword and leaves defaults out. With `unannotated`, the
    functions have no annotations and the program runs without strict mode,
    which infers their types from `main`'s calls; Python then calls each
    with arguments of other types, which the native entry must hand to the
    Python body. With `boundary`, a function Python calls natively writes
    through lists of lists, a dict of lists, a set, and objects that share
    rows and point at each other (`STATE_PATHS` too). With `structures`,
    classes with unannotated fields (a search tree with parent links, a
    doubly linked list) are edited in place by methods Python calls natively:
    rotations, unlinking, tuple-assigned swaps, new nodes linked in
    (`STATE_PATHS`, without strict mode)."""
    return _Generator(
        seed, prints, state, stdlib, calls, unannotated, boundary, structures
    ).program()


def printed_twice(results: dict[str, Result]) -> list[Mismatch]:
    """Every path on which a tagged line (`print_statement`) shows up more often
    than under CPython: output a fallback printed a second time."""
    expected = results["python"]
    tags = re.compile(r"<p\d+>")

    def counts(text: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for tag in tags.findall(text):
            found[tag] = found.get(tag, 0) + 1
        return found

    wanted = counts(expected.stdout)
    found: list[Mismatch] = []
    for path, result in results.items():
        if path == "python":
            continue
        extra = {t: n for t, n in counts(result.stdout).items() if n > wanted.get(t, 0)}
        if extra:
            found.append(Mismatch(path, f"printed twice: {sorted(extra)}", expected, result))
    return found


# -- running ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Result:
    """What one path did: its output, its exit status, its last stderr line."""

    path: str
    stdout: str
    status: int
    last_error: str
    #: The whole of stderr, for a report; not compared.
    stderr: str = ""

    @property
    def refused(self) -> bool:
        """The toolchain could not build it at all: a finding, not a result."""
        return self.status == -2


@dataclass(frozen=True, slots=True)
class Mismatch:
    path: str
    reason: str
    expected: Result
    found: Result


def _capped(command: list[str], memory: str) -> list[str]:
    """`command` in a user scope with a memory cap, where systemd can make one."""
    if shutil.which("systemd-run") and os.environ.get("PPY_FUZZ_NO_SCOPE") != "1":
        return [
            "systemd-run", "--user", "--scope", "-q",
            "-p", f"MemoryMax={memory}", "-p", "MemorySwapMax=0", *command,
        ]  # fmt: skip
    return command


def _execute(
    command: list[str], cwd: Path, timeout: float, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """Run `command` in a process group of its own; at `timeout`, kill the group.

    `subprocess.run(timeout=...)` kills only the process it started and then
    waits for its pipes to close, which a child `ppy run` spawned keeps open:
    a program that loops natively held a run for a day. Killing the whole
    group ends every process the command made, and the pipes with them.
    """
    with subprocess.Popen(
        _capped(command, "2G"),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    ) as process:
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=10)
            return TIMED_OUT, "", "timed out"
        return process.returncode, out, err


def _kill_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def _clean(stdout: str) -> str:
    """What a path printed, less `ppy run`'s own notes about compiling."""
    return "\n".join(line for line in stdout.splitlines() if not line.startswith("compiling "))


def _compilers() -> dict[str, str | None]:
    """Clang first: GCC 13's `-O1` use-after-scope poisoning reports a variable
    of a loop nest read where the source never reads it (seed 810), which the
    same C under clang, and under GCC at `-O0` and `-O2`, does not."""
    return {
        "c": shutil.which("clang") or shutil.which("cc") or shutil.which("gcc"),
        "cpp": shutil.which("clang++") or shutil.which("c++") or shutil.which("g++"),
    }


def run_program(
    source: str, paths: tuple[str, ...] = ALL_PATHS, timeout: float = 120.0
) -> dict[str, Result]:
    """Run `source` on each of `paths`, one after another, in a scratch project."""
    results: dict[str, Result] = {}
    python = sys.executable
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    compilers = _compilers()
    with tempfile.TemporaryDirectory(prefix="ppy-fuzz-") as scratch:
        root = Path(scratch)
        strict = "false" if source.startswith(UNANNOTATED_MARK) else "true"
        (root / "pyproject.toml").write_text(f"[tool.ppy]\nstrict = {strict}\n", encoding="utf-8")
        (root / "prog.ppy").write_text(source, encoding="utf-8")
        for path in paths:
            results[path] = _run_path(path, root, python, env, compilers, timeout)
    return results


def _run_path(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    path: str,
    root: Path,
    python: str,
    env: dict[str, str],
    compilers: dict[str, str | None],
    timeout: float,
) -> Result:
    if path == "python":
        status, out, err = _execute([python, "prog.ppy"], root, timeout, env)
    elif path == "ppy":
        status, out, err = _execute([python, "-m", "ppy_compiler", "prog.ppy"], root, timeout, env)
    elif path == "run":
        command = [python, "-m", "ppy_compiler", "run", "prog.ppy"]
        status, out, err = _execute(command, root, timeout, env)
    elif path == "standalone":
        command = [python, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "sa"]
        status, out, err = _execute(command, root, timeout, env)
        if status != 0:
            return Result(path, "", -2, _last_line(err or out), err or out)
        status, out, err = _execute([str(root / "sa" / "prog")], root, timeout, env)
    else:
        compiler = compilers.get(path)
        if compiler is None:
            return Result(path, "", -2, f"no {path} compiler", "")
        emitted = f"prog.{path}"
        command = [python, "-m", "ppy_compiler", "emit", path, "--standalone", "prog.ppy"]
        status, out, err = _execute([*command, "-o", emitted], root, timeout, env)
        if status != 0:
            return Result(path, "", -2, _last_line(err or out), err or out)
        standard = "-std=c11" if path == "c" else "-std=c++17"
        binary = f"prog_{path}"
        build = [
            compiler, standard, "-g", "-O1", "-fsanitize=address,undefined",
            "-fno-sanitize-recover=undefined", emitted, "-lm", "-o", binary,
        ]  # fmt: skip
        status, out, err = _execute(build, root, timeout, env)
        if status != 0:
            return Result(path, "", -2, _last_line(err), err)
        # What the program does, first; a program that stops on an error
        # leaves what it held to the exit, which LeakSanitizer would call a
        # leak. Only a clean exit is then held to freeing everything.
        quiet = _with_asan(env, "detect_leaks=0")
        status, out, err = _execute([str(root / binary)], root, timeout, quiet)
        if status == 0:
            checked = _with_asan(env, "detect_leaks=1")
            leaked, _out, report = _execute([str(root / binary)], root, timeout, checked)
            if leaked != 0:
                return Result(path, _clean(out), leaked, _last_line(report), report)
    return Result(path, _clean(out), status, _last_line(err), err)


def _with_asan(env: dict[str, str], options: str) -> dict[str, str]:
    """`env` with AddressSanitizer's options set to `options`."""
    changed = dict(env)
    changed["ASAN_OPTIONS"] = options
    return changed


def _overflow_allowed(expected: Result, found: Result) -> bool:
    """A native-only path stopping where CPython computed an integer past a
    word, having printed what CPython printed up to there."""
    if found.status != 1 or found.last_error != OVERFLOW_64:
        return False
    # Up to the character: a print with `end=""` leaves a line open.
    return expected.stdout.startswith(found.stdout)


def compare(results: dict[str, Result]) -> list[Mismatch]:
    """Every way a path differs from CPython, less the one allowed difference."""
    expected = results["python"]
    found: list[Mismatch] = []
    if expected.status == TIMED_OUT:
        # The reference itself ran out of time: nothing to hold the paths to.
        return found
    for path, result in results.items():
        if path == "python":
            continue
        if result.status == TIMED_OUT or (result.refused and result.last_error == "timed out"):
            found.append(Mismatch(path, "timed out", expected, result))
            continue
        if result.refused:
            found.append(Mismatch(path, "did not build", expected, result))
            continue
        if path in _NATIVE_ONLY and _overflow_allowed(expected, result):
            continue
        if result.stdout.rstrip("\n") != expected.stdout.rstrip("\n"):
            found.append(Mismatch(path, "stdout differs", expected, result))
        elif (result.status == 0) != (expected.status == 0):
            found.append(Mismatch(path, "exit status differs", expected, result))
        elif expected.status < 0 and result.status == expected.status:
            # Both killed by the same signal (the memory cap, say): neither
            # wrote a last line of its own to compare.
            continue
        elif expected.status != 0 and result.last_error != expected.last_error:
            found.append(Mismatch(path, "the error differs", expected, result))
    return found


# -- minimizing ---------------------------------------------------------------


def _statements(source: str) -> list[tuple[int, int]]:
    """Line spans (start, end, 1-based inclusive) of every statement inside a
    function, innermost last, so removing one leaves valid Python."""
    tree = ast.parse(source)
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in {"__init__", "bump", "biggest"}:
            spans.extend(
                (child.lineno, child.end_lineno or child.lineno)
                for child in ast.walk(node)
                if child is not node and isinstance(child, ast.stmt)
            )
    spans.extend(
        (node.lineno, node.end_lineno or node.lineno)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("fn")
    )
    return sorted(set(spans), key=lambda span: (span[1] - span[0], span[0]), reverse=True)


def _without(source: str, span: tuple[int, int]) -> str | None:
    lines = source.splitlines()
    start, end = span
    kept = lines[: start - 1] + lines[end:]
    candidate = "\n".join(kept) + "\n"
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        # A block left empty: give it a `pass`.
        indent = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
        kept = [*lines[: start - 1], f"{indent}pass", *lines[end:]]
        candidate = "\n".join(kept) + "\n"
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            return None
    del tree
    return candidate


def minimize(source: str, still_fails: Callable[[str], bool], attempts: int = 200) -> str:
    """Delete statements, largest first, while `still_fails` holds, starting
    over after each deletion that kept the failure, until none does or
    `attempts` runs of the predicate are spent."""
    current = source
    tried: set[str] = set()
    while attempts > 0:
        for span in _statements(current):
            candidate = _without(current, span)
            if candidate is None or candidate == current or candidate in tried:
                continue
            tried.add(candidate)
            attempts -= 1
            if still_fails(candidate):
                current = candidate
                break
            if attempts <= 0:
                break
        else:
            break
    return current
