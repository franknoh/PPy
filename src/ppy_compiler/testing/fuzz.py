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
    "TIMED_OUT",
    "Mismatch",
    "Result",
    "compare",
    "generate_program",
    "minimize",
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
    def __init__(self, seed: int, state: bool = False) -> None:
        self.rng = random.Random(seed)
        self.fresh = 0
        #: Whether the program also has module state and objects crossing
        #: `ppy run`'s boundary, drawn from a sequence of their own.
        self.with_state = state
        self.state = random.Random(seed ^ 0x5EED)

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
        if roll < 0.95:
            return f"int({self.float_expr(scope, depth + 1)})"
        return f"({left} if {self.bool_expr(scope, depth + 1)} else {self.int_expr(scope, 3)})"

    def float_expr(self, scope: _Scope, depth: int = 0) -> str:
        rng = self.rng
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
        if roll < 0.85:
            return f"({self.str_expr(scope, 2)} in {self.str_expr(scope, 2)})"
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
            self.lifted_statement(w, scope)
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
        w.lines.extend(_PRELUDE.splitlines())
        if self.with_state:
            w.lines.extend(_STATE_PRELUDE.splitlines())
            self.state_globals(w)
        calls: list[str] = []
        for _ in range(self.rng.randint(3, 6)):
            name = self.name("fn")
            kinds, _ret = self.function(w, name)
            calls.extend(
                f"{name}({', '.join(self.argument(k) for k in kinds)})"
                for _ in range(self.rng.randint(1, 3))
            )
        after = self.state_part(w) if self.with_state else []
        w.put("def main() -> None:")
        for call in calls:
            w.put(f"    print({call})")
        for line in after:
            w.put(f"    {line}")
        w.put("")
        w.put("")
        w.put("main()")
        return "\n".join(w.lines) + "\n"

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


def generate_program(seed: int, state: bool = False) -> str:
    """The program for `seed`: identical on every machine and every run. With
    `state`, it also reads and writes module globals and walks objects Python
    made, which only the paths with a Python boundary run (`STATE_PATHS`)."""
    return _Generator(seed, state).program()


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
        (root / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
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
    printed = found.stdout.splitlines()
    return expected.stdout.splitlines()[: len(printed)] == printed


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
