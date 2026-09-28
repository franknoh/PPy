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
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ALL_PATHS",
    "OVERFLOW_64",
    "Mismatch",
    "Result",
    "compare",
    "generate_program",
    "minimize",
    "run_program",
]

#: Every path a program can take, the reference first.
ALL_PATHS = ("python", "ppy", "run", "standalone", "c", "cpp")

#: What native code says where CPython would compute an integer no word holds.
OVERFLOW_64 = "OverflowError: the result does not fit in a 64-bit integer"

#: The paths that are native code with no Python to fall back to.
_NATIVE_ONLY = frozenset({"standalone", "c", "cpp"})

_PRELUDE = """\
import math
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
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.fresh = 0

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
        elif roll < 0.64:
            self.vec_statements(w, scope)
        elif roll < 0.7:
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
        elif roll < 0.94:
            w.put(f"if {self.bool_expr(scope)}:")
            w.depth += 1
            w.put(f"return {self.value(ret, scope)}")
            w.depth -= 1
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

    def program(self) -> str:
        w = _Writer()
        w.lines.extend(_PRELUDE.splitlines())
        calls: list[str] = []
        for _ in range(self.rng.randint(3, 6)):
            name = self.name("fn")
            kinds, _ret = self.function(w, name)
            calls.extend(
                f"{name}({', '.join(self.argument(k) for k in kinds)})"
                for _ in range(self.rng.randint(1, 3))
            )
        w.put("def main() -> None:")
        for call in calls:
            w.put(f"    print({call})")
        w.put("")
        w.put("")
        w.put("main()")
        return "\n".join(w.lines) + "\n"


def generate_program(seed: int) -> str:
    """The program for `seed`: identical on every machine and every run."""
    return _Generator(seed).program()


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
    try:
        done = subprocess.run(
            _capped(command, "2G"),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"
    return done.returncode, done.stdout, done.stderr


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def _clean(stdout: str) -> str:
    """What a path printed, less `ppy run`'s own notes about compiling."""
    return "\n".join(line for line in stdout.splitlines() if not line.startswith("compiling "))


def _compilers() -> dict[str, str | None]:
    return {
        "c": shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"),
        "cpp": shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"),
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
        quiet = {**env, "ASAN_OPTIONS": "detect_leaks=0"}
        status, out, err = _execute([str(root / binary)], root, timeout, quiet)
        if status == 0:
            checked = {**env, "ASAN_OPTIONS": "detect_leaks=1"}
            leaked, _out, report = _execute([str(root / binary)], root, timeout, checked)
            if leaked != 0:
                return Result(path, _clean(out), leaked, _last_line(report), report)
    return Result(path, _clean(out), status, _last_line(err), err)


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
    for path, result in results.items():
        if path == "python":
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
