"""Shapes the corpus kept in Python, lowered natively.

Each is a small form ordinary code takes, which TheAlgorithms/Python showed
keeping whole functions in Python:

- an `if`/`elif`/`else` whose every side returns or raises, which leaves no
  way to the end of the function; and a function that does fall off its end,
  which falls back so Python returns the `None` it gives;
- a list parameter tested for truth (`if not xs:`), compared with `[]`,
  unpacked (`a, b, c = xs`), returned, or sliced;
- a module-level string constant (`LETTERS = "ABC..."`) read in a function;
- a tuple assignment of collections (`holes, seen = [0] * n, []`), a swap
  of two lists, and a bare `list` annotation on an empty display;
- `*args` of numbers, called from Python and from native code;
- a nested function whose enclosing function stays in Python, handed the
  variables it shares as their cells hold them when Python calls it;
- standard-library calls whose results the checker now knows (`os.path`,
  `timeit`, `__file__`).

Each program is held to CPython under `ppy`, `ppy run`, and, where it has a
`main`, a standalone binary and emitted C and C++ under AddressSanitizer,
with `ppy explain` confirming the functions went native.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

ENDS = """
def sign(n: int) -> int:
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0


def power(a: int, b: int) -> int:
    if b == 0:
        return 1
    half = power(a, b // 2)
    if b % 2 == 0:
        return half * half
    else:
        return a * half * half


def checked(n: int) -> int:
    if n == 1:
        return 2
    elif n < 1:
        raise ValueError(f"n={n} has to be > 0")
    else:
        if n > 50:
            return -1
        else:
            return checked(n - 1) * 2


def main() -> None:
    total = 0
    for i in range(-3, 4):
        total += sign(i) * 10 + power(3, i + 3)
    print(total, power(2, 10), checked(5))
    try:
        checked(0)
    except ValueError as error:
        print("ValueError", error)


main()
"""

LISTS = """
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYMBOLS = (
    r\"\"\" !"#$%&'()*+,-./0123456789\"\"\"
)


def first_or(xs: list[int], default: int) -> int:
    if not xs:
        return default
    return xs[0]


def empty(xs: list[int]) -> bool:
    if xs == []:
        return True
    return [] == xs


def solve(eq1: list[int], eq2: list[int]) -> float:
    a1, b1, c1 = eq1
    a2, b2, c2 = eq2
    return (c1 * b2 - b1 * c2) / (a1 * b2 - b1 * a2)


def swap_halves(s: str) -> str:
    a, b = s.split(",")
    return b + a


def product(xs: list[float]) -> float:
    xs.append(2.0)
    p, q = xs
    return p * q


def middle(lst: list[int]) -> list[int]:
    m = len(lst) // 2
    return lst[m - 1 : m + 2]


def search(a_list: list[int], item: int) -> bool:
    if len(a_list) == 0:
        return False
    midpoint = len(a_list) // 2
    if a_list[midpoint] == item:
        return True
    if item < a_list[midpoint]:
        return search(a_list[:midpoint], item)
    return search(a_list[midpoint + 1 :], item)


def same(xs: list[int], k: int) -> list[int]:
    if k == 0:
        return xs
    return [x + k for x in xs]


def shift(message: str, key: int) -> str:
    out = ""
    for symbol in message:
        found = LETTERS.find(symbol.upper())
        if found == -1:
            out += symbol
        else:
            out += LETTERS[(found + key) % len(LETTERS)]
    return out + str(len(SYMBOLS)) + LETTERS[:3]


def holes(n: int) -> int:
    counts, seen = [0] * n, [1] * (n + 1)
    counts[0] = 5
    return sum(counts) + len(seen)


def swapped(n: int) -> int:
    a = [1, 2]
    b = [3]
    for _ in range(n):
        a, b = b, a
    return a[0] * 10 + len(b)


def stacks(n: int) -> list[int]:
    s1: list[int]
    s2: list[int]
    s1, s2 = [], []
    for i in range(n):
        s1.append(i)
        s2.append(i * i)
    return s1 + s2


def main() -> None:
    nums = [3, 4]
    none: list[int] = []
    print(first_or(nums, -1), first_or(none, -1), empty(none), empty(nums))
    print(solve([1, 2, 3], [2, 5, 1]))
    short = [1, 2]
    long = [1, 2, 3, 4]
    try:
        solve(short, [1, 2, 3])
    except ValueError as error:
        print("ValueError", error)
    try:
        solve(long, [1, 2, 3])
    except ValueError as error:
        print("ValueError", error)
    print(swap_halves("ab,cd"), product([1.5]))
    try:
        print(swap_halves("a,b,c"))
    except ValueError as error:
        print("ValueError", error)
    nothing: list[float] = []
    try:
        print(product(nothing))
    except ValueError as error:
        print("ValueError", error)
    sorted_items = [1, 3, 5, 7, 9]
    print(middle(sorted_items), search(sorted_items, 7), search(sorted_items, 4))
    print(same(nums, 0), same(nums, 2))
    print(shift("Hello, World", 3), holes(4), swapped(3), swapped(2), stacks(3))


main()
"""

VARIADIC = """
def total(scale: int, *xs: int) -> int:
    t = 0
    for x in xs:
        t += x * scale
    return t + len(xs)


def positive(*values: float) -> bool:
    return len(values) > 0 and all(value > 0.0 for value in values)


def caller(n: int) -> int:
    acc = 0
    for i in range(n):
        acc += total(2, i, i + 1, i + 2) + total(3)
    return acc


def main() -> None:
    print(total(2, 1, 2, 3), total(5), caller(10))
    print(positive(2.0, 4.5), positive(-1.0, 2.0), positive())


main()
"""

PROGRAMS = {
    "ends": (ENDS, ["sign", "power", "checked"]),
    "lists": (
        LISTS,
        [
            "first_or",
            "empty",
            "solve",
            "swap_halves",
            "product",
            "middle",
            "search",
            "same",
            "shift",
            "holes",
            "stacks",
        ],
    ),
    "variadic": (VARIADIC, ["total", "positive", "caller"]),
}


def _write(tmp_path: Path, source: str, *, strict: bool = True) -> Path:
    setting = "true" if strict else "false"
    (tmp_path / "pyproject.toml").write_text(f"[tool.ppy]\nstrict = {setting}\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )


def _output(done: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(line for line in done.stdout.splitlines() if not line.startswith("compiling"))


def _expected(tmp_path: Path, source: str, *, strict: bool = True) -> str:
    _write(tmp_path, source, strict=strict)
    done = _run(tmp_path, "prog.ppy")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _agrees(tmp_path: Path, expected: str) -> None:
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args


def _native(tmp_path: Path, functions: list[str]) -> None:
    for function in functions:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    source, natives = PROGRAMS[name]
    _agrees(tmp_path, _expected(tmp_path, source))
    _native(tmp_path, natives)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    expected = _expected(tmp_path, PROGRAMS[name][0])
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", "-x", "c", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@requires_llvm
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_emitted_source_frees_everything_once(
    tmp_path: Path, name: str, language: str, unsafe: bool
):
    expected = _expected(tmp_path, PROGRAMS[name][0])
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    emitted = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", *flags, "prog.ppy",
        "-o", emitted.name,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    subprocess.run(
        [compiler, standard, "-g", "-O1", "-fsanitize=address,undefined", str(emitted), "-lm",
         "-o", str(binary)],
        check=True,
    )  # fmt: skip
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


FALLS_OFF = """
def half(n: int) -> int:
    if n % 2 == 0:
        return n // 2
    elif n < 0:
        raise ValueError("negative and odd")


def first_square(n: int) -> int:
    for i in range(n):
        if i * i > n:
            return i


def tally(xs: list[int]) -> int:
    total = 0
    for x in xs:
        if x % 2 == 0:
            total += half(x)
    return total


print(half(10), half(7), first_square(10), first_square(0), tally([2, 4, 6, 7]))
try:
    half(-3)
except ValueError as error:
    print("ValueError", error)
try:
    print(tally([3]) + 1)
    print(half(3) + 1)
except TypeError as error:
    print("TypeError", error)
"""


@requires_llvm
@requires_cc
def test_a_function_that_falls_off_its_end_returns_none_through_python(tmp_path: Path):
    """CPython returns None where a function falls off its end, which an `int`
    result cannot hold: the native call falls back there, and Python runs it."""
    _agrees(tmp_path, _expected(tmp_path, FALLS_OFF))
    _native(tmp_path, ["half", "first_square", "tally"])


@requires_standalone
def test_a_standalone_build_refuses_a_function_that_falls_off(tmp_path: Path):
    _write(
        tmp_path,
        FALLS_OFF.split("\nprint(", maxsplit=1)[0] + "\n\ndef main() -> None:\n    half(3)\n",
    )
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode != 0
    assert "fall off the end" in built.stderr


COUNTER = """
COUNTER = 0
RATE = 0.5


def bump() -> int:
    global COUNTER
    COUNTER += 1
    return COUNTER


def bumped() -> int:
    global COUNTER
    COUNTER = COUNTER + 1
    x = COUNTER
    return x


def scaled(x: float) -> float:
    return x * RATE


print(bump(), bump(), bumped(), bump(), scaled(3.0))
RATE = 2.0
print(scaled(3.0))
"""


@requires_llvm
@requires_cc
def test_a_global_bound_again_keeps_no_value_from_its_first_binding(tmp_path: Path):
    """`COUNTER += 1` under `global` reads what the module holds when the
    function runs, not the 0 it was first bound to."""
    _agrees(tmp_path, _expected(tmp_path, COUNTER))


BARE = """
def joined(s: str) -> str:
    out: list = []
    for ch in s:
        out.append(ch.upper())
    return "-".join(out)


def squares(n: int) -> int:
    seen: dict = {}
    for i in range(n):
        seen[i] = i * i
    return sum(seen.values())


print(joined("abc"), squares(5))
"""


@requires_llvm
@requires_cc
def test_a_bare_container_annotation_on_an_empty_display_is_inferred(tmp_path: Path):
    """`out: list = []` under `--no-strict` says what `out = []` does, and the
    appends after it say what the list holds."""
    _agrees(tmp_path, _expected(tmp_path, BARE, strict=False))
    _native(tmp_path, ["joined", "squares"])


CELLS = """
import sys


def outer(n: int, k: int) -> int:
    scale = k * 2
    label = "xy"
    weights = [1, 2, 3]

    def work(m: int) -> int:
        acc = 0
        for i in range(m):
            acc += (i * scale) % 7 + len(label) + weights[i % len(weights)]
        return acc

    first = work(n)
    scale = 100
    return first + work(n) + len(sys.argv) * 0


def filled(n: int) -> list[int]:
    seen = [0] * n

    def sweep(k: int) -> int:
        t = 0
        for i in range(len(seen)):
            seen[i] = seen[i] + i * k
            t += seen[i]
        return t

    print(sweep(2), sweep(3), len(sys.argv) * 0)
    return seen[:5]


def grid_of(n: int) -> list[list[int]]:
    grid = [[0] * 3 for _ in range(3)]

    def fill(k: int) -> int:
        t = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                grid[i][j] = grid[i][j] + i * j * k + 1
                t += grid[i][j]
        return t

    print(fill(n), fill(n + 1), len(sys.argv) * 0)
    return grid


def late() -> int:
    def use(m: int) -> int:
        t = 0
        for i in range(m):
            t += i * v
        return t

    try:
        use(3)
    except NameError as error:
        print("NameError", error)
    v = 2
    return use(50) + len(sys.argv) * 0


print(outer(1000, 3), outer(10, 1), filled(100), grid_of(2), late())
"""


@requires_llvm
@requires_cc
def test_a_nested_function_is_handed_the_cells_it_shares(tmp_path: Path):
    """The function around each nested one stays in Python (it reads
    `sys.argv`); the nested one goes native, and Python hands it each variable
    it shares as the cell holds it at the call: a rebinding between calls is
    seen, a list it writes is written back, a list of lists written through
    an item too, and an empty cell is Python's `NameError`."""
    _agrees(tmp_path, _expected(tmp_path, CELLS))
    _native(
        tmp_path,
        [
            "outer.<locals>.work",
            "filled.<locals>.sweep",
            "grid_of.<locals>.fill",
            "late.<locals>.use",
        ],
    )


LIBRARY = """
import os
import os.path
from timeit import timeit


def exists(name: str) -> bool:
    return os.path.exists(f"{name}.missing")


def here() -> str:
    return os.path.basename(os.path.dirname(os.path.realpath(__file__)))


def joined(n: int) -> str:
    out = ""
    for i in range(n):
        out = os.path.join(out, str(i))
    return out


def timed(n: int) -> bool:
    return timeit(stmt="1 + 1", number=n) >= 0.0


print(exists("nothing"), here() == os.path.basename(os.getcwd()), joined(3), timed(10))
"""


@requires_llvm
@requires_cc
def test_library_calls_the_checker_knows_go_through_python(tmp_path: Path):
    _agrees(tmp_path, _expected(tmp_path, LIBRARY))
    _native(tmp_path, ["exists", "here", "joined", "timed"])
