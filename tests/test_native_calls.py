"""Calls: keywords and defaults, results back from Python, and nested functions.

A call between native functions binds its keywords and defaults at compile
time, as Python would bind them; a Python caller's keywords are bound at the
boundary. A call from native code into Python takes its result back: a
callee that changes nothing is run again unseen if the result is not what
the checker said; any other callee's result is taken only where it cannot be
anything else. A nested function that shares nothing with the function
around it gets a native entry of its own. Each program is held to CPython
under `ppy`, `ppy run`, a standalone binary where it is fully native, and
emitted C and C++ under AddressSanitizer with leak detection.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import _collect
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status
from ppy_compiler.backend.llvm.lowering import Unsupported
from ppy_compiler.backend.llvm.runtime import bind
from ppy_compiler.lowering.calls import bind_arguments

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

KEYWORDS = """
class Acc:
    def __init__(self, start: int = 0, step: int = 1) -> None:
        self.total = start
        self.step = step

    def add(self, x: int, times: int = 1, *, bonus: int = 0) -> int:
        self.total += x * times * self.step + bonus
        return self.total


def scale(x: int, factor: int = 3, *, offset: int = 0) -> int:
    return x * factor + offset


def label(n: int, prefix: str = "n", sep: str = "") -> str:
    return prefix + sep + str(n)


def ratio(a: float, b: float = 2.0, neg: bool = False) -> float:
    r = a / b
    return -r if neg else r


def use(n: int) -> int:
    total = 0
    acc = Acc(step=2)
    other = Acc(5)
    for i in range(n):
        total += scale(i) + scale(i, 2) + scale(i, factor=4) + scale(x=i, offset=1)
        total += scale(offset=-1, x=i, factor=2)
        total += len(label(i)) + len(label(i, sep="-")) + len(label(prefix="p", n=i))
        total += acc.add(i) + acc.add(i, bonus=3) + other.add(x=i, times=2)
        total += int(ratio(float(i)) + ratio(float(i), neg=True) + ratio(b=4.0, a=float(i)))
    return total


def main() -> None:
    print(use(10), scale(2, offset=5), label(3, sep=":"), ratio(3.0, neg=True))
    a = Acc(step=3)
    print(scale(4), scale(x=1, factor=2), label(n=7), a.add(2, bonus=1))


main()
"""

#: Nested functions under a native function, one shadowing a module function.
SHADOWED = """
def helper(x: int) -> int:
    return x + 1000


def outer(n: int) -> list[int]:
    def helper(x: int) -> int:
        if x <= 1:
            return x
        return helper(x - 1) + helper(x - 2)

    def square(x: int) -> int:
        return x * x

    out: list[int] = []
    for i in range(n):
        out.append(helper(i) + square(i))
    return out


def other(n: int) -> int:
    def helper(x: int) -> int:
        return x * 3

    return helper(n) + 1


def top(n: int) -> int:
    return helper(n)


def main() -> None:
    print(outer(15), other(4), top(1))


main()
"""

PROGRAMS = {
    "keywords": (KEYWORDS, ["scale", "label", "ratio", "use", "Acc.add", "Acc.__init__"]),
    "shadowed": (SHADOWED, ["outer", "other", "top"]),
}

#: Under `ppy run` only: a function around nested ones stays in Python
#: (`**options`), and native code calls into Python.
PYTHON_AROUND = """
import math


def cost(a: int, b: int) -> int:
    return -1


def solve(rows: list[tuple[int, int]], **options: int) -> int:
    scale = options.get("scale", 1)

    def cost(a: int, b: int) -> int:
        total = 0
        for i in range(a):
            total += (i * b) % 7
        return total

    def fib(n: int) -> int:
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    def scaled(n: int) -> int:
        return n * scale

    s = 0
    for a, b in rows:
        s += cost(a, b) + fib(a % 20) + scaled(a)
    return s


def stays(n: int) -> int:
    return len({i % 7 for i in range(n)})


def big(n: int) -> int:
    return len({i for i in range(n)}) * 10**n


def mixed(flag: bool) -> float:
    return len({1, 2}) if flag else 2.5


def shout(text: str) -> str:
    print("shouting", text)
    return text.upper() + "!"


def use(n: int) -> int:
    total = 0
    for i in range(n):
        q, r = divmod(i * 7, 3)
        total += q + r + stays(i) + round(i / 3) + int(math.floor(i / 2.0))
        total += len(str(i)) + abs(-i)
    return total


def use_big(n: int) -> int:
    return big(n) + 1


def use_mixed(flag: bool) -> float:
    return mixed(flag) * 2


def use_shout(n: int) -> int:
    s = 0
    for i in range(n):
        s += len(shout(str(i))) + len(format(i, "x"))
    return s


def keywords(n: int) -> float:
    return round(n / 7, ndigits=2) + float(int("12", base=10))


def bound(x: int, factor: int = 3, *, offset: int = 0) -> int:
    total = 0
    for i in range(x):
        total += i * factor + offset
    return total


print(solve([(i, i + 1) for i in range(30)], scale=2), solve([(3, 4)]), cost(1, 2))
print(use(30))
print(use_big(3), use_big(30))
print(use_mixed(True), use_mixed(False))
print(use_shout(3))
print(keywords(10))
print(bound(10), bound(10, 2), bound(x=10, offset=1), bound(10, factor=5, offset=-2))
"""

PYTHON_AROUND_NATIVE = [
    "solve.<locals>.cost",
    "solve.<locals>.fib",
    "use",
    "use_big",
    "use_mixed",
    "use_shout",
    "keywords",
    "bound",
]


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
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


def _expected(tmp_path: Path, source: str) -> str:
    _write(tmp_path, source)
    done = _run(tmp_path, "prog.ppy")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _agrees_and_goes_native(tmp_path: Path, source: str, natives: list[str]) -> None:
    expected = _expected(tmp_path, source)
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    source, natives = PROGRAMS[name]
    _agrees_and_goes_native(tmp_path, source, natives)


@requires_llvm
@requires_cc
def test_python_around_and_calls_into_python(tmp_path: Path):
    _agrees_and_goes_native(tmp_path, PYTHON_AROUND, PYTHON_AROUND_NATIVE)
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.solve.<locals>.scaled")
    assert "shares `scale` with the function around it" in explained.stdout, explained.stdout


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
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    expected = _expected(tmp_path, PROGRAMS[name][0])
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


# -- the boundary binds a Python caller's keywords ---------------------------------


@requires_llvm
def test_a_python_caller_s_keywords_and_defaults_reach_the_native_entry(write, analyze):
    path = write(
        "kw.ppy",
        """
        def scale(x: int, factor: int = 3, *, offset: int = 0) -> int:
            return x * factor + offset
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["kw"]
    engine = JitEngine(opt_level=2).open()
    engine.add(module.ir)
    engine.finalize()

    def scale(x: int, factor: int = 3, *, offset: int = 0) -> int:
        return x * factor + offset

    lowered = module.functions["kw.scale"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), scale)
    calls = [((2,), {}), ((2, 5), {}), ((2,), {"offset": 1}), ((), {"x": 2, "factor": 4})]
    for args, keywords in calls:
        assert binding.wrapper(*args, **keywords) == scale(*args, **keywords)
    assert binding.calls == len(calls) and binding.fallbacks == 0
    with pytest.raises(TypeError, match="unexpected keyword argument 'nope'"):
        binding.wrapper(2, nope=1)


VALUE_CLASS_ARGUMENT = """
from dataclasses import dataclass


@dataclass
class Clinic:
    doctors: int
    rate: float


@dataclass
class Report:
    seen: int = 0

    def add(self, n: int) -> None:
        self.seen += n


def run(clinic: Clinic, days: int = 2) -> Report:
    report = Report()
    for d in range(days):
        report.add(clinic.doctors + d)
    return report


print(run(Clinic(3, 1.5), 4).seen, run(Clinic(doctors=1, rate=0.5)).seen)
"""

#: Runs a built artifact in this process and says which entries are native,
#: and how many calls Python bound by keyword or default before making them.
_STATS = """
import sys
from pathlib import Path

import ppy_runtime.binding as binding
import ppy_runtime.launch as launch

made = []
plain = launch.PrebuiltBinder.bind
keyed = []
spell = binding._keyword_call


def counted(self, module, function, fallback):
    entry = plain(self, module, function, fallback)
    made.append((module + "." + function, entry))
    return entry


def spelled(*args):
    keyed.append(1)
    return spell(*args)


launch.PrebuiltBinder.bind = counted
binding._keyword_call = spelled
launch.main(Path(sys.argv[1]), [])
for name, entry in made:
    print("NATIVE", name, binding.signature_of(entry) is not None)
print("KEYED", len(keyed))
"""


@requires_llvm
@requires_cc
def test_a_value_class_argument_crosses_with_an_object_result(tmp_path: Path):
    """A function taking a value class and returning an object runs its native
    body when Python calls it, by position or with a default left out, which
    Python binds before it makes the call through the native entry."""
    expected = _expected(tmp_path, VALUE_CLASS_ARGUMENT)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    (tmp_path / "stats.py").write_text(_STATS, encoding="utf-8")
    ran = _run(tmp_path, "stats.py", str(tmp_path / "dist" / "ppy-bindings.json"))
    assert ran.returncode == 0, ran.stderr
    lines = ran.stdout.splitlines()
    assert lines[0] == expected
    assert "NATIVE prog.run True" in lines, lines
    assert "KEYED 1" in lines, lines


RAISES_WITH_A_LIST = """
def simplify(pts: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon!r}")
    keep: list[tuple[float, float]] = []
    for p in pts:
        if p[1] > epsilon:
            keep.append(p)
    return keep


try:
    simplify([(0.0, 0.0), (1.0, 2.0)], -1.0)
except ValueError as e:
    print(e)
print(simplify([(0.0, 0.0), (1.0, 2.0)], 1.0))
"""

#: Records, in order, the boundary letting go of its copies and the sweep.
_ORDER = """
import sys
from pathlib import Path

import ppy_runtime.binding as binding
import ppy_runtime.collection_boundary as crossing
import ppy_runtime.launch as launch

events = []
close = crossing.Boundary.close
sweep = binding._let_go_of_raised


def closed(self):
    if self._owned:
        events.append("close")
    close(self)


def swept(owner, native):
    events.append("sweep")
    sweep(owner, native)


crossing.Boundary.close = closed
binding._let_go_of_raised = swept
launch.main(Path(sys.argv[1]), [])
print("ORDER", " ".join(events))
"""


@requires_llvm
@requires_cc
def test_a_raised_call_lets_go_of_its_argument_copies_before_the_sweep(tmp_path: Path):
    """The sweep after a raised status frees everything on the thread's list; a
    copy of an argument freed there and released again by the boundary was a
    double free (`geometry/ramer_douglas_peucker.py` aborted under `ppy run`)."""
    expected = _expected(tmp_path, RAISES_WITH_A_LIST)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    (tmp_path / "order.py").write_text(_ORDER, encoding="utf-8")
    ran = _run(tmp_path, "order.py", str(tmp_path / "dist" / "ppy-bindings.json"))
    assert ran.returncode == 0, ran.stderr
    lines = ran.stdout.splitlines()
    assert "\n".join(lines[:-1]) == expected
    # The generated wrapper copies the list itself (`crossing.c`) and lets go
    # of its copies before the sweep the same way; the Python-level boundary,
    # where it is the one that serves the call, closes first too.
    assert lines[-1].strip() == "ORDER" or lines[-1].startswith("ORDER close sweep"), lines[-1]


# -- what binding refuses ------------------------------------------------------------


def _info(source: str):  # type: ignore[no-untyped-def]
    from ppy_compiler.analysis.symbols import FunctionInfo, ParamInfo
    from ppy_compiler.analysis.types import UNKNOWN

    node = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node, ast.FunctionDef)
    params = []
    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
    for arg, default in zip(node.args.args, defaults, strict=True):
        params.append(ParamInfo(arg.arg, UNKNOWN, has_default=default is not None, default=default))
    for arg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        params.append(
            ParamInfo(
                arg.arg,
                UNKNOWN,
                has_default=kw_default is not None,
                default=kw_default,
                kind="keyword_only",
            )
        )
    return FunctionInfo(node.name, f"m.{node.name}", "m", node, Path("m.py"), params=params)


def _call(source: str) -> ast.Call:
    found = ast.parse(source, mode="eval").body
    assert isinstance(found, ast.Call)
    return found


def test_binding_puts_keywords_and_constant_defaults_in_place():
    info = _info("def f(a, b=2, *, c=-1, d=(1, 'x')): pass")
    call = _call("f(1, c=3)")
    spelled = bind_arguments(info, call.args, call.keywords)
    assert [ast.unparse(a) for a in spelled] == ["1", "2", "3", "(1, 'x')"]


@pytest.mark.parametrize(
    ("definition", "call", "why"),
    [
        ("def f(a, b=[]): pass", "f(1)", "not a constant"),
        ("def f(a, b): pass", "f(b=g(), a=h())", "another order"),
        ("def f(a, b): pass", "f(1)", "wrong number"),
        ("def f(a, b): pass", "f(1, c=2)", "wrong number"),
        ("def f(a, *rest): pass", "f(1)", "`*args`"),
    ],
)
def test_binding_refuses_what_python_would_see(definition: str, call: str, why: str):
    info = _info(definition) if "*rest" not in definition else None
    if info is None:
        from ppy_compiler.analysis.symbols import FunctionInfo, ParamInfo
        from ppy_compiler.analysis.types import UNKNOWN

        node = ast.parse(definition).body[0]
        assert isinstance(node, ast.FunctionDef)
        info = FunctionInfo(
            "f",
            "m.f",
            "m",
            node,
            Path("m.py"),
            params=[ParamInfo("a", UNKNOWN), ParamInfo("rest", UNKNOWN, kind="var_positional")],
        )
    spelled = _call(call)
    with pytest.raises(Unsupported, match=why):
        bind_arguments(info, spelled.args, spelled.keywords)


def test_one_keyword_that_runs_code_may_move():
    info = _info("def f(a, b): pass")
    call = _call("f(b=g(), a=x)")
    assert [ast.unparse(a) for a in bind_arguments(info, call.args, call.keywords)] == [
        "x",
        "g()",
    ]
