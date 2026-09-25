"""Native strings: `str` as a counted handle into the C runtime.

Each program runs on every path and each named function goes native; a
standalone binary agrees; and the emitted C and C++, safe and unsafe, run
under AddressSanitizer with leak detection, which holds the reference
counting of strings, of lists of them, and of the collections keyed by them
to account. CPython's own output is the expected answer.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status
from ppy_runtime import collections as runtime

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

METHODS = r"""
from dataclasses import dataclass

from ppy import HashMap, Heap, TreeSet, Vec


@dataclass
class Person:
    name: str
    age: int


class Tag:
    def __init__(self, label: str) -> None:
        self.label: str = label
        self.hits: int = 0

    def hit(self, extra: str) -> str:
        self.hits += 1
        self.label = self.label + extra
        return self.label


def word_counts(text: str) -> int:
    counts = HashMap[str, int]()
    for word in text.lower().split():
        word = word.strip(".,!?")
        counts[word] = counts.get(word, 0) + 1
    best = 0
    for word in counts:
        if counts[word] > best:
            best = counts[word]
    return best * 1000 + len(counts)


def ordered(text: str) -> str:
    keys = TreeSet[str]()
    for word in text.split():
        keys.add(word)
    out = Vec[str]()
    for key in keys:
        out.push(key)
    return ",".join(["<", out[0], out[len(out) - 1], ">"])


def smallest(words: Vec[str]) -> str:
    heap = Heap[str]()
    for w in words:
        heap.push(w)
    return heap.pop() + "/" + heap.peek()


def parse(line: str) -> float:
    left, sep, right = line.partition("=")
    total = 0.0
    for piece in right.split(","):
        total += float(piece)
    return total + int(left.strip()) + len(sep)


def people() -> str:
    crew = Vec[Person]()
    crew.push(Person("ada", 36))
    crew.push(Person("linus", 21))
    names = ""
    for p in crew:
        names += f"{p.name.capitalize():<6}|{p.age:03d};"
    return names


def tags() -> str:
    t = Tag("x")
    t.hit("y")
    return t.hit("z") + str(t.hits)


def formats(n: int, x: float) -> str:
    parts = [f"{n:,}", f"{n:#x}", f"{n:+08d}", f"{x:.3e}", f"{x:10.2f}", f"{x:%}", f"{x!r}", f"{'hi'!r}", repr(3.0), str(True)]
    return " ".join(parts)


def chars(s: str) -> int:
    total = 0
    for c in s:
        total += ord(c)
    return total + ord(chr(955)) + len(chr(128512))


def lines(text: str) -> int:
    n = 0
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        n += len(line.rstrip())
    return n


def extremes(a: str, b: str, c: str) -> str:
    return min(a, b, c) + max(a, b, c)


def main() -> None:
    print(word_counts("The cat. The dog! the END, cat?"))
    print(ordered("pear apple fig banana apple"))
    v = Vec[str]()
    for w in "delta alpha charlie bravo".split():
        v.push(w)
    print(smallest(v))
    print(parse(" 7 =1.5,2.25, 3"))
    print(people())
    print(tags())
    print(formats(-1234567, 3.14159))
    print(chars("héllo"))
    print(lines("# head\nabc  \n\n  de\r\nlast"))
    print(extremes("mango", "apple", "zucchini"))
    s = "Hello, World"
    print(s[::-1], s[7:], s[-5:-1], s.find("o"), s.rfind("o"), s.count("l"), s.replace("l", "L", 2))
    print(s.upper(), s.lower(), s.title(), s.swapcase(), s.isalpha(), "123".isdigit(), s.center(18, "*"))
    print("a,b,,c".split(","), "  x  y ".rsplit(None, 1), "x".zfill(4), "-7".zfill(4), s.removeprefix("Hello"))


main()
"""

READING = """
import ppy
from ppy import HashMap, TreeMap


def tally(lines: int) -> str:
    counts = HashMap[str, int]()
    order = TreeMap[str, int]()
    for _ in range(lines):
        line = ppy.input[str]()
        for word in line.split():
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1
            order[key] = len(key)
    first = ppy.scan[str]()
    last = ppy.scan[str]()
    shown = ""
    for key in order:
        shown += f"{key}={counts[key]} "
    return shown.rstrip() + f" | {first}+{last} " + str(len(counts))


def main() -> None:
    n = int(ppy.input[str]())
    print(tally(n))


main()
"""

LISTS = """
def build(n: int) -> list[str]:
    out = ["zero"]
    for i in range(n):
        out.append(str(i * i))
    out.insert(1, "one")
    out.insert(-1, "x")
    return out


def run(n: int) -> str:
    words = build(n)
    words.reverse()
    popped = words.pop(0) + words.pop()
    words.sort()
    shown = ",".join(words)
    head, tail = "a:b".split(":")
    return shown + ";" + popped + ";" + head + tail + ";" + str(words.index("x"))


def main() -> None:
    print(run(6))
    print(build(3))


main()
"""

CLASSES = r"""
from dataclasses import dataclass, field

from ppy import HashMap, Vec


@dataclass
class Animal:
    name: str
    legs: int = 4

    def describe(self) -> str:
        return f"{self.name} walks on {self.legs}"


@dataclass
class Bird(Animal):
    song: str = "tweet"

    def describe(self) -> str:
        return f"{self.name} sings {self.song!r} on {self.legs}"


class Label:
    def __init__(self, text: str) -> None:
        self.text: str = text
        self.parts: Vec[str] = Vec[str]()

    def add(self, part: str) -> None:
        self.parts.push(part)
        self.text += "/" + part

    def __len__(self) -> int:
        return len(self.text)

    def __eq__(self, other: "Label") -> bool:
        return self.text == other.text


def zoo(n: int) -> str:
    animals = Vec[Animal]()
    for i in range(n):
        if i % 2:
            animals.push(Bird(f"b{i}", 2, "la" * i))
        else:
            animals.push(Animal(f"a{i}"))
    out = ""
    for a in animals:
        out += a.describe() + "; "
    return out


def labels(n: int) -> int:
    first = Label("root")
    second = Label("root")
    for i in range(n):
        first.add(str(i))
        second.add(str(i))
    names = HashMap[str, int]()
    for part in first.parts:
        names[part] = len(part)
    return len(first) * 10 + int(first == second) + len(names)


def main() -> None:
    print(zoo(4))
    print(labels(12))


main()
"""

PROGRAMS = {
    "methods": (
        METHODS,
        "",
        [
            "word_counts",
            "ordered",
            "smallest",
            "parse",
            "people",
            "tags",
            "formats",
            "chars",
            "lines",
            "extremes",
        ],
    ),
    # Reading input is native in a standalone build; under `ppy run` it is IO.
    "reading": (READING, "3\nThe cat saw\nthe DOG\n\nalpha\n  beta gamma\n", []),
    "lists": (LISTS, "", ["build", "run"]),
    # String fields, a subclass that adds one, and a class whose fields hold them.
    "classes": (CLASSES, "", ["zoo", "labels"]),
}


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str, text: str = "") -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        input=text,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _expected(tmp_path: Path, name: str) -> str:
    source, text, _ = PROGRAMS[name]
    _write(tmp_path, source)
    done = _run(tmp_path, "prog.ppy", text=text)
    assert done.returncode == 0, done.stderr
    return done.stdout


@requires_llvm
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    expected = _expected(tmp_path, name)
    _, text, natives = PROGRAMS[name]
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args, text=text)
        assert done.returncode == 0, done.stderr
        assert done.stdout == expected, args
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    expected = _expected(tmp_path, name)
    _, text, _ = PROGRAMS[name]
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], input=text, capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == expected


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
def test_emitted_source_frees_every_string_once(
    tmp_path: Path, name: str, language: str, unsafe: bool
):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    expected = _expected(tmp_path, name)
    _, text, _ = PROGRAMS[name]
    emitted = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path,
        "-m",
        "ppy_compiler",
        "emit",
        language,
        "--standalone",
        *flags,
        "prog.ppy",
        "-o",
        emitted.name,
    )
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    subprocess.run(
        [
            compiler,
            standard,
            "-g",
            "-O1",
            "-fsanitize=address,undefined",
            str(emitted),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    ran = subprocess.run(
        [str(binary)],
        input=text,
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == expected


BOUNDARY = """
import ppy


@ppy.native
def greet(name: str, times: int) -> str:
    out = ""
    for i in range(times):
        out += f"hi {name}#{i};"
    return out


@ppy.native
def vowels(text: str) -> int:
    n = 0
    for c in text:
        if c in "aeioué":
            n += 1
    return n


@ppy.native
def shout(text: str) -> str:
    return text.upper() + "!"


@ppy.native
def number(text: str) -> int:
    return int(text) * 2


def main() -> None:
    print(greet("ann", 3), greet("日本", 1))
    print(vowels("héllo wörld"), vowels("🙂ae"))
    print(shout("abc"), shout("straße"))
    print(number(" 21 "))
    try:
        number("x")
    except ValueError as error:
        print("ValueError", error)


main()
"""


@requires_llvm
def test_python_passes_and_takes_strings_across_the_boundary(tmp_path: Path):
    """A native function with `str` parameters and results is called from
    Python: UTF-8 in, a copy out, and a guard (a case outside ASCII, a bad
    literal) hands the call back to Python, which gives CPython's answer."""
    _write(tmp_path, BOUNDARY)
    expected = _run(tmp_path, "prog.ppy")
    assert expected.returncode == 0, expected.stderr
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert done.stdout == expected.stdout
    for function in ("greet", "vowels", "shout", "number"):
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


def test_the_runtime_answers_as_cpython_does(tmp_path: Path):
    """The runtime's methods, parsers, and formatters against CPython's, on
    ASCII and on text that is not."""
    compiler = c_compiler()
    if compiler is None:
        pytest.skip("no C compiler")
    source = tmp_path / "strings.c"
    source.write_text(runtime.library_source(), encoding="utf-8")
    library = tmp_path / "strings.so"
    subprocess.run(
        [compiler, "-std=c11", "-O1", "-shared", "-fPIC", "-o", str(library), str(source)],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    pointer, word = ctypes.c_void_p, ctypes.c_int64
    for name, result, arguments in (
        ("ppy_str_new", pointer, [ctypes.c_char_p, word]),
        ("ppy_str_data", pointer, [pointer]),
        ("ppy_str_bytes", word, [pointer]),
        ("ppy_str_len", word, [pointer]),
        ("ppy_coll_release", None, [pointer]),
        ("ppy_coll_len", word, [pointer]),
        ("ppy_seq_at", pointer, [pointer, word]),
        ("ppy_str_split", pointer, [pointer, pointer, word]),
        ("ppy_str_slice", pointer, [pointer, word, word, word, word]),
        ("ppy_str_find", word, [pointer, pointer, word, word, word, word]),
        ("ppy_str_replace", pointer, [pointer, pointer, pointer, word]),
        ("ppy_str_strip", pointer, [pointer, pointer, word]),
        ("ppy_str_from_float", pointer, [ctypes.c_double]),
        ("ppy_str_builder", pointer, [word]),
        ("ppy_str_finish", pointer, [pointer]),
        ("ppy_str_format_float", word, [pointer, ctypes.c_double, ctypes.c_char_p, word]),
        ("ppy_str_format_int", word, [pointer, word, ctypes.c_char_p, word]),
        ("ppy_str_to_int", word, [pointer, word, ctypes.POINTER(word)]),
    ):
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments

    def make(text: str) -> int:
        data = text.encode()
        return lib.ppy_str_new(data, len(data))

    def read(handle: int) -> str:
        text = ctypes.string_at(lib.ppy_str_data(handle), lib.ppy_str_bytes(handle)).decode()
        assert lib.ppy_str_len(handle) == len(text)
        lib.ppy_coll_release(handle)
        return text

    def pieces(listed: int) -> list[str]:
        found = []
        for i in range(lib.ppy_coll_len(listed)):
            handle = ctypes.c_void_p.from_address(lib.ppy_seq_at(listed, i)).value
            found.append(
                ctypes.string_at(lib.ppy_str_data(handle), lib.ppy_str_bytes(handle)).decode()
            )
        lib.ppy_coll_release(listed)
        return found

    samples = [
        "",
        "a",
        "  a b  c ",
        "héllo wörld",
        "a,b,,c",
        "日本語テキスト",
        "abab",
        "\u3000a\u3000",
    ]
    for text in samples:
        for sep in (None, ",", "a", "語"):
            for limit in (-1, 0, 1):
                got = pieces(lib.ppy_str_split(make(text), make(sep) if sep else None, limit))
                assert got == text.split(sep, limit), (text, sep, limit)
        for start in (None, -3, 0, 2):
            for stop in (None, -1, 4):
                for step in (1, 2, -1, -2):
                    given = (start is not None) | ((stop is not None) << 1)
                    got = read(lib.ppy_str_slice(make(text), start or 0, stop or 0, step, given))
                    assert got == text[start:stop:step], (text, start, stop, step)
        for sub in ("", "a", "語", "zz"):
            assert lib.ppy_str_find(make(text), make(sub), 0, 0, 0, 0) == text.find(sub)
            assert lib.ppy_str_find(make(text), make(sub), 0, 0, 0, 1) == text.rfind(sub)
            assert read(lib.ppy_str_replace(make(text), make(sub), make("-"), -1)) == text.replace(
                sub, "-"
            )
        assert read(lib.ppy_str_strip(make(text), None, 0)) == text.strip()
    for value in (0.0, -0.0, 0.1, 1e16, 1e-5, 1 / 3, 5e-324, float("inf"), 123456789.123):
        assert read(lib.ppy_str_from_float(value)) == repr(value)
    for spec in ("", ",", ">10", "+012", ".3", ".2f", "e", "g", "%", "08,", "z.1f"):
        for value in (0.0, -0.001, 1234567.891, float("nan"), -float("inf")):
            builder = lib.ppy_str_builder(0)
            assert lib.ppy_str_format_float(builder, value, spec.encode(), len(spec))
            assert read(lib.ppy_str_finish(builder)) == format(value, spec), (value, spec)
    for spec in ("", ",", "#x", "08b", "+d", "^9", "_", "X"):
        for value in (0, -1, 255, -1234567, 2**63 - 1):
            builder = lib.ppy_str_builder(0)
            assert lib.ppy_str_format_int(builder, value, spec.encode(), len(spec))
            assert read(lib.ppy_str_finish(builder)) == format(value, spec), (value, spec)
    for text, base in (("0x1f", 0), (" -12_3 ", 10), ("z", 36), ("010", 0), ("1__2", 10)):
        out = word(0)
        status = lib.ppy_str_to_int(make(text), base, ctypes.byref(out))
        try:
            wanted = int(text, base)
        except ValueError:
            assert status == 1, text
        else:
            assert (status, out.value) == (0, wanted), text
