"""`ppy build --library` and `ppy build --target`: packaged, and for another machine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, toolchain_status
from ppy_compiler.target import host_target

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_usable, _detail = toolchain_status()
requires_toolchain = pytest.mark.skipif(not _usable, reason=f"no native toolchain: {_detail}")

LIBRARY = """\
from ppy import Buffer, native


@native.export(name="ppy_dot")
def dot(a: Buffer[float], b: Buffer[float], n: int) -> float:
    total = 0.0
    for i in range(n):
        total += a[i] * b[i]
    return total


@native.export()
def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else (hi if x > hi else x)
"""

PLAIN = """\
def total(xs: list[int]) -> int:
    result = 0
    for x in xs:
        result += x
    return result
"""

UNEXPORTABLE = """

@native.export()
def spelled(x: int) -> int:
    import json
    return len(json.dumps(x))
"""

RAISING = """\
import math

from ppy import native


@native.export()
def hypot(a: float, b: float) -> float:
    return math.sqrt(a * a + b * b) + math.sin(0.0) * math.exp(a)


@native.export()
def pick(i: int) -> int:
    xs = [10, 20, 30]
    return xs[i]


@native.export()
def floordiv(a: int, b: int) -> int:
    return a // b


@native.export()
def lookup(k: int) -> int:
    table = {1: 10, 2: 20}
    return table[k]


@native.export()
def halve(x: float) -> float:
    if x < 0:
        raise ValueError("a negative input")
    return x / 2.0
"""

CONSUMER = """\
#include <stdio.h>
#include "NAME.h"

static void show(const char *what, double value) {
    const char *error = ppy_last_error();
    printf("%s %g %s\\n", what, value, error != NULL ? error : "ok");
}

int main(void) {
    show("hypot", hypot(3.0, 4.0));
    show("pick(1)", (double)pick(1));
    show("pick(7)", (double)pick(7));
    show("pick(-1)", (double)pick(-1));
    show("floordiv(1, 0)", (double)floordiv(1, 0));
    show("lookup(3)", (double)lookup(3));
    show("halve(-1)", halve(-1.0));
    show("halve(3)", halve(3.0));
    for (int i = 0; i < 1000; i++) {
        (void)lookup(3);
        (void)halve(-1.0);
    }
    printf("after many %s\\n", ppy_last_error());
    return 0;
}
"""

#: ELF `e_machine` values, to read an object's architecture off its header.
_ELF_MACHINES = {"x86_64": 62, "aarch64": 183, "riscv64": 243}


def _build(project: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", *args],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        env={"PPY_LOWERING": "ir", **__import__("os").environ},
    )


def _project(tmp_path: Path, source: str, name: str = "lib.ppy") -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / name).write_text(source, encoding="utf-8")
    return tmp_path / name


@requires_toolchain
def test_a_library_build_lays_out_lib_include_and_pkgconfig(tmp_path: Path):
    _project(tmp_path, LIBRARY)
    built = _build(tmp_path, "lib.ppy", "--library", "-o", "dist")
    assert built.returncode == 0, built.stderr
    name = tmp_path.name
    dist = tmp_path / "dist"
    library = dist / "lib" / f"libppy_{name}.so"
    header = dist / "include" / f"{name}.h"
    pc = dist / "lib" / "pkgconfig" / f"{name}.pc"
    assert library.is_file() and header.is_file() and pc.is_file()
    assert not (dist / f"libppy_{name}.so").exists(), "the library moved under lib/"
    assert not (dist / f"{name}.h").exists()
    assert "double ppy_dot(" in header.read_text(encoding="utf-8")
    spelled = pc.read_text(encoding="utf-8")
    assert f"Libs: -L${{libdir}} -lppy_{name}" in spelled and "Cflags: -I${includedir}" in spelled
    assert "prefix=${pcfiledir}/../.." in spelled
    manifest = json.loads((dist / "ppy-bindings.json").read_text(encoding="utf-8"))
    assert set(manifest["exports"]) == {"ppy_dot", "clamp"}
    assert "package:  dist" in built.stderr
    assert not list(dist.glob("ppy_wrappers_*")), "a C consumer needs no Python boundary"
    assert not (dist / "lib").is_file()


@requires_toolchain
def test_a_library_build_with_nothing_to_export_says_so(tmp_path: Path):
    _project(tmp_path, PLAIN)
    built = _build(tmp_path, "lib.ppy", "--library", "-o", "dist")
    assert built.returncode == 2
    assert "E1805" in built.stderr and "@ppy.native.export" in built.stderr


@requires_toolchain
def test_an_export_that_is_not_native_stops_the_library_build(tmp_path: Path):
    _project(tmp_path, LIBRARY + UNEXPORTABLE)
    built = _build(tmp_path, "lib.ppy", "--library", "-o", "dist")
    assert built.returncode == 2
    assert "E1807" in built.stderr and "`spelled` (lib.spelled)" in built.stderr, built.stderr
    assert not (tmp_path / "dist" / "include").exists()


def _pkg_config(pc: Path, field: str) -> list[str]:
    """A field of a pkg-config file with its variables expanded, as pkg-config would."""
    variables = {"pcfiledir": str(pc.parent)}
    for line in pc.read_text(encoding="utf-8").splitlines():
        name, equals, value = line.partition("=")
        if equals and ":" not in name:
            for known, spelled in variables.items():
                value = value.replace(f"${{{known}}}", spelled)
            variables[name.strip()] = value.strip()
        elif line.startswith(f"{field}:"):
            value = line.split(":", 1)[1]
            for known, spelled in variables.items():
                value = value.replace(f"${{{known}}}", spelled)
            return value.split()
    raise AssertionError(f"{pc} has no {field}")


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


def _raised(source: str, function: str, *arguments: object) -> str:
    """CPython's last line for what calling `function` of `source` raises."""
    names: dict[str, object] = {}
    exec(source, names)
    try:
        names[function](*arguments)  # type: ignore[operator]
    except Exception as error:  # noqa: BLE001 - the line is the point
        return f"{type(error).__name__}: {error}"
    raise AssertionError(f"{function}{arguments} did not raise")


@requires_toolchain
@pytest.mark.skipif(host_target().os != "linux", reason="links with the ELF toolchain")
def test_a_c_consumer_links_with_the_package_flags_and_reads_what_an_export_raised(
    tmp_path: Path,
):
    compiler = c_compiler()
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip("no C compiler with AddressSanitizer")
    _project(tmp_path, RAISING)
    built = _build(tmp_path, "lib.ppy", "--library", "-o", "dist")
    assert built.returncode == 0, built.stderr
    name = tmp_path.name
    dist = tmp_path / "dist"
    header = (dist / "include" / f"{name}.h").read_text(encoding="utf-8")
    assert "const char *ppy_last_error(void);" in header
    consumer = tmp_path / "consumer.c"
    consumer.write_text(CONSUMER.replace("NAME", name), encoding="utf-8")
    pc = dist / "lib" / "pkgconfig" / f"{name}.pc"
    binary = tmp_path / "consumer"
    linked = subprocess.run(
        # Only what the package file says: no `-lm` of the consumer's own.
        [compiler, "-Wall", "-Wextra", "-fsanitize=address,undefined", str(consumer),
         *_pkg_config(pc, "Cflags"), *_pkg_config(pc, "Libs"),
         f"-Wl,-rpath,{dist / 'lib'}", "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert linked.returncode == 0, linked.stderr
    assert "warning:" not in linked.stderr, linked.stderr
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1", "UBSAN_OPTIONS": "halt_on_error=1"},
    )
    assert ran.returncode == 0, ran.stderr
    expected = [
        "hypot 5 ok",
        "pick(1) 20 ok",
        f"pick(7) 0 {_raised(RAISING, 'pick', 7)}",
        "pick(-1) 30 ok",
        f"floordiv(1, 0) 0 {_raised(RAISING, 'floordiv', 1, 0)}",
        f"lookup(3) 0 {_raised(RAISING, 'lookup', 3)}",
        f"halve(-1) 0 {_raised(RAISING, 'halve', -1.0)}",
        "halve(3) 1.5 ok",
        f"after many {_raised(RAISING, 'halve', -1.0)}",
    ]
    assert ran.stdout.splitlines() == expected


@requires_llvm
def test_a_cross_build_emits_objects_for_the_target_and_says_what_it_left_out(tmp_path: Path):
    _project(tmp_path, LIBRARY)
    host = host_target()
    other = "aarch64-linux-gnu" if host.architecture != "aarch64" else "x86_64-linux-gnu"
    built = _build(tmp_path, "lib.ppy", "--target", other, "-o", "dist")
    assert built.returncode == 0, built.stderr
    canonical = f"{other.split('-', maxsplit=1)[0]}-unknown-linux-gnu"
    assert f"target:   {canonical}" in built.stderr
    objects = list((tmp_path / "dist").glob("*.o"))
    assert objects, built.stderr
    header = objects[0].read_bytes()
    assert header[:4] == b"\x7fELF"
    assert int.from_bytes(header[18:20], "little") == _ELF_MACHINES[other.split("-", maxsplit=1)[0]]
    manifest = json.loads((tmp_path / "dist" / "ppy-bindings.json").read_text(encoding="utf-8"))
    assert manifest["target"] == canonical
    assert "built on that machine" in built.stderr, "the wrapper and launcher are not for here"
    assert not list((tmp_path / "dist").glob("ppy_wrappers_*"))
    assert not (tmp_path / "dist" / "lib").exists()
    assert (tmp_path / "dist" / f"{tmp_path.name}.h").is_file(), "the header is target-neutral"
    ran = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "run",
            "--prebuilt",
            "dist/ppy-bindings.json",
            "lib.ppy",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 2
    assert f"built for {canonical}" in ran.stderr


@requires_llvm
def test_a_target_this_compiler_cannot_describe_is_refused(tmp_path: Path):
    _project(tmp_path, LIBRARY)
    built = _build(tmp_path, "lib.ppy", "--target", "x86_64-plan9")
    assert built.returncode == 2
    assert "E1002" in built.stderr and "operating system" in built.stderr
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\ntarget = "foo-linux-gnu"\n', encoding="utf-8"
    )
    configured = _build(tmp_path, "lib.ppy")
    assert configured.returncode == 2 and "architecture" in configured.stderr
