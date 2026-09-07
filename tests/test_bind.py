"""`ppy bind header`: a C header read by Clang, written as PPY bindings."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.bind import libclang_status

_ready, _detail = libclang_status()
requires_libclang = pytest.mark.skipif(not _ready, reason=_detail)
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

HEADER = """
    #ifndef SHAPES_H
    #define SHAPES_H
    #include <stdint.h>
    #include <stddef.h>

    #define SHAPES_VERSION 3
    #define SHAPES_SCALE 1.5
    #define SHAPES_MASK 0xff
    #define SHAPES_NAME "shapes"
    #define SHAPES_SQUARE(x) ((x) * (x))

    typedef int32_t shape_id;
    typedef unsigned long shape_count;

    enum shape_kind { SHAPE_CIRCLE = 1, SHAPE_SQUARE = 2, SHAPE_OTHER = 10 };

    typedef struct {
        double x;
        double y;
    } point;

    struct box {
        int width;
        int height;
        _Bool filled;
    };

    struct opaque;

    double shapes_area(double width, double height);
    int64_t shapes_sum(const int64_t *values, size_t count);
    void shapes_fill(double *values, size_t count, double value);
    shape_id shapes_register(enum shape_kind kind);
    int shapes_printf(const char *format, ...);
    double shapes_norm(point p);
    void shapes_sort(int *values, size_t count, int (*compare)(const void *, const void *));
    #endif
    """

IMPLEMENTATION = """
    #include "shapes.h"
    double shapes_area(double width, double height) { return width * height; }
    int64_t shapes_sum(const int64_t *values, size_t count) {
        int64_t total = 0;
        for (size_t i = 0; i < count; i++) total += values[i];
        return total;
    }
    void shapes_fill(double *values, size_t count, double value) {
        for (size_t i = 0; i < count; i++) values[i] = value;
    }
    shape_id shapes_register(enum shape_kind kind) { return (shape_id)kind * 100; }
    """

PROGRAM = """
    from ppy import native

    import shapes


    def main() -> None:
        values = native.stack_alloc[float](4)
        shapes.shapes_fill(values, 4, 2.5)
        counts = native.stack_alloc[int](3)
        native.store(native.offset(counts, 0), 5)
        native.store(native.offset(counts, 1), 7)
        native.store(native.offset(counts, 2), 9)
        print(shapes.shapes_area(3.0, 4.0), native.load(native.offset(values, 3)))
        print(shapes.shapes_sum(counts, 3), shapes.shapes_register(shapes.SHAPE_SQUARE))
        print(shapes.SHAPES_VERSION, shapes.SHAPES_SCALE, shapes.SHAPES_MASK)


    main()
    """


def _ppy(cwd: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@requires_libclang
def test_a_header_becomes_typed_bindings_and_what_it_cannot_is_listed(tmp_path: Path):
    (tmp_path / "shapes.h").write_text(textwrap.dedent(HEADER).lstrip("\n"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    printed = _ppy(tmp_path, "bind", "header", "shapes.h")
    assert printed.returncode == 0, printed.stderr
    text = printed.stdout
    assert 'lib = ffi.library("shapes")' in text
    assert "SHAPES_VERSION: int = 3" in text and "SHAPES_SCALE: float = 1.5" in text
    assert "SHAPES_MASK: int = 255" in text
    assert "SHAPE_SQUARE: int = 2" in text and "shape_kind = int" in text
    assert "shape_id = ppy.i32" in text and "shape_count = ppy.u64" in text
    assert "@dataclass\nclass point:\n    x: float\n    y: float" in text
    assert "class box:\n    width: ppy.i32\n    height: ppy.i32\n    filled: bool" in text
    assert (
        '@ffi.bind(lib, symbol="shapes_area")\n'
        "def shapes_area(width: float, height: float) -> float: ..."
    ) in text
    assert "def shapes_sum(values: native.const_ptr[int], count: ppy.u64) -> int: ..." in text
    assert (
        "def shapes_fill(values: native.ptr[float], count: ppy.u64, value: float) -> None: ..."
        in text
    )
    assert "def shapes_register(kind: ppy.u32) -> ppy.i32: ..." in text
    assert "#   SHAPES_NAME: a macro that is not one number" in text
    assert "SHAPES_SQUARE(" not in text
    assert "#   shapes_printf: a variadic function" in text
    assert "#   shapes_norm: parameter `p` is point" in text
    assert "#   shapes_sort: parameter `compare` is" in text
    assert "#   opaque: an opaque struct" in text
    written = _ppy(
        tmp_path, "bind", "header", "shapes.h", "-o", "shapes.ppy", "--library", "shapes"
    )
    assert written.returncode == 0, written.stderr
    assert (tmp_path / "shapes.ppy").read_text(encoding="utf-8") == text
    checked = _ppy(tmp_path, "check", "shapes.ppy")
    assert checked.returncode == 0, checked.stderr + checked.stdout


@requires_libclang
def test_a_header_clang_cannot_read_is_refused_with_its_line(tmp_path: Path):
    (tmp_path / "broken.h").write_text("int f(;\n", encoding="utf-8")
    done = _ppy(tmp_path, "bind", "header", "broken.h")
    assert done.returncode == 2
    assert "E1806" in done.stderr and "broken.h:1" in done.stderr
    missing = _ppy(tmp_path, "bind", "header", "absent.h")
    assert missing.returncode == 2 and "does not exist" in missing.stderr


@requires_libclang
@requires_cc
def test_the_bindings_call_the_library_on_every_path(tmp_path: Path):
    (tmp_path / "shapes.h").write_text(textwrap.dedent(HEADER).lstrip("\n"), encoding="utf-8")
    (tmp_path / "shapes.c").write_text(
        textwrap.dedent(IMPLEMENTATION).lstrip("\n"), encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
    )
    library = tmp_path / "libshapes.so"
    built = subprocess.run(
        [c_compiler(), "-shared", "-fPIC", "-O1", "-o", str(library), str(tmp_path / "shapes.c")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    assert _ppy(tmp_path, "bind", "header", "shapes.h", "-o", "shapes.ppy").returncode == 0
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    import os

    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    env["LD_LIBRARY_PATH"] = str(tmp_path)
    env["LIBRARY_PATH"] = str(tmp_path)
    outputs = []
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = subprocess.run(
            [sys.executable, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0] == "12.0 2.5\n21 200\n3 1.5 255\n"
