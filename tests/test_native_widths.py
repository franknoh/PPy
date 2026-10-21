"""Native storage widths and C callers, independent of Python's integer model."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _ppy(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _source(root: Path, source: str) -> None:
    (root / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (root / "lib.ppy").write_text(source, encoding="utf-8")


def _emit(root: Path, kind: str = "c") -> str:
    if kind == "llvm":
        kind = "llvm-ir"
    done = _ppy(root, "emit", kind, "lib.ppy")
    assert done.returncode == 0, done.stderr
    return done.stdout


def _compile(root: Path, text: str, kind: str = "c", caller: str = "") -> Path:
    compiler = shutil.which("c++" if kind == "cpp" else "cc")
    if compiler is None:
        pytest.skip("a C/C++ compiler is required")
    path = root / ("lib.cpp" if kind == "cpp" else "lib.c")
    inputs = [str(path)]
    if kind == "llvm":
        from llvmlite import binding as llvm

        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        module = llvm.parse_assembly(text)
        module.verify()
        machine = llvm.Target.from_default_triple().create_target_machine(reloc="pic")
        obj = root / "lib.o"
        obj.write_bytes(machine.emit_object(module))
        inputs = [str(obj)]
        if caller:
            path.write_text(_emit(root, "header") + caller, encoding="utf-8")
            inputs.append(str(path))
    else:
        path.write_text(text + caller, encoding="utf-8")
    output = root / ("caller" if caller else "lib.so")
    flags = [] if caller else ["-shared", "-fPIC"]
    done = subprocess.run(
        [
            compiler,
            "-std=c++17" if kind == "cpp" else "-std=c11",
            "-Werror",
            *flags,
            *inputs,
            "-lm",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return output


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
@pytest.mark.parametrize("prefix", ["ppy.", ""])
def test_explicit_scalar_abi(tmp_path: Path, kind: str, prefix: str):
    types = [
        (f"{sign}{bits}", getattr(ctypes, f"c_{'u' if sign == 'u' else ''}int{bits}"))
        for sign in ("i", "u")
        for bits in (8, 16, 32, 64)
    ]
    source = "import ppy\nfrom ppy import native, i8, i16, i32, i64, u8, u16, u32, u64\n"
    for name, _ in types:
        source += (
            f"@native.export\ndef echo_{name}(x: {prefix}{name}) -> {prefix}{name}:\n    return x\n"
        )
    _source(tmp_path, source)
    text = _emit(tmp_path, kind)
    header = _emit(tmp_path, "header")
    for name, _ in types:
        ctype = f"{'u' if name[0] == 'u' else ''}int{name[1:]}_t"
        assert f"{ctype} echo_{name}({ctype} x)" in header
        if kind != "llvm":
            assert f"{ctype} echo_{name}({ctype} x)" in text
    library = ctypes.CDLL(str(_compile(tmp_path, text, kind)))
    for name, ctype in types:
        function = getattr(library, f"echo_{name}")
        function.argtypes = [ctype]
        function.restype = ctype
        bits = int(name[1:])
        values = [0, (1 << (bits - (name[0] == "i"))) - 1]
        if name[0] == "i":
            values.append(-(1 << (bits - 1)))
        for value in values:
            assert function(value) == value


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_pointer_and_buffer_storage(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, Buffer, i32, i64
@native.export
def update(p: native.ptr[i32], q: native.ptr[i64], b: Buffer[i32]) -> i32:
    native.store(native.offset(p, 1), native.load(p) + 3)
    native.store(native.offset(q, 1), native.load(q) + 5)
    b[1] = b[0] + 7
    return b[1]
""",
    )
    text = _emit(tmp_path, kind)
    caller = """
#include <assert.h>
int main(void) {
    int32_t p[3] = {-9, 0, 123};
    int64_t q[3] = {INT64_C(5000000000), 0, 456};
    int32_t b[3] = {20, 0, 789};
    assert(update(p, q, b, 3) == 27);
    assert(p[1] == -6 && p[2] == 123);
    assert(q[1] == INT64_C(5000000005) && q[2] == 456);
    assert(b[1] == 27 && b[2] == 789);
    return 0;
}
"""
    executable = _compile(tmp_path, text, kind, caller)
    done = subprocess.run([str(executable)], capture_output=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("body", ["pass", "return", "return None"])
@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_void_only_export(tmp_path: Path, body: str, kind: str):
    _source(
        tmp_path, f"from ppy import native\n@native.export\ndef game_init() -> None:\n    {body}\n"
    )
    text = _emit(tmp_path, kind)
    assert "void game_init(void)" in _emit(tmp_path, "header")
    library = ctypes.CDLL(str(_compile(tmp_path, text, kind)))
    library.game_init.restype = None
    library.game_init()


def test_integer_width_restrictions_are_explicit(tmp_path: Path):
    _source(
        tmp_path,
        "from ppy import native\ndef read(p: native.ptr[int]) -> int:\n    return native.load(p)\ndef main() -> None:\n    p = native.stack_alloc[int](1)\n    native.store(p, 1)\n    print(read(p))\nmain()\n",
    )
    library = _ppy(tmp_path, "emit", "c", "lib.ppy", "--unsafe", "--int-width", "32")
    assert library.returncode != 0
    assert "requires unsafe standalone" in library.stderr
    standalone = _ppy(
        tmp_path, "emit", "c", "lib.ppy", "--standalone", "--unsafe", "--int-width", "32"
    )
    assert standalone.returncode != 0
    assert "fixed storage ABI" in standalone.stderr


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
@pytest.mark.parametrize("element,extent", [("i32", 8), ("i16", 3), ("i64", 5)])
def test_borrowed_c_rows(tmp_path: Path, kind: str, element: str, extent: int):
    _source(
        tmp_path,
        f"""from ppy import native, {element}
@native.export
def process(ice: native.ptr[{element}, {extent}]) -> None:
    for row in range(3):
        for col in range({extent}):
            ice[row, col] = ice[row, col] + row + col
    next_row = native.offset(ice, 1)
    next_row[0, 0] += 10
""",
    )
    text = _emit(tmp_path, kind)
    ctype = "int" if element == "i32" else f"int{element[1:]}_t"
    caller = f"""
#include <assert.h>
int main(void) {{
    {ctype} ice[3][{extent}];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < {extent}; ++c) ice[r][c] = r * 20 + c;
    process(ice);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < {extent}; ++c)
            assert(ice[r][c] == r * 21 + 2 * c + ((r == 1 && c == 0) ? 10 : 0));
    return 0;
}}
"""
    if element == "i32":
        caller = (
            "\n"
            + ("static_assert" if kind == "cpp" else "_Static_assert")
            + '(sizeof(int) == 4, "32-bit C int required");\n'
            + caller
        )
    executable = _compile(tmp_path, text, kind, caller)
    done = subprocess.run([str(executable)], capture_output=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_higher_rank_and_const_pointer_calls(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, i32
def read(a: native.const_ptr[i32, 3, 4]) -> i32:
    return a[1, 2, 3]
@native.export
def change(a: native.ptr[i32, 3, 4]) -> i32:
    a[0, 2, 3] = read(a)
    return a[0, 2, 3]
""",
    )
    text = _emit(tmp_path, kind)
    executable = _compile(
        tmp_path,
        text,
        kind,
        """
#include <assert.h>
int main(void) {
    int32_t a[2][3][4] = {0};
    a[1][2][3] = -123;
    assert(change(a) == -123);
    assert(a[0][2][3] == -123 && a[0][2][2] == 0);
    return 0;
}
""",
    )
    assert subprocess.run([str(executable)], check=False).returncode == 0


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_void_init_preserves_external_side_effects(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, i32
@native.extern("record_value")
def record_value(value: i32) -> None: ...
@native.export
def game_init() -> None:
    record_value(73)
@native.export
def answer() -> i32:
    game_init()
    return 42
""",
    )
    text = _emit(tmp_path, kind)
    executable = _compile(
        tmp_path,
        text,
        kind,
        """
#include <assert.h>
static int recorded;
void record_value(int32_t value) { recorded += value; }
int main(void) {
    game_init();
    assert(recorded == 73);
    assert(answer() == 42 && recorded == 146);
    return 0;
}
""",
    )
    assert subprocess.run([str(executable)], check=False).returncode == 0


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_checked_narrowing_and_unsigned_storage(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, i32, u32, u64
@native.export
def narrow(x: int) -> i32:
    return x
@native.export
def unsigned_values(p: native.ptr[u32], q: native.ptr[u64]) -> u64:
    p[1] = p[0]
    q[1] = q[0] + 1
    return q[1] >> 1
""",
    )
    text = _emit(tmp_path, kind)
    if kind != "llvm":
        assert "INT64_C(-2147483648)" in text
    library = ctypes.CDLL(str(_compile(tmp_path, text, kind)))
    narrow = library.ppy_lib_narrow
    narrow.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int32)]
    narrow.restype = ctypes.c_int32
    result = ctypes.c_int32(999)
    assert narrow(2147483647, ctypes.byref(result)) == 0
    assert result.value == 2147483647
    result.value = 999
    assert narrow(2147483648, ctypes.byref(result)) != 0
    assert result.value == 999
    assert narrow(-2147483648, ctypes.byref(result)) == 0
    assert result.value == -2147483648
    result.value = 999
    assert narrow(-2147483649, ctypes.byref(result)) != 0
    assert result.value == 999
    function = library.unsigned_values
    function.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64)]
    function.restype = ctypes.c_uint64
    p = (ctypes.c_uint32 * 3)(4294967295, 0, 123)
    q = (ctypes.c_uint64 * 3)(9223372036854775810, 0, 456)
    assert function(p, q) == 4611686018427387905
    assert p[1] == 4294967295 and p[2] == 123
    assert q[1] == 9223372036854775811 and q[2] == 456


@pytest.mark.parametrize(
    "annotation,index",
    [
        ("native.ptr[i32, 0]", "0, 0"),
        ("native.ptr[i32, -2]", "0, 0"),
        ("native.ptr[i32, 3]", "0"),
        ("native.ptr[i32, 3]", "0, 1.5"),
        ("native.const_ptr[i32, 3]", "0, 0"),
    ],
)
def test_invalid_array_access_is_rejected(tmp_path: Path, annotation: str, index: str):
    _source(
        tmp_path,
        f"from ppy import native, i32\n@native.export\ndef bad(p: {annotation}) -> None:\n    p[{index}] = 1\n",
    )
    done = _ppy(tmp_path, "emit", "c", "lib.ppy")
    assert done.returncode != 0
    assert "nothing to emit" not in done.stderr


def test_incompatible_pointer_types_are_rejected(tmp_path: Path):
    _source(
        tmp_path,
        """from ppy import native, i32, i64
def read(p: native.ptr[i32, 3]) -> i32:
    return p[0, 0]
@native.export
def bad(p: native.ptr[i64, 3]) -> i32:
    return read(p)
""",
    )
    done = _ppy(tmp_path, "emit", "c", "lib.ppy")
    assert done.returncode != 0
    assert "nothing to emit" not in done.stderr


def test_fixed_scalar_types_are_not_rewritten_by_int_width(tmp_path: Path):
    _source(
        tmp_path,
        """from ppy import i64
def echo(x: i64) -> i64:
    return x
def main() -> None:
    print(echo(5))
main()
""",
    )
    done = _ppy(tmp_path, "emit", "c", "lib.ppy", "--standalone", "--unsafe", "--int-width", "32")
    assert done.returncode != 0
    assert "explicit integer types" in done.stderr


@pytest.mark.parametrize("kind", ["i8", "i16", "i32", "u8", "u16", "u32", "u64"])
def test_python_boundary_uses_the_same_integer_width(tmp_path: Path, kind: str):
    from ppy_compiler.backend.llvm.wrapper_build import build_wrappers
    from ppy_compiler.ir import decode
    from ppy_compiler.lowering.abi import signature_from_ir
    from ppy_runtime.binding import bind

    _source(
        tmp_path,
        f"from ppy import native, {kind}\n@native.export\ndef echo(x: {kind}) -> {kind}:\n    return x\n",
    )
    module = decode(_emit(tmp_path, "ir"))
    signature = signature_from_ir(module.functions["lib_echo"])
    library = ctypes.CDLL(str(_compile(tmp_path, _emit(tmp_path))))
    address = ctypes.cast(library.ppy_lib_echo, ctypes.c_void_p).value
    assert address
    signed = kind[0] == "i"
    bits = int(kind[1:])
    value = -(1 << (bits - 1)) if signed else (1 << bits) - 1

    def fallback(x):
        return ("fallback", x)

    binding = bind(signature, address, fallback, owner=library)
    assert binding.wrapper(value) == value
    assert type(binding.wrapper(value)) is int
    outside = 1 << (bits - int(signed))
    assert binding.wrapper(outside) == ("fallback", outside)
    wrappers = build_wrappers("lib", {signature.qualname: signature}, tmp_path)
    assert wrappers.module is not None, wrappers.reason
    entry = wrappers.bind(signature.qualname, address, (), fallback)
    assert entry is not None
    assert entry(value) == value and type(entry(value)) is int
    assert entry(outside) == ("fallback", outside)


@pytest.mark.parametrize("from_ir", [False, True])
def test_example_build_links_to_an_unmodified_c_array(tmp_path: Path, from_ir: bool):
    example = Path(__file__).resolve().parents[1] / "examples/32_native_memory/c_arrays"
    _source(tmp_path, (example / "array_api.ppy").read_text(encoding="utf-8"))
    target = "lib.ppy"
    if from_ir:
        (tmp_path / "lib.ppyir").write_text(_emit(tmp_path, "ir"), encoding="utf-8")
        target = "lib.ppyir"
    built = _ppy(tmp_path, "build", target, "-o", "dist")
    assert built.returncode == 0, built.stderr
    header = next((tmp_path / "dist").glob("*.h"))
    shutil.copyfile(header, tmp_path / "array_api.h")
    caller = tmp_path / "caller.c"
    caller.write_text((example / "caller.c").read_text(encoding="utf-8"), encoding="utf-8")
    library = next((tmp_path / "dist").glob("libppy_*.so"))
    executable = tmp_path / "caller"
    compiled = subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Werror",
            str(caller),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-lm",
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "original C array updated: ice[7][7] = 63"


def test_c_array_example_under_address_and_undefined_sanitizers(tmp_path: Path):
    example = Path(__file__).resolve().parents[1] / "examples/32_native_memory/c_arrays"
    _source(tmp_path, (example / "array_api.ppy").read_text(encoding="utf-8"))
    implementation = tmp_path / "array_api.c"
    implementation.write_text(_emit(tmp_path), encoding="utf-8")
    (tmp_path / "array_api.h").write_text(_emit(tmp_path, "header"), encoding="utf-8")
    caller = tmp_path / "caller.c"
    shutil.copyfile(example / "caller.c", caller)
    executable = tmp_path / "caller"
    compiled = subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            str(implementation),
            str(caller),
            "-lm",
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stderr


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_sized_external_return_is_preserved(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, i32
@native.extern("read_value", pure=True)
def read_value() -> i32: ...
@native.export
def answer() -> i32:
    return read_value()
""",
    )
    executable = _compile(
        tmp_path,
        _emit(tmp_path, kind),
        kind,
        """
#include <assert.h>
int32_t read_value(void) { return -123; }
int main(void) { assert(answer() == -123); return 0; }
""",
    )
    assert subprocess.run([str(executable)], check=False).returncode == 0


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_unsigned_64_bit_constant(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, u64
@native.export
def maximum() -> u64:
    return 18446744073709551615
""",
    )
    library = ctypes.CDLL(str(_compile(tmp_path, _emit(tmp_path, kind), kind)))
    library.maximum.restype = ctypes.c_uint64
    assert library.maximum() == 18446744073709551615


def test_collection_boundary_preserves_signed_byte_result(tmp_path: Path):
    import json

    from ppy_runtime.binding import bind
    from ppy_runtime.manifest import _signature

    _source(
        tmp_path,
        """from ppy import native, i8
@native
def first(xs: list[int]) -> i8:
    xs.append(1)
    return -7
""",
    )
    built = _ppy(tmp_path, "build", "lib.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    manifest = json.loads((tmp_path / "dist/ppy-bindings.json").read_text(encoding="utf-8"))
    signature = _signature(manifest["entries"][0])
    library = ctypes.CDLL(str(tmp_path / "dist" / manifest["native_library"]))
    address = ctypes.cast(getattr(library, signature.symbol), ctypes.c_void_p).value
    assert address

    def fallback(xs):
        pytest.fail("the native collection function must run")

    binding = bind(signature, address, fallback, owner=library)
    xs = []
    result = binding.wrapper(xs)
    assert result == -7 and type(result) is int
    assert xs == [1] and binding.calls == 1


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_explicit_i64_buffer_can_be_forwarded(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, Buffer, i64
def read(b: Buffer[i64]) -> i64:
    return b[1]
@native.export
def forwarded(b: Buffer[i64]) -> i64:
    return read(b)
""",
    )
    executable = _compile(
        tmp_path,
        _emit(tmp_path, kind),
        kind,
        """
#include <assert.h>
int main(void) {
    int64_t b[2] = {0, INT64_C(5000000000)};
    assert(forwarded(b, 2) == INT64_C(5000000000));
    return 0;
}
""",
    )
    assert subprocess.run([str(executable)], check=False).returncode == 0


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_sized_stack_memory_and_local_types(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, i16, i32
@native.export
def scratch(x: i32) -> i32:
    small: i16 = x
    p = native.stack_alloc[i32](3)
    native.store(p, small)
    p[1] = p[0] + native.sizeof[i32]() + native.alignof[i16]()
    return p[1]
""",
    )
    library = ctypes.CDLL(str(_compile(tmp_path, _emit(tmp_path, kind), kind)))
    library.scratch.argtypes = [ctypes.c_int32]
    library.scratch.restype = ctypes.c_int32
    assert library.scratch(-123) == -117


def test_narrow_llvm_exports_follow_the_c_integer_extension_abi(tmp_path: Path):
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("Clang is required for this ABI cross-check")
    _source(
        tmp_path,
        """from ppy import native, i8, i16, u8, u16
@native.export
def signed_byte() -> i8:
    return -7
@native.export
def signed_half() -> i16:
    return -12345
@native.export
def unsigned_byte() -> u8:
    return 249
@native.export
def unsigned_half() -> u16:
    return 60000
""",
    )
    library = _compile(tmp_path, _emit(tmp_path, "llvm"), "llvm")
    caller = tmp_path / "caller.c"
    caller.write_text(
        _emit(tmp_path, "header")
        + """
#include <assert.h>
int main(void) {
    assert((long)signed_byte() == -7);
    assert((long)signed_half() == -12345);
    assert((unsigned long)unsigned_byte() == 249);
    assert((unsigned long)unsigned_half() == 60000);
    return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "caller"
    done = subprocess.run(
        [clang, "-O2", str(caller), str(library), f"-Wl,-rpath,{tmp_path}", "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stderr


@pytest.mark.parametrize("kind", ["c", "cpp", "llvm"])
def test_external_const_row_declaration_matches_c(tmp_path: Path, kind: str):
    _source(
        tmp_path,
        """from ppy import native, i32
@native.extern("read_row", pure=True)
def read_row(p: native.const_ptr[i32, 3]) -> i32: ...
@native.export
def read(p: native.ptr[i32, 3]) -> i32:
    return read_row(p)
""",
    )
    executable = _compile(
        tmp_path,
        _emit(tmp_path, kind),
        kind,
        """
#include <assert.h>
int32_t read_row(const int32_t (*p)[3]) { return p[1][2]; }
int main(void) {
    int32_t rows[2][3] = {{0}, {0, 0, -123}};
    assert(read(rows) == -123);
    return 0;
}
""",
    )
    assert subprocess.run([str(executable)], check=False).returncode == 0
