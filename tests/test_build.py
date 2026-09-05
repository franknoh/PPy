"""Native object emission, linking, launcher, and manifest (spec 4.2, 16.3, 26.2)."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import standalone_toolchain_status, toolchain_status
from ppy_compiler.backend.llvm.wrapper_build import wrapper_toolchain

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_usable, _detail = toolchain_status()
requires_toolchain = pytest.mark.skipif(not _usable, reason=f"no native toolchain: {_detail}")
#: A standalone build embeds no interpreter, so it asks for less: an
#: installation with a static or header-less CPython still builds one, and
#: skipping these for a missing libpython would skip exactly the case they
#: exist to cover.
_standalone, _standalone_detail = standalone_toolchain_status()
requires_c_compiler = pytest.mark.skipif(
    not _standalone, reason=f"no C compiler: {_standalone_detail}"
)
#: A generated wrapper is a CPython extension: a compiler and `Python.h`, and
#: no libpython, since the interpreter loading it already supplies the symbols.
_wrapper, _wrapper_detail = wrapper_toolchain()
requires_wrapper_toolchain = pytest.mark.skipif(
    not _wrapper, reason=f"no wrapper toolchain: {_wrapper_detail}"
)

pytestmark = requires_llvm

SOURCE = """
import ppy
from ppy import f64


@ppy.pure
@ppy.opt(3)
def distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64:
    dx: f64 = x2 - x1
    dy: f64 = y2 - y1
    return (dx * dx + dy * dy) ** 0.5


@ppy.pure
def total(xs: list[int]) -> int:
    result: int = 0
    for x in xs:
        result += x
    return result


def main() -> None:
    print(round(distance(0.0, 0.0, 3.0, 4.0), 6), total([1, 2, 3]))


if __name__ == "__main__":
    main()
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "app.ppy").write_text(textwrap.dedent(SOURCE).lstrip("\n"), encoding="utf-8")
    return tmp_path


def _build(project: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "app.ppy", "--backend", "llvm", *extra],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_writes_a_binding_manifest(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr

    manifest = json.loads((project / ".ppy-cache" / "native" / "ppy-bindings.json").read_text())
    assert manifest["abi_version"] == 1
    assert manifest["calling_convention"] == "c"

    entries = {entry["python_qualname"]: entry for entry in manifest["entries"]}
    # `distance` is native-eligible but too small to pay for the boundary;
    # the manifest lists only what Python callers should cross into.
    assert "app.distance" not in entries
    assert "app.total" in entries

    total = entries["app.total"]
    assert total["native_symbol"] == "ppy_app_total"
    assert total["returns"]["passed_as"] == "out_parameter"
    assert "re-run the Python implementation" in total["status"]["meaning"]

    buffer = entries["app.total"]["arguments"][0]
    assert buffer["semantic_type"] == "list[int]"
    assert buffer["ownership"] == "borrowed"


@requires_toolchain
def test_build_ships_the_python_abi_wrapper(project: Path):
    """The artifact carries the compiled boundary, so a launcher never pays ctypes."""
    result = _build(project)
    assert result.returncode == 0, result.stderr

    native = project / ".ppy-cache" / "native"
    manifest = json.loads((native / "ppy-bindings.json").read_text())
    wrappers = manifest["wrappers"]
    assert wrappers is not None
    assert (native / wrappers["library"]).is_file()
    assert "app.total" in wrappers["entries"]


@requires_toolchain
def test_build_emits_objects_and_links_a_library(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr

    native = project / ".ppy-cache" / "native"
    objects = list(native.glob("*.o"))
    # The boundary wrapper is a `.so` here too, so the project library is
    # named, not globbed for.
    libraries = list(native.glob("libppy_*.so"))
    assert objects and libraries
    assert libraries[0].stat().st_size > 0

    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(libraries[0])],
        capture_output=True,
        text=True,
        check=False,
    )
    if symbols.returncode == 0:
        assert "ppy_app_distance" in symbols.stdout


@requires_toolchain
def test_build_produces_a_runnable_native_executable(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr

    launcher = project / ".ppy-cache" / "native" / "app"
    assert launcher.is_file()
    assert launcher.stat().st_mode & 0o111

    native = subprocess.run(
        [str(launcher)], cwd=project, capture_output=True, text=True, check=False
    )
    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


@requires_toolchain
def test_the_launcher_forwards_program_arguments(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "args.ppy").write_text("import sys\n\nprint(sys.argv[1:])\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "args.ppy", "--backend", "llvm"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    launcher = tmp_path / ".ppy-cache" / "native" / "args"
    if not launcher.is_file():
        pytest.skip("no launcher was produced")
    native = subprocess.run(
        [str(launcher), "one", "two"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert native.stdout.strip() == "['one', 'two']"


@requires_toolchain
def test_the_launcher_runs_the_prebuilt_library_not_a_jit(project: Path):
    """The binary is `ppy run` in a compiled coat, fed by the built library.

    Deleting the library must break the launcher: a launcher that still
    worked would be quietly recompiling or, worse, quietly interpreting.
    """
    result = _build(project)
    assert result.returncode == 0, result.stderr
    native = project / ".ppy-cache" / "native"
    launcher = native / "app"
    if not launcher.is_file():
        pytest.skip("no launcher was produced")

    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    ran = subprocess.run([str(launcher)], cwd=project, capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == plain.stdout

    library = next(native.glob("libppy_*.so"))
    library.unlink()
    broken = subprocess.run(
        [str(launcher)], cwd=project, capture_output=True, text=True, check=False
    )
    assert broken.returncode != 0
    assert "E1801" in broken.stderr


@requires_toolchain
def test_run_binds_from_a_prebuilt_manifest(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr
    manifest = project / ".ppy-cache" / "native" / "ppy-bindings.json"

    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    ran = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", "--prebuilt", str(manifest), "app.ppy"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == plain.stdout


@requires_toolchain
def test_build_defaults_to_wrap_semantics_and_safe_restores_python_integers(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "spin.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            def spin(n: int) -> int:
                value: int = 3
                for _i in range(n):
                    value = value * 2654435761 + 1
                return value


            print(spin(41))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, "spin.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )

    def built(*extra: str) -> str:
        out = tmp_path / ("safe" if extra else "fast")
        result = subprocess.run(
            [sys.executable, "-m", "ppy_compiler", "build", *extra, "spin.ppy", "-o", str(out)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        launcher = out / "spin"
        if not launcher.is_file():
            pytest.skip("no launcher was produced")
        ran = subprocess.run(
            [str(launcher)], cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert ran.returncode == 0, ran.stderr
        return ran.stdout

    assert built().strip() == "8606135309836935036"
    assert built("--safe") == plain.stdout


@requires_toolchain
def test_the_launcher_never_imports_the_compiler(project: Path, tmp_path: Path):
    """A built artifact is compiled software: `ppy_compiler` may be gone.

    The compiler package is poisoned via PYTHONPATH; if any part of the
    runtime path imported it, the launcher would die on the spot.
    """
    result = _build(project)
    assert result.returncode == 0, result.stderr
    launcher = project / ".ppy-cache" / "native" / "app"
    if not launcher.is_file():
        pytest.skip("no launcher was produced")
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "ppy_compiler.py").write_text(
        'raise ImportError("a built artifact must not import the compiler")\n',
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    environment = dict(os.environ, PYTHONPATH=str(poison))
    ran = subprocess.run(
        [str(launcher)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == plain.stdout


@requires_toolchain
def test_the_manifest_records_the_program_and_the_abi(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((project / ".ppy-cache" / "native" / "ppy-bindings.json").read_text())
    program = manifest["program"]
    assert program["entry"] == "app"
    assert "app" in program["generated"]
    entry = next(e for e in manifest["entries"] if e["python_qualname"] == "app.total")
    assert entry["module"] == "app"
    assert entry["binding"] == "total"
    assert entry["abi"]["returns"] == ["i64"]
    assert [p["kind"] for p in entry["abi"]["parameters"]] == ["list"]


STANDALONE = """
import ppy


@ppy.pure
def collatz(limit: int) -> int:
    best: int = 0
    for start in range(1, limit):
        n: int = start
        steps: int = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best


def main() -> None:
    print("longest below", 300000, "->", collatz(300000))
    print("flag:", True)


main()
"""


@requires_c_compiler
def test_standalone_builds_a_python_free_native_executable(tmp_path: Path):
    """`--standalone`: no CPython inside, same bytes out.

    The strongest proof of independence available portably: the produced
    binary runs with an empty environment, from a directory with no project,
    and prints exactly what plain CPython prints.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "app.ppy").write_text(textwrap.dedent(STANDALONE).lstrip("\n"), encoding="utf-8")
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "--standalone", "app.ppy", "-o", "out"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    ran = subprocess.run(
        [str(tmp_path / "out" / "app")],
        cwd="/",
        capture_output=True,
        text=True,
        check=False,
        env={},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == plain.stdout


@requires_c_compiler
def test_standalone_reads_input_without_an_interpreter(tmp_path: Path):
    """`ppy.input[int]()` is the C reader when there is no CPython to ask."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    (tmp_path / "doubled.ppy").write_text(
        textwrap.dedent(
            """
            \"\"\"Reads a number and prints twice it.\"\"\"

            import ppy


            @ppy.pure
            def twice(value: int) -> int:
                return value + value


            def main() -> None:
                print(twice(ppy.input[int]()))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "build",
            "--standalone",
            "doubled.ppy",
            "-o",
            "dist",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    binary = tmp_path / "dist" / "doubled"
    assert binary.is_file()

    ran = subprocess.run(
        [str(binary)], input="21\n", capture_output=True, text=True, check=False, env={}
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "42"

    linked = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=False)
    if linked.returncode == 0:
        assert "libpython" not in linked.stdout


@requires_c_compiler
def test_standalone_rejects_a_python_reachable_graph(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "floaty.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            def scaled(x: float) -> float:
                return x * 2.0


            def main() -> None:
                print(scaled(1.5))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "--standalone", "floaty.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode != 0
    assert "E1803" in built.stderr
    assert "floaty.main" in built.stderr


def test_build_honours_an_explicit_output_directory(project: Path, tmp_path: Path):
    destination = tmp_path / "out"
    result = _build(project, "-o", str(destination))
    assert result.returncode == 0, result.stderr
    assert (destination / "ppy-bindings.json").is_file()


def test_doctor_reports_the_native_toolchain(project: Path):
    result = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "doctor"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "native toolchain" in result.stdout


@requires_c_compiler
def test_standalone_allocates_and_fills_its_own_buffers(tmp_path: Path):
    """No `array.array` to make, so the program makes the memory itself."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    (tmp_path / "total.ppy").write_text(
        textwrap.dedent(
            """
            \"\"\"Sums the numbers it is given.\"\"\"

            import ppy
            from ppy import Buffer


            @ppy.pure
            @ppy.opt(3)
            def total(xs: Buffer[int]) -> int:
                out: int = 0
                for i in range(len(xs)):
                    out += xs[i]
                return out


            @ppy.opt(3)
            def doubled(xs: Buffer[int], into: Buffer[int]) -> int:
                for i in range(len(xs)):
                    into[i] = xs[i] + xs[i]
                return len(xs)


            def main() -> None:
                count: int = ppy.input[int]()
                values: Buffer[int] = ppy.input[Buffer[int]](count)
                room: Buffer[int] = ppy.buffer[int](count)
                doubled(values, room)
                print(total(room))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "--standalone", "total.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr

    ran = subprocess.run(
        [str(tmp_path / "dist" / "total")],
        input="4\n1 2 3 4\n",
        capture_output=True,
        text=True,
        check=False,
        env={},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "20"


@requires_c_compiler
def test_standalone_prints_a_byte_element_without_widening_it_by_hand(tmp_path: Path):
    """A byte element is storage, so reading one hands out an `int`.

    `print(b[i])` and `print(b[i] + 0)` are the same expression; a build that
    took only the second would be making the width part of the type.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    (tmp_path / "bytes.ppy").write_text(
        textwrap.dedent(
            """
            \"\"\"Reads bytes back out of a byte buffer.\"\"\"

            import ppy
            from ppy import Buffer


            def main() -> None:
                signed: Buffer[ppy.i8] = ppy.buffer[ppy.i8](2)
                unsigned: Buffer[ppy.u8] = ppy.buffer[ppy.u8](2)
                signed[0] = -128
                unsigned[0] = 250
                unsigned[1] = 250
                print(signed[0])
                print(unsigned[0] + 0)
                print(unsigned[0] + unsigned[1])


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "--standalone", "bytes.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr

    ran = subprocess.run(
        [str(tmp_path / "dist" / "bytes")], capture_output=True, text=True, check=False, env={}
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.split() == ["-128", "250", "500"]


#: One builder, run as its own process: analyze, say so, then wait for the
#: word and compile into the cache every sibling is also writing. The
#: handshake is what makes them overlap -- analysis takes a second or two and
#: never takes the same time twice.
_BUILDER = """
import json, os, sys, time
from pathlib import Path

from ppy_compiler.driver.pipeline import analyze_paths, open_project
from ppy_compiler.backend.llvm import _collect
from ppy_compiler.backend.llvm.wrapper_build import build_wrappers

source, cache, gate = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
stagger = float(sys.argv[4])
bundle = analyze_paths(open_project(source), [source], backend="llvm")
module = _collect(bundle)[source.stem]
signatures = {q: lowered.signature for q, lowered in module.functions.items()}
(gate.parent / f"ready.{os.getpid()}").touch()
deadline = time.time() + 60
while not gate.is_file() and time.time() < deadline:
    time.sleep(0.005)
# Staggered, not simultaneous: the window is one process loading the library
# while a later one is still writing it, which needs them out of step.
time.sleep(stagger)
built = build_wrappers(source.stem, signatures, cache)
print(json.dumps({"ok": built.ok, "reason": built.reason, "path": str(built.path)}))
"""

_SHARED = """
import ppy


@ppy.pure
@ppy.opt(3)
def collatz(limit: int) -> int:
    best: int = 0
    for start in range(1, limit):
        n: int = start
        steps: int = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        if steps > best:
            best = steps
    return best
"""


@requires_wrapper_toolchain
def test_separate_processes_build_one_wrapper_without_tearing_it(tmp_path: Path):
    """The failure this guards against was between processes, not threads.

    Several compilations of one wrapper is the ordinary case -- a fresh cache
    and a parallel run -- and they agree on the file name because it is the
    source's digest. Each has to publish by rename, or a sibling loads a
    library still being written.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    source = tmp_path / "shared.ppy"
    source.write_text(textwrap.dedent(_SHARED).lstrip("\n"), encoding="utf-8")
    builder = tmp_path / "builder.py"
    builder.write_text(textwrap.dedent(_BUILDER).lstrip("\n"), encoding="utf-8")
    cache = tmp_path / "shared-cache"
    gate = tmp_path / "go"

    workers = 6
    with contextlib.ExitStack() as stack:
        running = [
            stack.enter_context(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(builder),
                        str(source),
                        str(cache),
                        str(gate),
                        str(index * 0.08),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmp_path,
                )
            )
            for index in range(workers)
        ]
        deadline = time.time() + 300
        while len(list(tmp_path.glob("ready.*"))) < workers and time.time() < deadline:
            time.sleep(0.05)
        gate.touch()
        done = [(process.communicate(timeout=300), process.returncode) for process in running]

    for (_out, err), code in done:
        assert code == 0, err
    results = [json.loads(out.strip().splitlines()[-1]) for (out, _err), _code in done]
    refused = [result["reason"] for result in results if not result["ok"]]
    assert not refused, refused
    assert len({result["path"] for result in results}) == 1, "one name, one library"
    assert not list((cache / "wrappers").glob("*.part*")), "a draft was left behind"


@requires_c_compiler
def test_standalone_takes_a_buffer_main_filled_and_handed_on(tmp_path: Path):
    """Writing memory you allocated and then passing it is one program.

    Purity cares that the write became visible to the callee; a standalone
    build cares whether the write needs CPython, and it does not -- the
    memory is a native allocation and the callee is native code in the same
    binary. Reusing the first answer for the second refused this outright.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    (tmp_path / "filled.ppy").write_text(
        textwrap.dedent(
            """
            \"\"\"Fills a buffer here and sums it there.\"\"\"

            import ppy
            from ppy import Buffer


            @ppy.pure
            @ppy.opt(3)
            def total(xs: Buffer[int]) -> int:
                out: int = 0
                for i in range(len(xs)):
                    out += xs[i]
                return out


            def main() -> None:
                room: Buffer[int] = ppy.buffer[int](4)
                room[0] = 10
                room[3] = 32
                print(total(room))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "--standalone", "filled.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr

    ran = subprocess.run(
        [str(tmp_path / "dist" / "filled")], capture_output=True, text=True, check=False, env={}
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "42"

    plain = subprocess.run(
        [sys.executable, "filled.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert plain.stdout == ran.stdout, "and CPython says the same"


_SQUARES = """
import ppy


@ppy.pure
@ppy.opt(3)
def total(n: int) -> int:
    out: int = 0
    for i in range(n):
        out += i * i
    return out


def main() -> None:
    print(total(1000))


main()
"""


def _run(tmp_path: Path, *args: str, importtime: bool = False) -> subprocess.CompletedProcess:
    command = [sys.executable, *(["-X", "importtime"] if importtime else []), "-m", "ppy_compiler"]
    return subprocess.run(
        [*command, "run", *args], cwd=tmp_path, capture_output=True, text=True, check=False
    )


def _project(tmp_path: Path, name: str, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    path = tmp_path / name
    path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return path


@requires_toolchain
def test_a_warm_run_launches_the_last_build_without_the_compiler(tmp_path: Path):
    """The second run of one program imports neither LLVM nor the analyzer.

    A warm run used to be a cold run minus code generation: the compiler was
    imported, LLVM initialized, and the program analyzed again to find out
    that nothing had changed. Now it is the launcher, and the imports say so.
    """
    _project(tmp_path, "squares.ppy", _SQUARES)
    cold = _run(tmp_path, "squares.ppy")
    assert cold.returncode == 0, cold.stderr
    assert cold.stdout.strip() == "332833500"
    runs = list((tmp_path / ".ppy-cache" / "run").iterdir())
    assert len(runs) == 1 and (runs[0] / "ppy-bindings.json").is_file()

    warm = _run(tmp_path, "squares.ppy", importtime=True)
    assert warm.returncode == 0, warm.stderr
    assert warm.stdout.strip() == "332833500"
    imported = {line.split("|")[-1].strip() for line in warm.stderr.splitlines() if "|" in line}
    heavy = {
        name
        for name in imported
        if name.startswith(("llvmlite", "ppy_compiler.analysis", "ppy_compiler.backend", "libcst"))
        or name in {"ppy_compiler.driver.commands", "ppy_compiler.driver.pipeline"}
    }
    assert not heavy, f"a warm run imported the compiler: {sorted(heavy)[:5]}"
    assert "ppy_runtime.launch" in imported, "it ran through the launcher"


@requires_toolchain
def test_an_edited_source_misses_the_warm_cache(tmp_path: Path):
    """The artifact is named by what it was built from; a change is a new name."""
    path = _project(tmp_path, "squares.ppy", _SQUARES)
    assert _run(tmp_path, "squares.ppy").stdout.strip() == "332833500"
    path.write_text(path.read_text(encoding="utf-8").replace("total(1000)", "total(10)"))
    edited = _run(tmp_path, "squares.ppy")
    assert edited.returncode == 0, edited.stderr
    assert edited.stdout.strip() == "285"
    assert len(list((tmp_path / ".ppy-cache" / "run").iterdir())) == 2, (
        "two programs, two artifacts"
    )


@requires_toolchain
def test_the_guard_mode_is_part_of_the_artifacts_name(tmp_path: Path):
    """`ppy run` keeps Python integers and `--unsafe` wraps; they cannot share a build."""
    _project(
        tmp_path,
        "wide.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def grow(n: int) -> int:
            value: int = 1
            for _ in range(n):
                value = value * 3
            return value


        print(grow(45))
        """,
    )
    exact = str(3**45)
    wrapped = str(((3**45) + 2**63) % 2**64 - 2**63)
    for _ in range(2):  # cold, then warm: both must keep their semantics
        assert _run(tmp_path, "wide.ppy").stdout.strip() == exact
        assert _run(tmp_path, "--unsafe", "wide.ppy").stdout.strip() == wrapped
    assert len(list((tmp_path / ".ppy-cache" / "run").iterdir())) == 2


@requires_toolchain
def test_a_program_that_specializes_at_runtime_keeps_the_jit_and_says_so(tmp_path: Path):
    """`@ppy.jit` compiles while the program runs; a launcher cannot, so it is not cached."""
    _project(
        tmp_path,
        "spec.ppy",
        """
        import ppy


        @ppy.jit
        def scale(x: int, k: int) -> int:
            return x * k


        print(sum(scale(i, 3) for i in range(100)))
        """,
    )
    for _ in range(2):
        done = _run(tmp_path, "spec.ppy")
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "14850"
    runs = list((tmp_path / ".ppy-cache" / "run").iterdir())
    assert len(runs) == 1
    assert not (runs[0] / "ppy-bindings.json").exists(), "nothing to launch"
    assert "specializes at runtime" in (runs[0] / "needs-jit").read_text(encoding="utf-8")


@requires_toolchain
def test_a_program_with_nothing_native_still_has_an_artifact(tmp_path: Path):
    """The generated Python alone is a launchable artifact; the library is optional."""
    _project(
        tmp_path,
        "hello.ppy",
        """
        import sys


        def greet(name: str) -> str:
            return f"hello, {name}"


        print(greet(sys.argv[1] if len(sys.argv) > 1 else "world"))
        """,
    )
    assert _run(tmp_path, "hello.ppy", "--", "there").stdout.strip() == "hello, there"
    assert _run(tmp_path, "hello.ppy").stdout.strip() == "hello, world"
    manifest = next((tmp_path / ".ppy-cache" / "run").glob("*/ppy-bindings.json"))
    assert json.loads(manifest.read_text(encoding="utf-8"))["native_library"] is None


NATIVE_IMPORT_KERNEL = """
def total(n: int) -> int:
    acc = 0
    for i in range(n):
        acc += i * i
    return acc
"""

NATIVE_IMPORT_MAIN = """
import ppy
import kernel

print(kernel.total(1000), type(kernel.__loader__).__name__, type(kernel.total).__name__)
print(sorted(ppy.native_imports().items()))
"""


def _python(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )


def test_import_ppy_serves_a_kernel_natively(tmp_path: Path):
    """`import ppy` is the whole integration: a `.ppy` kernel imported from a
    plain `.py` program is built once into the cache and bound natively; the
    next process finds the build; `PPY_IMPORT=python` keeps it source."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "kernel.ppy").write_text(NATIVE_IMPORT_KERNEL.lstrip("\n"), encoding="utf-8")
    (tmp_path / "main.py").write_text(NATIVE_IMPORT_MAIN.lstrip("\n"), encoding="utf-8")
    expected = str(sum(i * i for i in range(1000)))

    first = _python(tmp_path)
    assert first.returncode == 0, first.stderr
    lines = first.stdout.splitlines()
    assert lines[0].split() == [expected, "GeneratedLoader", "builtin_function_or_method"]
    assert lines[1] == "[('kernel', ('total',))]"
    assert "built kernel.ppy natively" in first.stderr
    assert "native: kernel" in first.stderr

    second = _python(tmp_path)
    assert second.stdout == first.stdout
    assert "built" not in second.stderr, "the second process finds the first one's build"

    source = _python(tmp_path, PPY_IMPORT="python")
    assert source.stdout.splitlines() == [f"{expected} PPySourceLoader function", "[]"]
    assert "[ppy]" not in source.stderr

    (tmp_path / "pyproject.toml").write_text(
        "[tool.ppy]\nnative-import = false\n", encoding="utf-8"
    )
    off = _python(tmp_path)
    assert off.stdout.splitlines() == [f"{expected} PPySourceLoader function", "[]"]


def _ppy(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


@requires_toolchain
def test_build_warm_prepares_what_import_ppy_serves(tmp_path: Path):
    """`ppy build --warm` builds ahead what the first import of a kernel would
    build, into the same place, so a program launched on many ranks at once
    finds one artifact instead of each rank building it. A flag that would
    key a different artifact is refused -- nothing would find it -- and a
    kernel that does not check clean is an error before the launch, not a
    note on every rank's stderr during it."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "kernel.ppy").write_text(NATIVE_IMPORT_KERNEL.lstrip("\n"), encoding="utf-8")
    (tmp_path / "main.py").write_text(NATIVE_IMPORT_MAIN.lstrip("\n"), encoding="utf-8")

    warmed = _ppy(tmp_path, "build", "--warm", ".")
    assert warmed.returncode == 0, warmed.stderr
    assert "built kernel.ppy natively" in warmed.stderr
    assert "warm: 1 built, 0 already built" in warmed.stderr

    served = _python(tmp_path)
    assert served.returncode == 0, served.stderr
    assert served.stdout.splitlines()[0].split()[1] == "GeneratedLoader"
    assert "built" not in served.stderr, "the import finds the warm build"

    again = _ppy(tmp_path, "build", "--warm", "kernel.ppy")
    assert again.returncode == 0, again.stderr
    assert "kernel.ppy: already built" in again.stderr
    assert "warm: 0 built, 1 already built" in again.stderr

    refused = _ppy(tmp_path, "build", "--warm", "--safe", "kernel.ppy")
    assert refused.returncode == 2
    assert "--safe" in refused.stderr

    (tmp_path / "broken.ppy").write_text(
        "def wrong(n: int) -> int:\n    return n + missing\n", encoding="utf-8"
    )
    broken = _ppy(tmp_path, "build", "--warm", ".")
    assert broken.returncode == 1
    assert "broken.ppy" in broken.stderr
    assert "1 with check errors" in broken.stderr


def test_a_region_whose_library_is_gone_is_the_python_body(tmp_path: Path):
    """The manifest's regions section names an extension beside it; a missing
    one is the Python body, not a broken artifact -- the region only ever
    removed Python round trips -- and the runtime needs no compiler and no
    torch to decide that."""
    from ppy_runtime.launch import PrebuiltBinder
    from ppy_runtime.manifest import load

    payload = {
        "abi_version": 1,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "program": {
            "entry": "m",
            "modules": ["m"],
            "generated": {},
            "search_paths": [],
            "safeguards": "hoisted",
        },
        "entries": [],
        "regions": {"m": {"library": "ppy_torch_0.so", "entries": {"layer": "ppy_region_m_layer"}}},
    }
    path = tmp_path / "ppy-bindings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load(path).regions is None, "no library, no regions"

    (tmp_path / "ppy_torch_0.so").write_bytes(b"not an extension")
    manifest = load(path)
    assert manifest.regions is not None
    assert manifest.regions["m"].entries == {"layer": "ppy_region_m_layer"}
    binder = PrebuiltBinder(manifest, None)
    assert binder.region_names("m") == frozenset({"layer"})
    assert binder.region_names("other") == frozenset()

    def layer(x):
        return x

    assert binder.region("m", "layer", layer) is layer, "an unloadable extension serves Python"
    assert binder.region_bindings[0].routed is False
