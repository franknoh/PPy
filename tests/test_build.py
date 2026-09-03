"""Native object emission, linking, launcher, and manifest (spec 4.2, 16.3, 26.2)."""

from __future__ import annotations

import contextlib
import json
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
    import os

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
