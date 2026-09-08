"""`ppy.aio`: coroutines under asyncio, on the IR, and on the native async runtime."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy import aio
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.ir import decode, verify
from ppy_runtime import aio as runtime

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_runtime = pytest.mark.skipif(
    not (llvm_available() and runtime.available()), reason=f"no async runtime: {runtime.reason()}"
)

PROGRAM = """
    import ppy
    from ppy import aio, native


    async def wait_and_double(n: int) -> int:
        total = 0
        for _ in range(2):
            await aio.sleep(0.005)
            total = total + n
        return total


    async def echo_once(listening: int) -> int:
        client = await aio.accept(listening)
        room = native.stack_alloc[ppy.u8](8)
        got = await aio.read(client, room, 8)
        sent = await aio.write(client, room, got)
        aio.close(client)
        return sent


    async def send_hello(port: int) -> int:
        fd = await aio.connect("127.0.0.1", port)
        message = native.stack_alloc[ppy.u8](3)
        native.store(message, 104)
        native.store(native.offset(message, 1), 105)
        native.store(native.offset(message, 2), 33)
        await aio.write(fd, message, 3)
        back = native.stack_alloc[ppy.u8](8)
        got = await aio.read(fd, back, 8)
        aio.close(fd)
        return got * 100 + native.load(native.offset(back, 2))


    async def main() -> int:
        doubled = await wait_and_double(21)
        listening = aio.listen("127.0.0.1", 0)
        port = aio.port(listening)
        served = aio.spawn(echo_once(listening))
        got = await send_hello(port)
        sent = await served
        aio.close(listening)
        return doubled * 1000000 + got * 1000 + sent


    async def divide(n: int) -> int:
        await aio.sleep(0.0)
        return 100 // n


    def program() -> int:
        return aio.run(main())
    """


def _program() -> dict:  # type: ignore[type-arg]
    namespace: dict = {}
    exec(textwrap.dedent(PROGRAM), namespace)  # the program under test
    return namespace


def test_the_asyncio_reference_sleeps_serves_and_answers_errors():
    program = _program()
    assert program["program"]() == 42333003
    assert aio.run(program["divide"](4)) == 25
    with pytest.raises(ZeroDivisionError):
        aio.run(program["divide"](0))
    first = aio.listen("127.0.0.1", 0)
    assert aio.listen("127.0.0.1", aio.port(first)) < 0, "a taken port is a negative errno"
    aio.close(first)
    assert aio.port(12345) < 0 and aio.run(aio.accept(12345)) < 0
    assert not aio.compiled(program["main"])


def test_the_checker_types_the_vocabulary_and_names_a_misuse(write, codes):
    assert codes(write("aio_ok.ppy", PROGRAM), backend="llvm") == []
    misused = write(
        "aio_bad.ppy",
        """
        import ppy
        from ppy import aio, native


        async def bad(fd: int, room: native.ptr[float]) -> int:
            await aio.sleep("soon")
            n = await aio.read(fd, room, 4)
            aio.close(fd, fd)
            aio.frobnicate()
            return n


        def host() -> int:
            return aio.run(3)
        """,
    )
    assert codes(misused, backend="llvm") == ["E1301", "E1645", "E1645", "E1645", "E1645"]


@requires_llvm
def test_a_coroutine_lowers_to_a_starter_and_a_resume_function(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "aio_prog.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    emitted = _ppy(tmp_path, "emit", "ir", "aio_prog.ppy")
    assert emitted.returncode == 0, emitted.stderr
    text = emitted.stdout
    if not runtime.available():
        assert "ppy.async" not in text, "without the runtime a coroutine stays in Python"
        return
    assert "func @aio_prog_main() -> future<i64> attrs" in text and "ppy.async = true" in text
    assert 'func @aio_prog_main_resume(%frame: ptr<i64>) -> () attrs {ppy.abi = "resume"' in text
    assert "async.frame_new {slots = " in text and "async.spawn %" in text
    assert "async.suspend %frame, %" in text and "async.result %frame : i64" in text
    assert "async.sleep %" in text and "async.accept %" in text and "async.connect %" in text
    assert "async.start %" in text and "async.complete %frame, %" in text
    assert "async.fail %frame {code = 1, label = " in text, "the division's guard fails the future"
    assert "core.guard" not in text.split("func @aio_prog_divide_resume")[1].split("\n}\n")[0]
    assert not verify(decode(text))
    assert "func @aio_prog_program" not in text, "`aio.run` keeps its caller in Python"
    c = _ppy(tmp_path, "emit", "c", "aio_prog.ppy")
    assert c.returncode == 0, c.stderr
    assert "static void ppy_aio_prog_main_resume(int64_t *a0)" in c.stdout
    assert "ppy_aio_spawn(" in c.stdout and "ppy_aio_await(" in c.stdout
    assert "/* compile with: the async runtime, ppy_runtime/aio/ppy_aio.c */" in c.stdout
    assert "-lppy_aio" not in c.stdout


IR_ROAD = '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n'


@requires_runtime
def test_the_native_runtime_runs_the_program_the_same(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(IR_ROAD, encoding="utf-8")
    entry = tmp_path / "aio_run.ppy"
    entry.write_text(
        textwrap.dedent(PROGRAM).lstrip("\n")
        + textwrap.dedent(
            """

            print(aio.compiled(main), program(), aio.run(divide(4)))
            try:
                aio.run(divide(0))
            except ZeroDivisionError:
                print("python divides")
            except aio.NativeGuardFailed as failure:
                print("native guard:", "divide" in str(failure))
            """
        ),
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, entry.name], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    native_run = _ppy(tmp_path, "run", entry.name)
    assert plain.returncode == 0, plain.stderr
    assert native_run.returncode == 0, native_run.stderr
    assert plain.stdout == "False 42333003 25\npython divides\n"
    assert native_run.stdout == "True 42333003 25\nnative guard: True\n"


@requires_runtime
def test_a_native_future_is_awaited_from_asyncio_and_a_built_artifact_carries_the_runtime(
    tmp_path: Path,
):
    (tmp_path / "pyproject.toml").write_text(IR_ROAD, encoding="utf-8")
    entry = tmp_path / "aio_mixed.ppy"
    entry.write_text(
        textwrap.dedent(PROGRAM).lstrip("\n")
        + textwrap.dedent(
            """


            async def outer() -> int:
                first = await wait_and_double(2)
                await aio.sleep(0.001)
                return first + await main()


            print(aio.run(outer()))
            """
        ),
        encoding="utf-8",
    )
    ran = _ppy(tmp_path, "run", entry.name)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == "42333007\n"
    built = _ppy(tmp_path, "build", "aio_mixed.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    prebuilt = _ppy(tmp_path, "run", "--prebuilt", "dist/ppy-bindings.json", "aio_mixed.ppy")
    assert prebuilt.returncode == 0, prebuilt.stderr
    assert prebuilt.stdout == "42333007\n"


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_ppy_loads_neither_asyncio_nor_the_async_runtime():
    """A program that never awaits pays nothing for `ppy.aio` at import."""
    probe = (
        "import sys\n"
        "import ppy\n"
        "from ppy import aio\n"
        "loaded = sorted(m for m in sys.modules if m in {'asyncio', 'socket', 'ppy_runtime.aio'})\n"
        "print(','.join(loaded))\n"
        "print(aio.NativeFuture.__name__, 'asyncio' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    first, second = done.stdout.splitlines()[:2]
    assert first == "", f"`import ppy` loaded {first}"
    assert second == "NativeFuture False", second
