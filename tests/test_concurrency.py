"""`ppy.atomic` and `ppy.concurrent`: threads and atomics on every path."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    from ppy import atomic, concurrent, native


    def worker(counter: native.ptr[int], mutex: native.ptr[int], total: native.ptr[int], rounds: int) -> None:
        for _ in range(rounds):
            atomic.fetch_add(counter, 1, order="relaxed")
            concurrent.lock(mutex)
            native.store(total, native.load(total) + 2)
            concurrent.unlock(mutex)


    def waiter(flag: native.ptr[int], mutex: native.ptr[int], condition: native.ptr[int], out: native.ptr[int]) -> None:
        concurrent.lock(mutex)
        while native.load(flag) == 0:
            concurrent.wait(condition, mutex)
        native.store(out, native.load(flag) * 10)
        concurrent.unlock(mutex)


    def run(counter: native.ptr[int], mutex: native.ptr[int], total: native.ptr[int], rounds: int) -> int:
        first = concurrent.spawn(worker, counter, mutex, total, rounds)
        second = concurrent.spawn(worker, counter, mutex, total, rounds)
        third = concurrent.spawn(worker, counter, mutex, total, rounds)
        fourth = concurrent.spawn(worker, counter, mutex, total, rounds)
        concurrent.join(first)
        concurrent.join(second)
        concurrent.join(third)
        concurrent.join(fourth)
        atomic.fence()
        return atomic.load(counter)


    def signal(flag: native.ptr[int], mutex: native.ptr[int], condition: native.ptr[int], out: native.ptr[int]) -> int:
        waiting = concurrent.spawn(waiter, flag, mutex, condition, out)
        concurrent.lock(mutex)
        native.store(flag, 4)
        concurrent.notify(condition)
        concurrent.unlock(mutex)
        concurrent.join(waiting)
        return native.load(out)


    def main() -> None:
        counter = native.stack_alloc[int](1)
        mutex = native.stack_alloc[int](1)
        total = native.stack_alloc[int](1)
        threads = 4
        print(run(counter, mutex, total, 500), native.load(total), threads)
        old, swapped = atomic.compare_exchange(counter, 2000, 7)
        print(old, swapped, atomic.exchange(counter, 1), atomic.load(counter, order="acquire"))
        print(atomic.fetch_or(counter, 6), atomic.fetch_and(counter, 3), atomic.fetch_xor(counter, 1), atomic.fetch_sub(counter, 2), atomic.load(counter))

        flag = native.stack_alloc[int](1)
        condition = native.stack_alloc[int](1)
        out = native.stack_alloc[int](1)
        print(signal(flag, mutex, condition, out), concurrent.thread_id() != 0)


    main()
    """


def _three_paths(tmp_path: Path, source: str) -> tuple[str, str, str]:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
    )
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
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
    return outputs[0], outputs[1], outputs[2]


@requires_llvm
def test_threads_and_atomics_agree_on_every_path(tmp_path: Path):
    plain, python, native = _three_paths(tmp_path, PROGRAM)
    assert plain == python == native
    lines = plain.splitlines()
    assert lines[0] == "2000 4000 4"
    assert lines[1] == "2000 True 7 1"
    assert lines[2] == "1 7 3 2 0"
    assert lines[3] == "40 True"
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "ir", "prog.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    text = emitted.stdout
    assert "dialect concurrency 1" in text and "dialect atomic 1" in text
    assert "concurrency.spawn" in text and "callee = @prog_worker" in text
    assert "func @prog_run" in text and "func @prog_signal" in text, "the spawners lower natively"
    assert "atomic.fetch_add" in text and 'order = "relaxed"' in text
    assert "concurrency.condition_wait" in text and "concurrency.mutex_lock" in text
    assert 'ppy.libraries = ("pthread",)' in text or "pthread" in text


def test_the_reference_implementation_synchronizes_like_the_machine():
    import threading

    from ppy import atomic, concurrent, native

    slot = native.stack_alloc[int](1)
    assert atomic.fetch_add(slot, 5) == 0 and atomic.load(slot) == 5
    assert atomic.compare_exchange(slot, 4, 9) == (5, False)
    assert atomic.compare_exchange(slot, 5, 9) == (5, True)
    byte = native.stack_alloc[__import__("ppy").u8](1)
    assert atomic.fetch_add(byte, 300) == 0 and atomic.load(byte) == 300 % 256
    with pytest.raises(ValueError):
        atomic.load(slot, order="sequential")
    with pytest.raises(ValueError):
        atomic.fence("relaxed")
    mutex = native.stack_alloc[int](1)
    barrier = native.stack_alloc[int](2)
    seen: list[int] = []

    def worker(k: int) -> None:
        concurrent.barrier(barrier, 3)
        concurrent.lock(mutex)
        seen.append(k)
        concurrent.unlock(mutex)

    handles = [concurrent.spawn(worker, k) for k in range(3)]
    for handle in handles:
        concurrent.join(handle)
    assert sorted(seen) == [0, 1, 2]
    assert concurrent.thread_id() == threading.get_ident()

    def failing() -> None:
        raise ValueError("inside")

    with pytest.raises(ValueError, match="inside"):
        concurrent.join(concurrent.spawn(failing))


def test_the_checker_names_what_atomics_and_threads_refuse(write, codes):
    path = write(
        "bad.ppy",
        """
        from ppy import atomic, concurrent, native


        def task(n: int) -> int:
            return n


        def f(p: native.ptr[float], q: native.const_ptr[int], r: native.ptr[int]) -> None:
            atomic.fetch_add(p, 1)
            atomic.store(q, 1)
            atomic.load(r, order="release")
            atomic.load(r, order="strong")
            atomic.fetch_add(r, 1, memory="relaxed")
            concurrent.lock(p)
            concurrent.spawn(task, 1)
            concurrent.spawn(3)
            concurrent.join(5)
            concurrent.barrier(r)
        """,
    )
    found = codes(path)
    assert found.count("E1641") == 5, found
    assert found.count("E1642") >= 5, found
