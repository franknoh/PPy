"""The first build's shortcuts: a runtime object compiled once, passes verified
where they can break, a module's LLVM IR emitted only when read, and the warm
lookup made once per run."""

from __future__ import annotations

import os
import shutil
import stat
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import NativeModule
from ppy_compiler.backend.llvm.link import runtime_object
from ppy_compiler.ir import (
    I64,
    Builder,
    IRModule,
    Pass,
    PassContext,
    PassManager,
    PassVerificationError,
)
from ppy_compiler.ir.dialects import core

# -- the runtime object ---------------------------------------------------------


def _compiler() -> str:
    found = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if found is None:
        pytest.skip("no C compiler")
    return found


def _counting(tmp_path: Path, real: str) -> tuple[str, Path]:
    """A compiler that records each call, then runs the real one."""
    calls = tmp_path / "calls"
    script = tmp_path / "counting-cc"
    script.write_text(f'#!/bin/sh\necho x >> "{calls}"\nexec "{real}" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script), calls


@pytest.fixture
def user_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache" / "ppy" / "runtime-objects"


def _runtime(tmp_path: Path, text: str = "int ppy_answer(void) { return 42; }\n") -> Path:
    source = tmp_path / "runtime.c"
    source.write_text(text)
    return source


def test_a_runtime_is_compiled_once_and_reused(tmp_path, user_cache):
    source = _runtime(tmp_path)
    compiler, calls = _counting(tmp_path, _compiler())
    first = runtime_object(source, compiler)
    assert first.parent == user_cache and first.suffix == ".o" and first.is_file()
    assert runtime_object(source, compiler) == first
    assert calls.read_text().count("x") == 1
    # No draft is left behind.
    assert [p.name for p in user_cache.iterdir()] == [first.name]


def test_a_runtime_object_is_named_for_its_compiler(tmp_path, user_cache):
    source = _runtime(tmp_path)
    real = _compiler()
    wrapped, _calls = _counting(tmp_path, real)
    assert runtime_object(source, real) != runtime_object(source, wrapped)


def test_a_changed_runtime_or_header_builds_a_new_object(tmp_path, user_cache):
    header = tmp_path / "runtime.h"
    header.write_text("#define ANSWER 42\n")
    source = _runtime(tmp_path, '#include "runtime.h"\nint ppy_answer(void) { return ANSWER; }\n')
    compiler = _compiler()
    first = runtime_object(source, compiler)
    header.write_text("#define ANSWER 43\n")
    second = runtime_object(source, compiler)
    source.write_text('#include "runtime.h"\nint ppy_answer(void) { return -ANSWER; }\n')
    third = runtime_object(source, compiler)
    assert len({first, second, third}) == 3


def test_a_runtime_that_cannot_be_compiled_is_linked_as_source(tmp_path, user_cache):
    source = _runtime(tmp_path, "this is not C\n")
    assert runtime_object(source, _compiler()) == source
    assert not list(user_cache.iterdir())


def test_a_runtime_is_linked_as_source_when_the_cache_cannot_be_written(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    blocked.write_text("")  # a file where the cache directory would go
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocked))
    source = _runtime(tmp_path)
    assert runtime_object(source, _compiler()) == source


# -- verification between passes ------------------------------------------------


class _Breaker(Pass):
    name = "breaker"

    def run(self, module, ctx):
        module.functions["f"].entry.operations[-1].erase()
        return True


def _module() -> IRModule:
    module = IRModule("m")
    function = module.add_function("f", [("x", I64)], [I64])
    entry = function.add_entry_block()
    core.ret(Builder(entry), entry.arguments[0])
    return module


def test_the_pipelines_own_passes_are_not_verified_one_by_one():
    manager = PassManager(PassContext()).add(_Breaker())
    manager.run(_module())  # the caller verifies once at the end


def test_every_pass_is_verified_when_asked():
    manager = PassManager(PassContext(verify_after_each=True)).add(_Breaker())
    with pytest.raises(PassVerificationError, match="breaker"):
        manager.run(_module())


def test_a_pass_added_at_a_stage_is_verified_as_it_runs():
    manager = PassManager(PassContext())
    manager.add_stage("before-backend")
    manager.register_stage_pass("before-backend", _Breaker)
    with pytest.raises(PassVerificationError, match="breaker"):
        manager.run(_module())


# -- a module's own LLVM IR -----------------------------------------------------


def test_a_module_emits_its_llvm_ir_once_when_first_read():
    emitted: list[str] = []

    def emit() -> str:
        emitted.append("m")
        return "; module m"

    module = NativeModule("m", emitter=emit)
    assert not emitted
    assert module.ir == "; module m"
    assert module.ir == "; module m"
    assert emitted == ["m"]


def test_setting_a_modules_llvm_ir_replaces_its_emitter():
    module = NativeModule("m", emitter=lambda: pytest.fail("emitted"))
    module.ir = "; given"
    assert module.ir == "; given"


# -- the warm lookup ------------------------------------------------------------


def test_a_run_that_builds_looks_for_a_warm_build_once(tmp_path, monkeypatch):
    from ppy_compiler.driver import warm
    from ppy_compiler.driver.cli import main

    looked: list[Path] = []
    real = warm.locate

    def counting(file, options):  # type: ignore[no-untyped-def]
        looked.append(file)
        return real(file, options)

    monkeypatch.setattr(warm, "locate", counting)
    monkeypatch.chdir(tmp_path)
    Path("prog.py").write_text(
        textwrap.dedent(
            """
            def total(n: int) -> int:
                s = 0
                for i in range(n):
                    s += i
                return s

            print(total(10))
            """
        ),
        encoding="utf-8",
    )
    assert main(["run", "prog.py"]) == 0
    assert len(looked) == 1
    assert os.path.isdir(".ppy-cache")
