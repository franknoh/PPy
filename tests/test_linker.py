"""The IR linker and whole-program optimization: modules into one program, calls across them."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.ir import (
    F64,
    I64,
    U8,
    BufferType,
    Builder,
    IRModule,
    PtrType,
    Successor,
    decode,
    encode,
    verify,
)
from ppy_compiler.ir.dialects import core
from ppy_compiler.ir.linker import LinkError, link
from ppy_compiler.ir.transforms import GlobalDCE, Inline, internalize, whole_program

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")


def _twice(module: IRModule, private: bool = False) -> None:
    twice = module.add_function(
        "pkg_a_twice",
        [("x", I64)],
        [I64],
        visibility="private" if private else "public",
        attributes={"ppy.symbol": "ppy_pkg_a_twice", "ppy.qualname": "pkg.a.twice"},
    )
    b = Builder(twice.add_entry_block())
    x = twice.entry.arguments[0]
    big = core.cmp(b, "gt", x, core.const(b, 10, I64))
    yes = twice.body.add_block("yes")
    no = twice.body.add_block("no")
    core.cond_br(b, big, Successor(yes), Successor(no))
    b = Builder(yes)
    core.ret(b, core.add(b, x, x, overflow="wrap"))
    b = Builder(no)
    core.ret(b, core.mul(b, x, core.const(b, 3, I64), overflow="wrap"))


def _modules() -> tuple[IRModule, IRModule]:
    a = IRModule("pkg.a")
    a.require("math", 1)
    _twice(a)
    a.add_global("ppy.str.0", BufferType(U8), "hello", visibility="private")
    helper = a.add_function("helper", [("x", F64)], [F64], visibility="private")
    b = Builder(helper.add_entry_block())
    core.ret(b, helper.entry.arguments[0])
    a.attributes["ppy.libraries"] = ("m",)
    m = IRModule("pkg.main")
    m.add_function(
        "pkg_a_twice",
        [("x", I64)],
        [I64],
        attributes={
            "ppy.symbol": "ppy_pkg_a_twice",
            "ppy.qualname": "pkg.a.twice",
            "ppy.external": True,
        },
    )
    m.add_global("ppy.str.0", BufferType(U8), "world", visibility="private")
    helper2 = m.add_function("helper", [("x", F64)], [F64], visibility="private")
    b = Builder(helper2.add_entry_block())
    core.ret(b, core.mul(b, helper2.entry.arguments[0], helper2.entry.arguments[0]))
    main = m.add_function(
        "pkg_main_run",
        [("n", I64)],
        [I64],
        attributes={"ppy.symbol": "ppy_pkg_main_run", "ppy.qualname": "pkg.main.run"},
    )
    b = Builder(main.add_entry_block())
    b.create(
        "core.call_intrinsic",
        (),
        (PtrType(U8),),
        {"intrinsic": "ppy.string_data", "symbol": "ppy.str.0"},
    )
    core.call(b, "helper", (core.const(b, 2.0, F64),), (F64,))
    first = core.call(b, "pkg_a_twice", (main.entry.arguments[0],), (I64,)).results[0]
    second = core.call(b, "pkg_a_twice", (core.const(b, 20, I64),), (I64,)).results[0]
    core.ret(b, core.add(b, first, second, overflow="wrap"))
    m.attributes["ppy.libraries"] = ("pthread",)
    return a, m


def test_linking_resolves_declarations_and_renames_private_collisions():
    a, m = _modules()
    linked = link([a, m], "pkg")
    program = linked.module
    assert not verify(program)
    assert list(program.functions) == ["pkg_a_twice", "helper", "pkg_main__helper", "pkg_main_run"]
    assert not program.functions["pkg_a_twice"].is_declaration, "the definition answered"
    assert list(program.globals) == ["ppy.str.0", "pkg_main__ppy.str.0"]
    assert linked.renamed == {
        ("pkg.main", "ppy.str.0"): "pkg_main__ppy.str.0",
        ("pkg.main", "helper"): "pkg_main__helper",
    }
    text = encode(program)
    assert 'symbol = "pkg_main__ppy.str.0"' in text and "{callee = @pkg_main__helper}" in text
    assert program.dialects == {"core": 1, "math": 1}
    assert program.attributes["ppy.libraries"] == ("m", "pthread")
    assert not linked.unresolved


def test_an_unresolved_declaration_is_named_and_a_public_clash_is_refused():
    a, m = _modules()
    gone = m.add_function("pkg_b_gone", [("x", I64)], [I64], attributes={"ppy.external": True})
    b = Builder().before(m.functions["pkg_main_run"].entry.operations[-1])
    core.call(b, "pkg_b_gone", (core.const(b, 1, I64),), (I64,))
    assert link([a, m], "pkg").unresolved == ("pkg_b_gone",)
    del gone
    other = IRModule("pkg.other")
    _twice(other)
    with pytest.raises(LinkError, match="@pkg_a_twice is defined by two modules"):
        link([a, other], "pkg")
    shared_a, shared_b = IRModule("g.a"), IRModule("g.b")
    for module in (shared_a, shared_b):
        instance = module.add_function(
            "largest_int", [("x", I64)], [I64], attributes={"ppy.generic": "g.largest"}
        )
        core.ret(Builder(instance.add_entry_block()), instance.entry.arguments[0])
    linked = link([shared_a, shared_b], "g")
    assert linked.shared == ("largest_int",) and list(linked.module.functions) == ["largest_int"]


@requires_llvm
def test_whole_program_optimization_inlines_and_drops_what_nothing_reaches():
    import ctypes

    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    a, m = _modules()
    program = link([a, m], "pkg").module
    assert internalize(program, {"pkg_main_run"}) == 1, "twice becomes private; run stays public"
    ctx = whole_program(program, {"pkg_main_run"})
    assert not verify(program)
    assert list(program.functions) == ["pkg_main_run"], "helpers and the inlined twice are gone"
    assert any("3 call(s) inlined" in remark for remark in ctx.remarks), ctx.remarks
    assert "core.call %" not in encode(program) and "{callee = " not in encode(program)
    engine = JitEngine(opt_level=2).open()
    engine.add(str(emit_module(program)))
    engine.finalize()
    run = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))(
        engine.address("ppy_pkg_main_run")
    )
    out = ctypes.c_int64(0)
    assert run(4, ctypes.byref(out)) == 0 and out.value == 12 + 40


def test_inlining_leaves_alone_what_it_must():
    from ppy_compiler.ir import PassContext, PassManager

    module = IRModule("edges")
    loop = module.add_function("loop", [("n", I64)], [I64], attributes={"ppy.symbol": "loop"})
    b = Builder(loop.add_entry_block())
    core.ret(b, core.call(b, "loop", (loop.entry.arguments[0],), (I64,)).results[0])
    marked = module.add_function("kept", [("n", I64)], [I64], attributes={"ppy.noinline": True})
    b = Builder(marked.add_entry_block())
    core.ret(b, marked.entry.arguments[0])
    caller = module.add_function("caller", [("n", I64)], [I64])
    b = Builder(caller.add_entry_block())
    core.ret(b, core.call(b, "kept", (caller.entry.arguments[0],), (I64,)).results[0])
    PassManager(PassContext()).add(Inline()).add(GlobalDCE()).run(module)
    assert encode(module).count("core.call") == 2, "recursion and @ppy.noinline stay calls"
    assert list(module.functions) == ["loop", "kept", "caller"], "public functions stay"


PACKAGE = {
    "util.ppy": """
        def twice(x: int) -> int:
            if x > 10:
                return x + x
            return x * 3


        def shout(x: int) -> int:
            print(x)
            return x
        """,
    "app.ppy": """
        from util import shout, twice


        def run(n: int) -> int:
            return twice(n) + twice(20)


        def loud(n: int) -> int:
            return shout(n) + 1


        def main() -> None:
            print(run(4), loud(2))


        main()
        """,
}


def _write(directory: Path) -> None:
    (directory / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
    )
    for name, source in PACKAGE.items():
        (directory / name).write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@requires_llvm
def test_a_call_across_modules_is_a_declaration_the_program_links(tmp_path: Path):
    _write(tmp_path)
    emitted = _ppy(tmp_path, "emit", "ir", "app.ppy")
    assert emitted.returncode == 0, emitted.stderr
    declaration = emitted.stdout.split("module @app")[1]
    assert "extern func @util_twice(i64) -> i64 attrs {" in declaration
    assert 'ppy.external = true, ppy.qualname = "util.twice"' in declaration
    assert "core.call %" in declaration and "{callee = @util_twice}" in declaration
    assert "func @app_loud" not in emitted.stdout, (
        "shout prints, so it stays in Python -- and so does loud"
    )
    linked = _ppy(tmp_path, "emit", "linked-ir", "app.ppy")
    assert linked.returncode == 0, linked.stderr
    program = linked.stdout
    assert program.count("func @util_twice(") == 1 and "extern func" not in program
    assert "{callee = @util_twice}" not in program, "inlined into run"
    assert "func @app_run" in program
    assert not verify(decode(program)), "the linked program reads back and verifies"
    plain = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ppy, runpy; ppy.install(); runpy.run_path('app.ppy', run_name='__main__')",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    ran = _ppy(tmp_path, "run", "app.ppy")
    assert plain.returncode == 0 and ran.returncode == 0, (plain.stderr, ran.stderr)
    assert plain.stdout == ran.stdout == "2\n52 3\n"
    built = _ppy(tmp_path, "build", ".", "-o", "dist")
    assert built.returncode == 0, built.stderr
    assert (tmp_path / "dist" / "program.o").is_file(), "one object for the program"
    assert not (tmp_path / "dist" / "app.o").exists()
