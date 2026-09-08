"""Profile-guided optimization: counters in, counts back, hot and cold and weights applied."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.from_ir import emit_module
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.driver.profile import Collector, describe
from ppy_compiler.driver.report import categorize
from ppy_compiler.ir import (
    I64,
    Builder,
    IRModule,
    PassContext,
    PassManager,
    Successor,
    decode,
    encode,
    verify,
)
from ppy_compiler.ir.dialects import core
from ppy_compiler.ir.transforms import (
    AnnotateProfile,
    FunctionProfile,
    GlobalDCE,
    Inline,
    Instrument,
    Profile,
    ProfileError,
    cfg_digest,
)
from ppy_compiler.ir.transforms.profile import PROFILE_MAP

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")


def _countdown() -> IRModule:
    """`f(n)`: a loop that counts n down, adding 2 for multiples of 3 and 1 otherwise."""
    module = IRModule("m")
    f = module.add_function(
        "m_f", [("n", I64)], [I64], attributes={"ppy.symbol": "m_f", "ppy.qualname": "m.f"}
    )
    entry = f.add_entry_block()
    head = f.body.add_block("loop.head", [("i", I64), ("acc", I64)])
    body = f.body.add_block("loop.body")
    bump2 = f.body.add_block("bump2")
    bump1 = f.body.add_block("bump1")
    latch = f.body.add_block("next", [("bump", I64)])
    exit_ = f.body.add_block("loop.exit")
    b = Builder(entry)
    core.br(b, Successor(head, [f.entry.arguments[0], core.const(b, 0, I64)]))
    b = Builder(head)
    i = head.arguments[0]
    acc = head.arguments[1]
    core.cond_br(b, core.cmp(b, "gt", i, core.const(b, 0, I64)), Successor(body), Successor(exit_))
    b = Builder(body)
    remainder = core.mod(b, i, core.const(b, 3, I64), overflow="wrap")
    is_three = core.cmp(b, "eq", remainder, core.const(b, 0, I64))
    core.cond_br(b, is_three, Successor(bump2), Successor(bump1))
    b = Builder(bump2)
    core.br(b, Successor(latch, [core.const(b, 2, I64)]))
    b = Builder(bump1)
    core.br(b, Successor(latch, [core.const(b, 1, I64)]))
    b = Builder(latch)
    next_i = core.sub(b, i, core.const(b, 1, I64), overflow="wrap")
    core.br(b, Successor(head, [next_i, core.add(b, acc, latch.arguments[0], overflow="wrap")]))
    core.ret(Builder(exit_), acc)
    return module


def test_instrumenting_counts_blocks_and_taken_edges_and_leaves_a_legend():
    module = _countdown()
    digest = cfg_digest(module.functions["m_f"])
    ctx = PassContext()
    assert PassManager(ctx).add(Instrument()).run(module).changed
    text = encode(module)
    assert text.count("prof.hit ") == 7 and text.count("prof.hit_if ") == 2, text
    assert "prof" in module.dialects and not verify(module)
    legend = json.loads(module.attributes[PROFILE_MAP])
    entry = legend["functions"]["m_f"]
    assert legend["counters"] == 9 and entry["cfg"] == digest and entry["entry"] == "entry"
    assert set(entry["blocks"]) == {
        "entry",
        "loop.head",
        "loop.body",
        "bump2",
        "bump1",
        "next",
        "loop.exit",
    }
    assert set(entry["edges"]) == {"loop.head", "loop.body"}
    assert entry["qualname"] == "m.f"
    again = decode(text)
    assert encode(again) == text, "the instrumented module round-trips through .ppyir"
    assert not PassManager(PassContext()).add(Instrument()).run(module).changed, "once"
    assert any("9 counter(s) instrument 1 function(s)" in r for r in ctx.remarks)


@requires_llvm
def test_the_counters_are_read_back_through_the_engine_and_the_legend():
    module = _countdown()
    PassManager(PassContext()).add(Instrument()).run(module)
    engine = JitEngine(opt_level=2).open()
    engine.add(str(emit_module(module)))
    engine.finalize()
    f = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))(
        engine.address("m_f")
    )
    out = ctypes.c_int64(0)
    assert f(9, ctypes.byref(out)) == 0 and out.value == 3 * 2 + 6 * 1
    assert f(3, ctypes.byref(out)) == 0 and out.value == 2 + 2

    class Native:
        ir = str(emit_module(module))

    profile = Collector().harvest(engine, {"m": Native()}, compiler="test", program="m")
    record = profile.functions["m.f"]
    assert record.calls == 2 and record.symbol == "m_f" and record.blocks["loop.exit"] == 2
    assert record.blocks["loop.head"] == 9 + 1 + 3 + 1 and record.blocks["next"] == 12
    assert record.blocks["bump2"] == 4 and record.blocks["bump1"] == 8
    assert record.branches["loop.head"] == (12, 2), "taken into the body, not taken to the exit"
    assert record.branches["loop.body"] == (4, 8), "4 multiples of 3 among 9..1 and 3..1"


def test_annotating_writes_weights_hotness_and_trip_counts_and_refuses_a_stale_profile():
    module = _countdown()
    f = module.functions["m_f"]
    record = FunctionProfile(
        symbol="m_f",
        cfg=cfg_digest(f),
        calls=2,
        blocks={
            "entry": 2,
            "loop.head": 14,
            "loop.body": 12,
            "bump2": 4,
            "bump1": 8,
            "next": 12,
            "loop.exit": 2,
        },
        branches={"loop.head": (12, 2), "loop.body": (4, 8)},
    )
    profile = Profile(
        functions={"m.f": record, "m.idle": FunctionProfile(cfg="x", calls=0)}, runs=1
    )
    assert profile.hot_threshold() == 2 and profile.kind("m.f") == "hot"
    assert profile.kind("m.idle") == "cold" and profile.kind("m.other") is None
    ctx = PassContext()
    assert PassManager(ctx).add(AnnotateProfile(profile)).run(module).changed
    assert f.attributes["ppy.profile.calls"] == 2 and f.attributes["ppy.profile.hot"] is True
    head = next(b for b in f.body.blocks if b.name == "loop.head")
    body = next(b for b in f.body.blocks if b.name == "loop.body")
    latch = next(b for b in f.body.blocks if b.name == "next")
    assert head.terminator.attributes["ppy.weights"] == (12, 2)
    assert body.terminator.attributes["ppy.weights"] == (4, 8)
    assert body.terminator.attributes["ppy.profile.count"] == 12
    assert latch.terminator.attributes["ppy.profile.trips"] == 6.0, "12 back edges over 2 entries"
    assert any(
        "`m.f` is hot (2 call(s)); 2 branch(es) weighted, 1 loop(s)" in r for r in ctx.remarks
    )
    text = encode(module)
    assert (
        "ppy.weights = [12, 2]" in text
        and decode(text).functions["m_f"].attributes["ppy.profile.hot"]
    ), "the annotations travel through .ppyir"
    if llvm_available():
        llvm = str(emit_module(module))
        assert (
            '!"branch_weights", i32 13, i32 3' in llvm and '!"function_entry_count", i64 2' in llvm
        )
        assert " hot " in llvm or " hot\n" in llvm
    stale = _countdown()
    record.cfg = "0000000000000000"
    ctx = PassContext()
    assert not PassManager(ctx).add(AnnotateProfile(profile)).run(stale).changed
    assert "ppy.profile.calls" not in stale.functions["m_f"].attributes
    assert any("stale for `m.f`" in r for r in ctx.remarks)
    assert categorize(ctx.remarks[0]) == "profile stale"
    assert categorize("profile: `m.f` is hot (2 call(s))") == "profile applied"


def test_the_inliner_reads_the_profile():
    module = IRModule("p")
    big_body = 30

    def function(name: str, **attributes: object):  # type: ignore[no-untyped-def]
        callee = module.add_function(
            name, [("n", I64)], [I64], visibility="private", attributes=dict(attributes)
        )
        b = Builder(callee.add_entry_block())
        value = callee.entry.arguments[0]
        for _ in range(big_body):
            value = core.add(b, value, core.const(b, 1, I64), overflow="wrap")
        core.ret(b, value)
        return callee

    function("hot", **{"ppy.profile.hot": True, "ppy.profile.calls": 500})
    function("cold", **{"ppy.profile.cold": True, "ppy.profile.calls": 0})
    function("warm")
    caller = module.add_function("caller", [("n", I64)], [I64])
    b = Builder(caller.add_entry_block())
    n = caller.entry.arguments[0]
    total = core.call(b, "hot", (n,), (I64,)).results[0]
    total = core.add(b, total, core.call(b, "cold", (n,), (I64,)).results[0], overflow="wrap")
    total = core.add(b, total, core.call(b, "warm", (n,), (I64,)).results[0], overflow="wrap")
    core.ret(b, total)
    ctx = PassContext()
    PassManager(ctx).add(Inline()).add(GlobalDCE()).run(module)
    text = encode(module)
    assert text.count("core.call") == 2, (
        "the hot callee's budget stretched; cold and warm stayed calls"
    )
    assert (
        "cold" in module.functions and "warm" in module.functions and "hot" not in module.functions
    )
    assert any("1 call(s) inlined" in r and "1 left by the profile" in r for r in ctx.remarks), (
        ctx.remarks
    )


def test_describe_names_kinds_shapes_and_schemas():
    numpy = pytest.importorskip("numpy")
    assert describe(3) == "int" and describe(2.5) == "float" and describe(None) == "None"
    assert describe(numpy.zeros((4, 3))) == "ndarray[float64;4x3]"
    assert describe([1, 2, 3]) == "list[3]"
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    assert describe(frame) == "DataFrame[a:int64,b:float64;2 rows]"


def test_profiles_merge_by_graph_and_refuse_what_they_are_not(tmp_path: Path):
    first = Profile(
        functions={
            "m.f": FunctionProfile(cfg="abc", calls=2, blocks={"entry": 2}, branches={"e": (1, 1)})
        },
        runs=1,
    )
    written = first.write(tmp_path / "p.ppyprof")
    assert written.runs == 1 and Profile.load(tmp_path / "p.ppyprof").functions["m.f"].calls == 2
    second = Profile(
        functions={
            "m.f": FunctionProfile(cfg="abc", calls=3, blocks={"entry": 3}, branches={"e": (2, 1)}),
            "m.g": FunctionProfile(cfg="def", calls=1),
        },
        runs=1,
    )
    merged = second.write(tmp_path / "p.ppyprof")
    assert merged.runs == 2 and merged.functions["m.f"].calls == 5
    assert merged.functions["m.f"].branches["e"] == (3, 2) and merged.functions["m.g"].calls == 1
    moved = Profile(functions={"m.f": FunctionProfile(cfg="zzz", calls=7)}, runs=1)
    assert moved.write(tmp_path / "p.ppyprof").functions["m.f"].calls == 7, "a new graph replaces"
    (tmp_path / "bad.ppyprof").write_text('{"kind": "other"}', encoding="utf-8")
    with pytest.raises(ProfileError, match="not a ppy profile"):
        Profile.load(tmp_path / "bad.ppyprof")
    with pytest.raises(ProfileError, match="no profile at"):
        Profile.load(tmp_path / "missing.ppyprof")


PROGRAM = """
    def compute(xs: list[int]) -> int:
        total = 0
        for x in xs:
            if x % 4 == 0:
                total = total + x
            else:
                total = total - 1
        return total


    def unused(n: int) -> int:
        return n * 7


    def main() -> None:
        xs: list[int] = []
        for i in range(400):
            xs.append(i)
        acc = 0
        for _ in range(50):
            acc = acc + compute(xs)
        print(acc)


    main()
    """


def _write(directory: Path) -> None:
    (directory / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
    )
    (directory / "main.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@requires_llvm
def test_a_profiled_run_writes_the_profile_and_a_guided_build_uses_it(tmp_path: Path):
    _write(tmp_path)
    plain = subprocess.run(
        [sys.executable, "main.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    profiled = _ppy(tmp_path, "run", "--profile", "main.ppy")
    assert profiled.returncode == 0, profiled.stderr
    assert profiled.stdout == plain.stdout
    written = tmp_path / "main.ppyprof"
    assert written.is_file() and "profile: main.ppyprof" in profiled.stderr, profiled.stderr
    profile = Profile.load(written)
    compute = profile.functions["main.compute"]
    assert compute.calls == 50 and profile.functions["main.unused"].calls == 0
    assert profile.kind("main.compute") == "hot" and profile.kind("main.unused") == "cold"
    skewed = [pair for pair in compute.branches.values() if pair == (50 * 100, 50 * 300)]
    assert skewed, f"one branch is taken for the multiples of 4 only: {compute.branches}"
    assert compute.arguments["0"] == {"list[400]": 50}
    again = _ppy(tmp_path, "run", "--profile", "main.ppy")
    assert again.returncode == 0 and again.stdout == plain.stdout
    merged = Profile.load(written)
    assert merged.runs == 2 and merged.functions["main.compute"].calls == 100

    built = _ppy(
        tmp_path,
        "build",
        "--pgo",
        "main.ppyprof",
        "main.ppy",
        "--report-opt",
        "--report-opt-json",
        "report.json",
        "-o",
        "dist",
    )
    assert built.returncode == 0, built.stderr
    assert "profile: main.ppyprof (2 runs" in built.stdout, built.stdout
    assert "main.compute: hot, 100 calls" in built.stdout and "main.unused: cold" in built.stdout
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["profile"]["functions"]["main.compute"]["kind"] == "hot"
    categories = report["modules"]["main"]["categories"]
    assert categories.get("profile applied", 0) >= 1, categories
    guided = _ppy(tmp_path, "run", "--pgo", "main.ppyprof", "main.ppy")
    assert guided.returncode == 0, guided.stderr
    assert guided.stdout == plain.stdout, "a profile changes nothing the program computes"
    warm = _ppy(tmp_path, "run", "--pgo", "main.ppyprof", "main.ppy")
    assert warm.returncode == 0 and warm.stdout == plain.stdout

    source = tmp_path / "main.ppy"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "total = total - 1",
            "total = total - 1\n            if x > 398:\n                total = total + 2",
        ),
        encoding="utf-8",
    )
    stale = _ppy(tmp_path, "build", "--pgo", "main.ppyprof", "main.ppy", "-o", "dist2")
    assert stale.returncode == 0, stale.stderr
    assert "W2009" in stale.stderr and "main.compute" in stale.stderr, stale.stderr
    missing = _ppy(tmp_path, "build", "--pgo", "nothing.ppyprof", "main.ppy")
    assert missing.returncode == 2 and "no profile at" in missing.stderr
