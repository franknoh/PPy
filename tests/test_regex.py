"""Regular expressions matched natively: the regex dialect, its lowering to a
backtracking matcher, and the `re` surface the frontend lowers."""

from __future__ import annotations

import ctypes
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.ir import BOOL, I64, Builder, IRModule, decode, encode, verify
from ppy_compiler.ir.dialects import core, regex
from ppy_compiler.ir.passes import PassContext
from ppy_compiler.ir.transforms import LowerRegex
from ppy_compiler.ir.types import U8, BufferType

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

# -- the pattern -----------------------------------------------------------------------


def test_a_pattern_is_read_by_cpythons_parser():
    compiled = regex.analyse(rb"(?P<key>\w+)\s*=\s*(\d+)", re.MULTILINE)
    assert compiled.groups == 2 and compiled.names == {"key": 1} and compiled.spans == 6
    assert isinstance(compiled.tree, regex.Seq)
    folded = regex.analyse(rb"(?i)[b-d]").tree
    assert isinstance(folded, regex.Chars)
    assert sorted(chr(c) for c in range(256) if folded.has(c)) == list("BCDbcd")
    assert isinstance(regex.analyse(rb".", re.DOTALL).tree, regex.Chars)


@pytest.mark.parametrize(
    ("pattern", "flags", "reason"),
    [
        (rb"(a)\1", 0, "backreference"),
        (rb"(?=a)b", 0, "lookahead"),
        (rb"(?:ab)*+", 0, "possessive"),
        (rb"(?>ab)", 0, "atomic"),
        (rb"a", re.LOCALE, "re.LOCALE"),
        (rb"(", 0, "does not parse"),
        ("a", 0, "bytes literal"),
    ],
)
def test_what_the_matcher_cannot_reproduce_is_refused(pattern, flags, reason):
    with pytest.raises(regex.Unsupported, match=reason):
        regex.analyse(pattern, flags)  # type: ignore[arg-type]


# -- the operation ---------------------------------------------------------------------


def _module(pattern: bytes, flags: int, mode: str) -> tuple[IRModule, regex.Compiled]:
    module = IRModule("t")
    module.require("regex", 1)
    compiled = regex.analyse(pattern, flags)
    results = [BOOL, *([I64] * compiled.spans)]
    function = module.add_function(
        "t_f", [("buf", BufferType(U8)), ("pos", I64), ("endpos", I64)], results
    )
    entry = function.add_entry_block()
    b = Builder(entry)
    arguments = entry.arguments
    op = regex.create(b, mode, arguments[0], arguments[1], arguments[2], compiled)
    core.ret(b, *op.results)
    return module, compiled


def test_the_operation_verifies_prints_and_reads_back():
    module, _ = _module(rb'(a|")\\d+\n', re.IGNORECASE, "search")
    assert not verify(module)
    text = encode(module)
    assert "dialect regex 1" in text
    assert "regex.search %buf, %pos, %endpos {flags = 2, pattern = " in text
    again = decode(text)
    assert not verify(again)
    assert encode(again) == text
    op = again.functions["t_f"].entry.operations[0]  # type: ignore[union-attr]
    assert regex.pattern_of(op) == (rb'(a|")\\d+\n', re.IGNORECASE)


def test_the_verifier_holds_the_operation_to_its_pattern():
    module = IRModule("t")
    module.require("regex", 1)
    function = module.add_function(
        "t_f", [("buf", BufferType(U8)), ("pos", I64), ("endpos", I64)], [BOOL, I64, I64]
    )
    entry = function.add_entry_block()
    b = Builder(entry)
    op = b.create(
        "regex.search",
        tuple(entry.arguments),
        [BOOL, I64, I64],
        {"pattern": "(a)(b)", "flags": 0},
    )
    core.ret(b, *op.results)
    messages = [error.message for error in verify(module)]
    assert any("yields (bool, i64 x 6)" in m for m in messages)
    op.attributes["pattern"] = "(a)\\1"
    assert any("backreference" in m for m in (e.message for e in verify(module)))


# -- the matcher -----------------------------------------------------------------------


def _lowered(pattern: bytes, flags: int, mode: str) -> tuple[IRModule, regex.Compiled]:
    module, compiled = _module(pattern, flags, mode)
    assert LowerRegex().run(module, PassContext(None))
    errors = verify(module)
    assert not errors, errors
    assert "ppy.regex.0" in module.functions
    assert not any(op.dialect == "regex" for op in module.functions["t_f"].entry.operations)  # type: ignore[union-attr]
    return module, compiled


def test_a_lowered_matcher_is_plain_core_ir():
    module, _ = _lowered(rb"(a+)(b|c)*\d\s*$", 0, "search")
    matcher = module.functions["ppy.regex.0"]
    assert matcher.attributes["ppy.synthesized"] == "regex.search"
    dialects = {op.dialect for block in matcher.body.blocks for op in block.operations}
    assert dialects == {"core"}
    assert any(
        op.name == "core.guard" and op.attributes.get("label") == "regex.stack.ok"
        for block in matcher.body.blocks
        for op in block.operations
    )


class _Runner:
    """A lowered matcher as a C function: (status, found, spans)."""

    def __init__(self, address: int, spans: int) -> None:
        prototype = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int8),
            *([ctypes.POINTER(ctypes.c_int64)] * spans),
        )
        self.function = prototype(address)
        self.spans = spans

    def __call__(self, data: bytes, pos: int, endpos: int) -> tuple[int, bool, list[int]]:
        buffer = (ctypes.c_uint8 * max(len(data), 1))(*data)
        found = ctypes.c_int8(0)
        outs = [ctypes.c_int64(0) for _ in range(self.spans)]
        status = self.function(
            buffer, len(data), pos, endpos, ctypes.byref(found), *[ctypes.byref(o) for o in outs]
        )
        return status, bool(found.value), [o.value for o in outs]


def _jit(module: IRModule, keep: list) -> int:  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    keep.append(engine)
    return engine.address("t_f")


def _compiled(module: IRModule, directory: Path, index: int) -> int:
    from ppy_compiler.backend.c import Language, emit_module

    source = directory / f"t{index}.c"
    source.write_text(emit_module(module, Language.C), encoding="utf-8")
    library = directory / f"libt{index}.so"
    done = subprocess.run(
        [
            c_compiler(),
            "-std=c11",
            "-Wall",
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(library),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0 and "warning:" not in done.stderr, done.stderr
    handle = ctypes.CDLL(str(library))
    return ctypes.cast(handle.t_f, ctypes.c_void_p).value or 0


def _expected(pattern, flags, mode, data, pos, endpos, groups):  # type: ignore[no-untyped-def]
    match = getattr(re.compile(pattern, flags), mode)(data, pos, endpos)
    if match is None:
        return False, [-1] * (2 * (groups + 1))
    spans: list[int] = []
    for group in range(groups + 1):
        spans.extend(match.span(group))
    return True, spans


PATTERNS = [
    (rb"a|ab", 0),
    (rb"(a|ab)(c|bcd)(d*)", 0),
    (rb"(a+)(b|c)*\d\s*$", 0),
    (rb"[^a-z\d]+", re.IGNORECASE),
    (rb"(?P<w>\w+)\b", re.MULTILINE),
    (rb"x{2,5}?y", 0),
    (rb"^\s*(\w+)\s*=\s*(\d+)\s*$", re.MULTILINE),
    (rb"(a*)*b", 0),
    (rb"(a|b)*?c", 0),
    (rb"a.*?b", re.DOTALL),
    (rb"((a)|(b))+", 0),
    (rb"\Ba$", 0),
    (rb"", 0),
]
ALPHABET = b"ab c\n1.A_"


@requires_llvm
@pytest.mark.parametrize("backend", ["llvm", pytest.param("c", marks=requires_cc)])
def test_the_matcher_answers_exactly_what_re_answers(backend, tmp_path: Path):
    """Found or not, and every group's span, on random bytes and bounds."""
    keep: list = []
    rng = random.Random(7)
    for index, (pattern, flags) in enumerate(PATTERNS):
        for mode in regex.MODES:
            module, compiled = _lowered(pattern, flags, mode)
            address = (
                _jit(module, keep)
                if backend == "llvm"
                else _compiled(module, tmp_path, index * 3 + regex.MODES.index(mode))
            )
            run = _Runner(address, compiled.spans)
            for _ in range(40):
                n = rng.randint(0, 12)
                data = bytes(rng.choice(ALPHABET) for _ in range(n))
                pos, endpos = rng.randint(-1, n + 1), rng.randint(-1, n + 2)
                want = _expected(pattern, flags, mode, data, pos, endpos, compiled.groups)
                status, found, spans = run(data, pos, endpos)
                assert status == 0 and (found, spans) == want, (pattern, mode, data, pos, endpos)


@requires_llvm
def test_a_match_that_needs_more_stack_falls_back():
    keep: list = []
    module, compiled = _lowered(rb"(a|b)*c", 0, "search")
    run = _Runner(_jit(module, keep), compiled.spans)
    assert run(b"ab" * 100 + b"c", 0, 201) == (0, True, [0, 201, 199, 200])
    status, _found, _spans = run(b"ab" * 3000, 0, 6000)
    assert status == 1, "the fallback status: Python's `re` answers instead"


# -- the surface ---------------------------------------------------------------------

PROGRAM = """
    import array
    import re

    import ppy
    from ppy import Buffer

    WORD = re.compile(rb"[A-Za-z]+")
    PAIR = re.compile(rb"(?P<key>\\w+)\\s*=\\s*(\\d+)", re.I | re.M)
    BACK = re.compile(rb"(a)\\1")


    def count_words(text: Buffer[ppy.u8]) -> int:
        n = 0
        pos = 0
        while True:
            m = WORD.search(text, pos)
            if m is None:
                return n
            n += 1
            pos = m.end()


    def sum_values(text: Buffer[ppy.u8]) -> int:
        total = 0
        pos = 0
        while pos < len(text):
            m = PAIR.search(text, pos)
            if not m:
                break
            value = 0
            for i in range(m.start(2), m.end(2)):
                value = value * 10 + (text[i] - 48)
            total += value
            first, last = m.span("key")
            total += last - first
            pos = m.end()
        return total


    def is_identifier(text: Buffer[ppy.u8]) -> bool:
        return re.fullmatch(rb"[A-Za-z_]\\w*", text) is not None


    def doubled(text: Buffer[ppy.u8]) -> bool:
        return BACK.search(text) is not None


    def main() -> None:
        text = array.array("B", b"alpha = 12, beta=7 gamma  =   100; delta")
        print(count_words(text), sum_values(text))
        print(is_identifier(array.array("B", b"snake_case9")), is_identifier(array.array("B", b"9")))
        print(doubled(array.array("B", b"xaay")), doubled(array.array("B", b"xay")))


    main()
    """


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_checker_types_the_re_surface(write, analyze):
    path = write("words.ppy", PROGRAM)
    bundle = analyze(path, backend="llvm")
    assert [d.code for d in bundle.diagnostics.sorted()] == []
    symbols = bundle.symbols.modules["words"]
    assert symbols.pattern_globals == {
        "WORD": (rb"[A-Za-z]+", 0),
        "PAIR": (rb"(?P<key>\w+)\s*=\s*(\d+)", re.IGNORECASE | re.MULTILINE),
        "BACK": (rb"(a)\1", 0),
    }
    functions = bundle.analysis.modules["words"].functions
    assert "ReadGlobal" not in str(functions["words.count_words"].effects)


def test_a_group_is_typed_but_stays_on_python(write, analyze):
    path = write(
        "groups.ppy",
        """
        import re

        import ppy
        from ppy import Buffer

        WORD = re.compile(rb"[A-Za-z]+")


        def first(text: Buffer[ppy.u8]) -> int:
            m = WORD.search(text)
            if m is None:
                return 0
            return len(m.group(0))


        def flags() -> int:
            return re.I | re.M
        """,
    )
    bundle = analyze(path, backend="llvm")
    assert [d.code for d in bundle.diagnostics.sorted()] == []


@requires_llvm
def test_the_program_runs_natively_and_answers_like_python(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "words.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    plain = subprocess.run(
        [sys.executable, "words.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout == "4 133\nTrue False\nTrue False\n"
    native = _ppy(tmp_path, "run", "words.ppy")
    assert native.returncode == 0, native.stderr
    assert native.stdout.endswith(plain.stdout)
    emitted = _ppy(tmp_path, "emit", "ir", "words.ppy")
    assert emitted.returncode == 0, emitted.stderr
    text = emitted.stdout
    assert "dialect regex 1" in text
    for name in ("count_words", "sum_values", "is_identifier"):
        assert f"func @words_{name}(%text: buffer<u8>" in text, f"{name} lowered natively"
    assert "func @words_doubled(buffer<u8>" in text, "a backreference stays on Python"
    assert "private func @ppy.regex." in text
    explained = _ppy(tmp_path, "explain", "words.ppy:44")
    assert "backreference" in explained.stdout
