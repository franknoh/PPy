"""Typed input: `ppy.input` reads lines, `ppy.scan` reads tokens, `ppy.read_*` fill buffers;
the compiled scanner and the Python fallback agree on every byte (spec 28)."""

from __future__ import annotations

import array
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ppy

_FALLBACK = "from ppy import _io\n_io._SOURCE.append(_io._Fallback())\n"


def _piped(text: str, program: str, *, fallback: bool = False) -> str:
    """Run a snippet in a fresh interpreter with `text` on standard input."""
    source = ("import ppy\n" + _FALLBACK if fallback else "") + textwrap.dedent(program)
    done = subprocess.run(
        [sys.executable, "-c", source],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )
    return done.stdout.strip()


def _both(text: str, program: str) -> str:
    """The snippet's output, which the compiled reader and the fallback must share."""
    compiled = _piped(text, program)
    assert _piped(text, program, fallback=True) == compiled, "the fallback must agree"
    return compiled


#: A snippet that prints what an expression gives, or the exception it raises.
_SHOW = """
    import ppy
    from ppy import Buffer
    try:
        print(repr({expression}))
    except Exception as error:
        print(type(error).__name__, error)
    """


def _show(expression: str) -> str:
    return _SHOW.format(expression=expression)


# -- the low-level scanner -------------------------------------------------------------


def test_read_ints_fills_a_buffer():
    output = _both(
        "4 10 -20 30 40\n",
        """
        import array, ppy
        head = array.array("q", [0])
        ppy.read_ints(head)
        rest = array.array("q", [0] * head[0])
        print(ppy.read_ints(rest), list(rest))
        """,
    )
    assert output == "4 [10, -20, 30, 40]"


def test_reading_continues_where_the_last_call_stopped():
    output = _both(
        "1 2 3 4 5 6\n",
        """
        import array, ppy
        first = array.array("q", [0, 0])
        second = array.array("q", [0, 0, 0, 0])
        ppy.read_ints(first)
        ppy.read_ints(second)
        print(list(first), list(second))
        """,
    )
    assert output == "[1, 2] [3, 4, 5, 6]"


def test_a_short_input_reports_how_much_it_read():
    output = _both(
        "7 8\n",
        """
        import array, ppy
        buffer = array.array("q", [0] * 5)
        print(ppy.read_ints(buffer), list(buffer))
        """,
    )
    assert output == "2 [7, 8, 0, 0, 0]"


def test_read_token_fills_bytes_or_wide_slots():
    output = _both(
        "hello world\n",
        """
        import array, ppy
        narrow = array.array("b", bytes(16))
        wide = array.array("q", [0] * 16)
        first = ppy.read_token(narrow)
        second = ppy.read_token(wide)
        print(bytes(narrow[:first]).decode(), list(wide[:second]))
        """,
    )
    assert output == "hello [119, 111, 114, 108, 100]"


def test_read_token_cuts_at_the_buffers_capacity_and_drops_the_rest():
    """The caller chose the capacity; the rest of an over-long token is consumed."""
    output = _both(
        "abcdefgh next\n",
        """
        import array, ppy
        room = array.array("b", bytes(3))
        got = ppy.read_token(room)
        print(got, bytes(room[:got]), ppy.scan[str]())
        """,
    )
    assert output == "3 b'abc' next"


def test_read_ints_refuses_a_token_that_is_not_an_integer():
    """`abc` where an integer is expected is an error, never skipped -- on both readers."""
    output = _both(
        "1 2 abc 4\n",
        """
        import array, ppy
        buffer = array.array("q", [0] * 4)
        try:
            ppy.read_ints(buffer)
        except ValueError as error:
            print(error, list(buffer))
        """,
    )
    assert output == "expected an integer, got 'abc' [1, 2, 0, 0]"


def test_a_read_only_buffer_is_refused():
    with pytest.raises(TypeError):
        ppy.read_ints(memoryview(b"12345678").toreadonly())


def test_a_buffer_of_the_wrong_width_is_refused():
    with pytest.raises(TypeError):
        ppy.read_ints(array.array("i", [0, 0]))


# -- ppy.input: lines ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expression", "expected"),
    [
        ("abc def\n", "ppy.input[str]()", "'abc def'"),
        ("  padded\t\n", "ppy.input[str]()", "'  padded\\t'"),
        ("123\n", "ppy.input[int]()", "123"),
        ("  42  \n", "ppy.input[int]()", "42"),
        ("1_000\n", "ppy.input[int]()", "1000"),
        ("-7\n", "ppy.input[int]()", "-7"),
        ("1 2\n", "ppy.input[int]()", "ValueError invalid literal for int() with base 10: '1 2'"),
        ("abc\n", "ppy.input[int]()", "ValueError invalid literal for int() with base 10: 'abc'"),
        ("2.5\n", "ppy.input[float]()", "2.5"),
        ("1 2\n", "ppy.input[float]()", "ValueError could not convert string to float: '1 2'"),
        ("\n", "ppy.input[str]()", "''"),
        ("", "ppy.input[str]()", "EOFError EOF when reading a line"),
        ("", "ppy.input[int]()", "EOFError EOF when reading a line"),
        ("last line", "ppy.input[str]()", "'last line'"),
        ("crlf\r\n", "ppy.input[str]()", "'crlf'"),
        ("1 2\n", "ppy.input[tuple[int, int]]()", "(1, 2)"),
        ("1 2.5 x\n", "ppy.input[tuple[int, float, str]]()", "(1, 2.5, 'x')"),
        (
            "1\n2\n",
            "ppy.input[tuple[int, int]]()",
            "ValueError expected 2 field(s) on the line, got 1",
        ),
        (
            "1 2 3\n",
            "ppy.input[tuple[int, int]]()",
            "ValueError expected 2 field(s) on the line, got 3",
        ),
        (
            "1 2 3 4 5 6 7 8 9\n",
            "ppy.input[tuple[int, int]]()",
            "ValueError expected 2 field(s) on the line, got 9",
        ),
        (
            "1 2 3 x\n",
            "ppy.input[tuple[int, int]]()",
            "ValueError invalid literal for int() with base 10: 'x'",
        ),
        ("+1 -2_0\n", "ppy.input[tuple[int, int]]()", "(1, -20)"),
        (
            "1 99999999999999999999\n",
            "ppy.input[tuple[int, int]]()",
            "OverflowError int too big to convert",
        ),
        (
            "1 2.5\n",
            "ppy.input[tuple[int, int]]()",
            "ValueError invalid literal for int() with base 10: '2.5'",
        ),
        ("1 2.5\n", "ppy.input[tuple[int, float]]()", "(1, 2.5)"),
        ("", "ppy.input[tuple[int, int]]()", "EOFError EOF when reading a line"),
        ("\n", "ppy.input[tuple[int, int]]()", "ValueError expected 2 field(s) on the line, got 0"),
        ("1 2 3\n", "ppy.input[list[int]]()", "[1, 2, 3]"),
        ("1.5 2\n", "ppy.input[list[float]]()", "[1.5, 2.0]"),
        ("a  b\n", "ppy.input[list[str]]()", "['a', 'b']"),
        ("\n", "ppy.input[list[int]]()", "[]"),
        (
            "1 x\n",
            "ppy.input[list[int]]()",
            "ValueError invalid literal for int() with base 10: 'x'",
        ),
        ("1 2 3\n", "list(ppy.input[Buffer[int]]())", "[1, 2, 3]"),
        ("  +1\t-2 1_000 \r\n", "list(ppy.input[Buffer[int]]())", "[1, -2, 1000]"),
        ("\n", "list(ppy.input[Buffer[int]]())", "[]"),
        ("1 2 3", "list(ppy.input[Buffer[int]]())", "[1, 2, 3]"),
        ("", "ppy.input[Buffer[int]]()", "EOFError EOF when reading a line"),
        (
            "9223372036854775807 -9223372036854775808\n",
            "list(ppy.input[Buffer[int]]())",
            "[9223372036854775807, -9223372036854775808]",
        ),
        (
            "1 9223372036854775808\n",
            "ppy.input[Buffer[int]]()",
            "OverflowError int too big to convert",
        ),
        (
            "1 x 3\n",
            "ppy.input[Buffer[int]]()",
            "ValueError invalid literal for int() with base 10: 'x'",
        ),
        (
            "1__0\n",
            "ppy.input[Buffer[int]]()",
            "ValueError invalid literal for int() with base 10: '1__0'",
        ),
        (
            "_1\n",
            "ppy.input[Buffer[int]]()",
            "ValueError invalid literal for int() with base 10: '_1'",
        ),
        (
            "5_\n",
            "ppy.input[Buffer[int]]()",
            "ValueError invalid literal for int() with base 10: '5_'",
        ),
        (
            "-\n",
            "ppy.input[Buffer[int]]()",
            "ValueError invalid literal for int() with base 10: '-'",
        ),
        (
            "1 2\n",
            "ppy.input[Buffer[int]](2)",
            (
                "TypeError `ppy.input[Buffer[int]]()` reads the whole line and takes no count; "
                "`ppy.scan[Buffer[int]](n)` reads n tokens"
            ),
        ),
        (
            "1 2\n",
            "ppy.input[Buffer[float]]()",
            "TypeError a line of integers reads into `Buffer[int]`, not `Buffer[float]`",
        ),
        (
            "1\n",
            "ppy.input[int]('n? ')",
            "TypeError `ppy.input[T]()` takes no argument; print a prompt first, then read",
        ),
        (
            "1\n",
            "ppy.input[dict]()",
            "TypeError <class 'dict'> is not something `ppy.input` knows how to read",
        ),
    ],
)
def test_input_reads_one_line_the_way_the_builtin_does(text, expression, expected):
    assert _both(text, _show(expression)) == expected


def test_input_reads_line_after_line_and_never_borrows_from_the_next():
    output = _both(
        "1 2\n3\nhello world\n",
        """
        import ppy
        print(ppy.input[tuple[int, int]](), ppy.input[int](), repr(ppy.input[str]()))
        """,
    )
    assert output == "(1, 2) 3 'hello world'"


def test_a_refused_tuple_line_was_consumed_whole_and_a_reader_is_reused():
    """The next read starts on the next line, and `ppy.input[T]` is one object per `T`."""
    output = _both(
        "1 2 3 4 5 6 7 8 9 10\n1 x 3\n7 8\n",
        """
        import ppy
        read = ppy.input[tuple[int, int]]
        assert read is ppy.input[tuple[int, int]]
        for _ in range(2):
            try:
                read()
            except ValueError as error:
                print(error)
        print(read())
        """,
    )
    assert output == (
        "expected 2 field(s) on the line, got 10\n"
        "invalid literal for int() with base 10: 'x'\n"
        "(7, 8)"
    )


def test_a_line_of_integers_is_read_whole_and_the_next_line_stays_next():
    """The buffer grows with the line; the read consumes the line and nothing after it."""
    line = " ".join(str(i) for i in range(5000))
    output = _both(
        line + "\n7\n",
        """
        import ppy
        from ppy import Buffer
        values = ppy.input[Buffer[int]]()
        print(len(values), sum(values), values.typecode, ppy.input[int]())
        """,
    )
    assert output == "5000 12497500 q 7"


def test_a_refused_field_still_consumed_its_whole_line():
    """As `int(input().split()...)` raised after `input()` had read the line."""
    output = _both(
        "1 x 3\n4 5\n",
        """
        import ppy
        from ppy import Buffer
        try:
            ppy.input[Buffer[int]]()
        except ValueError as error:
            print(error, "|", list(ppy.input[Buffer[int]]()))
        """,
    )
    assert output == "invalid literal for int() with base 10: 'x' | [4, 5]"


def test_the_buffer_line_read_matches_the_array_idiom():
    """`array.array("q", map(int, input().split()))`, which the converter writes it for."""
    for text in (
        "1 2 3\n",
        " 4\t5 \n",
        "\n",
        "1_0 -0 +7\n",
        "1 x\n",
        "1 99999999999999999999\n",
        "",
    ):
        idiom = _piped(
            text,
            """
            import array
            try:
                print(list(array.array("q", map(int, input().split()))))
            except Exception as error:
                print(type(error).__name__, error)
            """,
        )
        assert _both(text, _show("list(ppy.input[Buffer[int]]())")) == idiom, text


def test_a_long_line_is_read_whole():
    """A high-level read never cuts: the old scalar read stopped at 4096 bytes."""
    long = "x" * 10000 + "é" * 100
    output = _both(
        long + "\n" + "next\n",
        "import ppy\nline = ppy.input[str]()\nprint(len(line), line[-1], ppy.input[str]())",
    )
    assert output == "10100 é next"


def test_input_matches_the_builtin_on_the_edges():
    """What `input()` returns, `ppy.input[str]()` returns; where it raises, so does ours."""
    cases = ["a\n\nb", "a\nb\n", "", "\n", "no newline", "  spaced  \n"]
    for text in cases:
        expected = _piped(
            text,
            """
            out = []
            try:
                while True:
                    out.append(input())
            except EOFError:
                pass
            print(out)
            """,
        )
        got = _both(
            text,
            """
            import ppy
            out = []
            try:
                while True:
                    out.append(ppy.input[str]())
            except EOFError:
                pass
            print(out)
            """,
        )
        assert got == expected, text


# -- ppy.scan: tokens ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expression", "expected"),
    [
        ("abc def\n", "ppy.scan[str]()", "'abc'"),
        ("\n\n  abc\n", "ppy.scan[str]()", "'abc'"),
        ("1\n2\n", "ppy.scan[tuple[int, int]]()", "(1, 2)"),
        ("12 x\n", "ppy.scan[int]()", "12"),
        ("+7\n", "ppy.scan[int]()", "7"),
        ("-0\n", "ppy.scan[int]()", "0"),
        ("abc 3\n", "ppy.scan[int]()", "ValueError expected an integer, got 'abc'"),
        ("12x 3\n", "ppy.scan[int]()", "ValueError expected an integer, got '12x'"),
        ("1_000\n", "ppy.scan[int]()", "ValueError expected an integer, got '1_000'"),
        ("-\n", "ppy.scan[int]()", "ValueError expected an integer, got '-'"),
        ("9223372036854775807\n", "ppy.scan[int]()", "9223372036854775807"),
        ("-9223372036854775808\n", "ppy.scan[int]()", "-9223372036854775808"),
        (
            "9223372036854775808\n",
            "ppy.scan[int]()",
            "ValueError the integer 9223372036854775808 does not fit in 64 bits",
        ),
        (
            "99999999999999999999\n",
            "ppy.scan[int]()",
            "ValueError the integer 99999999999999999999 does not fit in 64 bits",
        ),
        ("2.5 x\n", "ppy.scan[float]()", "2.5"),
        ("1 0\n", "ppy.scan[tuple[bool, bool]]()", "(True, False)"),
        ("", "ppy.scan[int]()", "EOFError the input ended where an integer was expected"),
        ("   \n", "ppy.scan[str]()", "EOFError the input ended where a token was expected"),
        (
            "1\n",
            "ppy.scan[int]('n? ')",
            "TypeError `ppy.scan[T]()` takes no argument; print a prompt first, then read",
        ),
        (
            "1\n",
            "ppy.scan[Buffer[int]]()",
            "TypeError reading a buffer needs how many values to read",
        ),
        (
            "1\n",
            "ppy.scan[dict]()",
            "TypeError <class 'dict'> is not something `ppy.scan` knows how to read",
        ),
    ],
)
def test_scan_reads_tokens_wherever_they_fall(text, expression, expected):
    assert _both(text, _show(expression)) == expected


def test_scan_fills_a_buffer_by_count():
    output = _both(
        "3\n10 20\n30 40\n",
        """
        import ppy
        from ppy import Buffer
        count = ppy.scan[int]()
        values = ppy.scan[Buffer[int]](count)
        print(count, list(values), ppy.scan[int]())
        """,
    )
    assert output == "3 [10, 20, 30] 40"


def test_a_token_read_leaves_the_rest_of_its_line():
    """A token stops before the whitespace after it, as `scanf` does."""
    output = _both(
        "5\nabc\n",
        """
        import ppy
        print(ppy.scan[int](), repr(ppy.input[str]()), repr(ppy.input[str]()))
        """,
    )
    assert output == "5 '' 'abc'"


def test_a_long_token_is_read_whole():
    long = "y" * 20000
    output = _both(long + " tail\n", "import ppy\nprint(len(ppy.scan[str]()), ppy.scan[str]())")
    assert output == "20000 tail"


def test_the_scanner_needs_the_type_it_is_reading():
    with pytest.raises(TypeError, match=re.escape("ppy.scan")):
        ppy.scan(int)
    with pytest.raises(TypeError, match=re.escape("ppy.input")):
        ppy.input(int)


def test_the_two_readers_agree_on_random_input():
    """Every kind of read, over input with every kind of edge in it, on both readers."""
    import random

    rng = random.Random(3)
    pieces = [
        "12",
        "-5",
        "+3",
        "x",
        "1_0",
        "999999999999999999999",
        " ",
        "\t",
        "\n",
        "\r\n",
        "",
        "abc",
    ]
    program = """
        import ppy
        from ppy import Buffer
        import array
        out = []
        reads = [
            lambda: ppy.scan[int](),
            lambda: ppy.scan[str](),
            lambda: ppy.input[str](),
            lambda: ppy.input[int](),
            lambda: list(ppy.scan[Buffer[int]](2)),
            lambda: (b := array.array("b", bytes(2)), ppy.read_token(b), bytes(b))[1:],
        ]
        for step in range(40):
            read = reads[step % len(reads)]
            try:
                out.append(repr(read()))
            except (ValueError, EOFError, UnicodeDecodeError) as error:
                out.append(type(error).__name__ + ":" + str(error))
        print("|".join(out))
        """
    for _ in range(12):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
        _both(text, program)


# -- prompts and buffers -----------------------------------------------------------------


def test_a_prompt_is_a_print_before_the_read():
    output = _both(
        "7\n",
        """
        import ppy
        print("n? ", end="")
        value = ppy.input[int]()
        print("|", value)
        """,
    )
    assert output == "n? | 7"


def test_a_buffer_is_zeroed_and_typed_by_its_element():
    """`ppy.buffer[T](n)` is what a program allocates for itself."""
    for element, code, width in [
        (int, "q", 8),
        (float, "d", 8),
        (ppy.i8, "b", 1),
        (ppy.u8, "B", 1),
    ]:
        room = ppy.buffer[element](3)
        assert len(room) == 3
        assert list(room) == [0, 0, 0] if code != "d" else list(room) == [0.0, 0.0, 0.0]
        assert room.itemsize == width
        assert room.typecode == code


def test_a_buffer_refuses_an_element_it_cannot_hold():
    with pytest.raises(TypeError, match="not an element type"):
        ppy.buffer[str](4)


def test_a_buffer_refuses_a_size_that_is_not_a_count():
    with pytest.raises(ValueError, match="fewer than no elements"):
        ppy.buffer[int](-1)
    with pytest.raises(TypeError):
        ppy.buffer[int]("4")


def test_a_buffer_needs_the_element_it_is_holding():
    with pytest.raises(TypeError, match=re.escape("ppy.buffer")):
        ppy.buffer(int)


def test_the_readers_are_declared_on_the_package():
    """PEP 562 defers the import; the public surface still says the names are there."""
    for name in (
        "input",
        "scan",
        "read_ints",
        "read_token",
        "reader_available",
        "buffer",
        "check",
        "assume",
    ):
        assert name in ppy.__all__, name
        assert getattr(ppy, name) is not None
    assert ppy.scan is ppy._io.scan and ppy.input is ppy._io.input
