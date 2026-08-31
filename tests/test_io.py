"""The typed reader: `ppy.read_ints` and `ppy.read_token` (spec 28)."""

from __future__ import annotations

import array
import os
import subprocess
import sys
import textwrap

import pytest

import ppy
from ppy import _io


def _piped(text: str, program: str) -> str:
    """Run a snippet in a fresh interpreter with `text` on standard input."""
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(_io.Path(__file__).parent.parent / "src")},
    )
    return done.stdout.strip()


def test_read_ints_fills_a_buffer():
    output = _piped(
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
    output = _piped(
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
    output = _piped(
        "7 8\n",
        """
        import array, ppy
        buffer = array.array("q", [0] * 5)
        print(ppy.read_ints(buffer), list(buffer))
        """,
    )
    assert output == "2 [7, 8, 0, 0, 0]"


def test_read_token_fills_bytes_or_wide_slots():
    output = _piped(
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


def test_the_python_fallback_matches_the_compiled_reader():
    """Without a C compiler the answers are the same, only slower."""
    output = _piped(
        "3 5 6 7\n",
        """
        import array, ppy
        from ppy import _io
        _io._LOADED.append(None)   # no compiled reader
        head = array.array("q", [0])
        _io.read_ints(head)
        rest = array.array("q", [0] * head[0])
        print(_io.read_ints(rest), list(rest))
        """,
    )
    assert output == "3 [5, 6, 7]"


def test_a_read_only_buffer_is_refused():
    with pytest.raises(TypeError):
        ppy.read_ints(memoryview(b"12345678").toreadonly())


def test_a_buffer_of_the_wrong_width_is_refused():
    with pytest.raises(TypeError):
        ppy.read_ints(array.array("i", [0, 0]))


def test_input_reads_by_type():
    output = _piped(
        "3\n10 20\nhello\n1 2 3\n",
        """
        import ppy
        from ppy import Buffer
        count = ppy.input(int)
        pair = ppy.input(tuple[int, int])
        word = ppy.input(str)
        values = ppy.input(Buffer[int], count)
        print(count, pair, word, list(values))
        """,
    )
    assert output == "3 (10, 20) hello [1, 2, 3]"


def test_input_at_the_end_raises_eof():
    output = _piped(
        "",
        """
        import ppy
        try:
            ppy.input(int)
        except EOFError as error:
            print("eof:", error)
        """,
    )
    assert output.startswith("eof:")


def test_input_refuses_a_shape_it_cannot_read():
    with pytest.raises(TypeError):
        ppy.input(dict)
