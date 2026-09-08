"""The `ppy` modules that implement a namespace are ordinary code when the checker reads them."""

from __future__ import annotations


def test_a_namespace_modules_own_helpers_are_not_namespace_calls(write, codes):
    write("ppy/__init__.py", "")
    aio = write(
        "ppy/aio.py",
        """
        def _negative(code: int) -> int:
            return -code


        def read(sock: int, n: int) -> int:
            if n < 0:
                return _negative(22)
            return sock + n
        """,
    )
    assert "E1645" not in codes(aio, backend="llvm")
    cpu = write(
        "ppy/cpu.py",
        """
        def _width() -> int:
            return 8


        def lanes() -> int:
            return _width()
        """,
    )
    assert "E1643" not in codes(cpu)
    native = write(
        "ppy/native.py",
        """
        def _check(n: int) -> int:
            return n


        def load(n: int) -> int:
            return _check(n)
        """,
    )
    assert "E1630" not in codes(native)
