"""What CPython prints when native code's guard stands for an exception.

Under `ppy run` a failed guard hands the call back to Python, which raises
for itself. A standalone binary has no Python to hand it to, so it prints the
line a traceback ends with and exits with status 1, as CPython does. Where
the text differs between Python versions (`1 // 0` said "integer division or
modulo by zero" before 3.14 and "division by zero" since), it is asked of
the interpreter running the compiler, so the binary agrees with the Python
that built it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Operation

__all__ = [
    "OVERFLOW",
    "empty_extreme",
    "negative_shift",
    "overflow_text",
    "raised_text",
    "said",
    "zero_division",
]

#: Where native code has only a machine word and Python would have gone on
#: with a larger integer. CPython never says this; nothing it says fits.
OVERFLOW = "OverflowError: the result does not fit in a 64-bit integer"

#: The exception a guard of each kind stands for, where it names none itself.
_KINDS = {
    "bounds": "IndexError",
    "zero_division": "ZeroDivisionError",
    "overflow": "OverflowError",
    "range": "OverflowError",
    "contract": "RuntimeError",
    "assert": "AssertionError",
}


def overflow_text(width: int) -> str:
    """What a standalone binary says when a `width`-bit result overflows."""
    return f"OverflowError: the result does not fit in a {width}-bit integer"


def raised_text(op: Operation) -> str:
    """What a failed `core.guard` says where there is no Python to fall back to."""
    text = op.attributes.get("raises")
    if text:
        return str(text)
    kind = str(op.attributes.get("kind", ""))
    if kind == "overflow":
        return OVERFLOW
    name = _KINDS.get(kind, "RuntimeError")
    message = str(op.attributes.get("message") or "")
    return f"{name}: {message}" if message else name


def said(action: Callable[[], object]) -> str:
    """The last line of the traceback `action` raises: `Type: text`."""
    try:
        action()
    except Exception as error:  # noqa: BLE001 - the point is to see what it says
        text = str(error)
        name = type(error).__name__
        return f"{name}: {text}" if text else name
    raise ValueError("the action did not raise")


def _zero() -> int:
    return 0


def _floor_ints() -> object:
    return 1 // _zero()


def _mod_ints() -> object:
    return 1 % _zero()


def _divide_ints() -> object:
    return 1 / _zero()


def _divide_floats() -> object:
    return 1.0 / float(_zero())


def _floor_floats() -> object:
    return 1.0 // float(_zero())


def _mod_floats() -> object:
    return 1.0 % float(_zero())


_DIVISIONS: dict[tuple[str, bool], Callable[[], object]] = {
    ("//", False): _floor_ints,
    ("%", False): _mod_ints,
    ("/", False): _divide_ints,
    ("/", True): _divide_floats,
    ("//", True): _floor_floats,
    ("%", True): _mod_floats,
}


@cache
def zero_division(operator: str, floats: bool) -> str:
    """What dividing by zero says: `operator` is `/`, `//`, or `%`."""
    return said(_DIVISIONS[(operator, floats)])


def _min_of_empty() -> object:
    return min(list[int]())


def _max_of_empty() -> object:
    return max(list[int]())


@cache
def empty_extreme(operation: str) -> str:
    """What `min([])` or `max([])` says."""
    return said(_min_of_empty if operation == "min" else _max_of_empty)


def _shift_negative() -> object:
    return 1 << -(_zero() + 1)


@cache
def negative_shift() -> str:
    """What shifting by a negative count says."""
    return said(_shift_negative)
