"""The builtin dialects, registered into every registry on first use."""

from __future__ import annotations

from ..dialect import Dialect

__all__ = ["builtin_dialects"]


def builtin_dialects() -> list[Dialect]:
    from .core import CoreDialect
    from .math import MathDialect

    return [CoreDialect(), MathDialect()]
