"""The builtin dialects, registered into every registry on first use."""

from __future__ import annotations

from ..dialect import Dialect

__all__ = ["builtin_dialects"]


def builtin_dialects() -> list[Dialect]:
    from .atomic import AtomicDialect
    from .concurrency import ConcurrencyDialect
    from .core import CoreDialect
    from .cpu import CpuDialect
    from .math import MathDialect
    from .parallel import ParallelDialect
    from .simd import SimdDialect

    return [
        CoreDialect(),
        MathDialect(),
        SimdDialect(),
        CpuDialect(),
        AtomicDialect(),
        ConcurrencyDialect(),
        ParallelDialect(),
    ]
