"""The builtin dialects, registered into every registry on first use."""

from __future__ import annotations

from ..dialect import Dialect

__all__ = ["builtin_dialects"]


def builtin_dialects() -> list[Dialect]:
    from .aio import AsyncDialect
    from .arrow import ArrowDialect
    from .atomic import AtomicDialect
    from .columnar import ColumnarDialect
    from .concurrency import ConcurrencyDialect
    from .core import CoreDialect
    from .cpu import CpuDialect
    from .fft import FftDialect
    from .gpu import GpuDialect
    from .layout import LayoutDialect
    from .linalg import LinalgDialect
    from .math import MathDialect
    from .parallel import ParallelDialect
    from .prof import ProfDialect
    from .simd import SimdDialect
    from .sparse import SparseDialect
    from .special import SpecialDialect
    from .tensor import TensorDialect

    return [
        CoreDialect(),
        MathDialect(),
        SimdDialect(),
        CpuDialect(),
        AtomicDialect(),
        ConcurrencyDialect(),
        ParallelDialect(),
        LayoutDialect(),
        TensorDialect(),
        LinalgDialect(),
        FftDialect(),
        SpecialDialect(),
        SparseDialect(),
        ColumnarDialect(),
        ArrowDialect(),
        GpuDialect(),
        AsyncDialect(),
        ProfDialect(),
    ]
