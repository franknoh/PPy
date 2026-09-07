"""Pretty Python runtime package.

Importing `ppy` is cheap: it installs the `.ppy` import hook and exposes
directives and annotation markers. It never initializes LLVM, loads compiler
services, or starts background processes (spec 6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ppy_runtime.version import VERSION as __version__

from . import atomic, concurrent, cpu, ffi, simd
from ._directives import (
    DIRECTIVE_ATTR,
    Directive,
    attach,
    directives_of,
    dynamic,
    fastmath,
    inline,
    jax,
    jit,
    noinline,
    opt,
    parallel,
    pure,
    reflective,
    specialize,
)
from ._importer import (
    PPyAmbiguousModuleWarning,
    add_import_root,
    import_roots,
    install,
    is_installed,
    uninstall,
)
from ._markers import (
    NUMERIC_MARKERS,
    Array,
    ArraySpec,
    Borrowed,
    Buffer,
    BufferSpec,
    Contiguous,
    DType,
    Dynamic,
    FloatWidth,
    IntWidth,
    Length,
    Mut,
    NoAlias,
    Owned,
    Range,
    Shape,
    Vector,
    VectorSpec,
    check,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)
from ._native import native_import, native_imports
from ._native_api import native
from .autodiff import grad, value_and_grad

__all__ = [
    "DIRECTIVE_ATTR",
    "NUMERIC_MARKERS",
    "Array",
    "ArraySpec",
    "Borrowed",
    "Buffer",
    "BufferSpec",
    "Contiguous",
    "DType",
    "Directive",
    "Dynamic",
    "FloatWidth",
    "IntWidth",
    "Length",
    "Mut",
    "NoAlias",
    "Owned",
    "PPyAmbiguousModuleWarning",
    "Range",
    "Shape",
    "Vector",
    "VectorSpec",
    "__version__",
    "add_import_root",
    "atomic",
    "attach",
    "check",
    "concurrent",
    "cpu",
    "directives_of",
    "dynamic",
    "f16",
    "f32",
    "f64",
    "fastmath",
    "ffi",
    "grad",
    "i8",
    "i16",
    "i32",
    "i64",
    "import_roots",
    "inline",
    "input",
    "install",
    "is_installed",
    "jax",
    "jit",
    "native",
    "native_import",
    "native_imports",
    "noinline",
    "opt",
    "parallel",
    "pure",
    "read_ints",
    "read_token",
    "reader_available",
    "reflective",
    "simd",
    "specialize",
    "u8",
    "u16",
    "u32",
    "u64",
    "uninstall",
    "value_and_grad",
]

if TYPE_CHECKING:  # the names below are real; PEP 562 just defers the import
    from ._io import (  # pylint: disable=redefined-builtin
        input,
        read_ints,
        read_token,
        reader_available,
    )

#: Reading input pulls in `ctypes` and the compiled reader, which a program
#: that never reads should not pay for; PEP 562 defers it to first use.
_READERS = frozenset({"input", "read_ints", "read_token", "reader_available"})


def __getattr__(name: str) -> object:
    if name == "buffer":
        from ._alloc import buffer as allocate

        return allocate
    if name in _READERS:
        from . import _io

        return getattr(_io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


install()
