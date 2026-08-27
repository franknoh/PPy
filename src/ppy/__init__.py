"""Pretty Python runtime package.

Importing `ppy` is cheap: it installs the `.ppy` import hook and exposes
directives and annotation markers. It never initializes LLVM, loads compiler
services, or starts background processes (spec 6).
"""

from __future__ import annotations

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
    native,
    noinline,
    opt,
    parallel,
    pure,
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
    Buffer,
    BufferSpec,
    Contiguous,
    DType,
    FloatWidth,
    IntWidth,
    Length,
    NoAlias,
    Range,
    Shape,
    Vector,
    VectorSpec,
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DIRECTIVE_ATTR",
    "Directive",
    "attach",
    "directives_of",
    "pure",
    "opt",
    "jit",
    "parallel",
    "native",
    "inline",
    "noinline",
    "specialize",
    "fastmath",
    "dynamic",
    "jax",
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "f16", "f32", "f64",
    "Array", "Vector", "Buffer",
    "IntWidth", "FloatWidth",
    "ArraySpec", "VectorSpec", "BufferSpec",
    "Range", "Length", "NoAlias", "Shape", "DType", "Contiguous",
    "NUMERIC_MARKERS",
    "install",
    "uninstall",
    "is_installed",
    "add_import_root",
    "import_roots",
    "PPyAmbiguousModuleWarning",
]

install()
