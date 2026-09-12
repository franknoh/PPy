"""The backends, and the interface an external one implements.

`Backend` and the types around it are the public surface a backend package
writes against, with `ppy_compiler.ir` for the IR it receives; the registry
finds and loads backends, builtin and installed. The packages beside this
one -- `llvm`, `c`, `nvvm`, `stablehlo`, `python` -- are the compiler's own.
"""

from .base import (
    BACKEND_API_VERSION,
    Backend,
    BackendConfig,
    BackendContext,
    BackendError,
    BackendUnavailable,
    BackendValidationError,
    BuildResult,
    EmitFormat,
    ToolchainStatus,
)
from .registry import (
    ENTRY_POINT_GROUP,
    BackendCatalog,
    BackendInfo,
    BackendLoadError,
    available_backends,
    discover_external_backends,
    emit_format_owner,
    load_backend,
)

__all__ = [
    "BACKEND_API_VERSION",
    "ENTRY_POINT_GROUP",
    "Backend",
    "BackendCatalog",
    "BackendConfig",
    "BackendContext",
    "BackendError",
    "BackendInfo",
    "BackendLoadError",
    "BackendUnavailable",
    "BackendValidationError",
    "BuildResult",
    "EmitFormat",
    "ToolchainStatus",
    "available_backends",
    "discover_external_backends",
    "emit_format_owner",
    "load_backend",
]
