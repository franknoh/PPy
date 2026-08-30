"""Compiler-side alias: generated-module execution lives in `ppy_runtime`."""

from ppy_runtime.execute import (
    ExecutionResult,
    NativeBinder,
    execute,
    format_traceback,
    install_loader,
)

__all__ = ["ExecutionResult", "NativeBinder", "execute", "format_traceback", "install_loader"]
