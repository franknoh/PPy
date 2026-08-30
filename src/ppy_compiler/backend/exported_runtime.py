"""Compiler-side alias: exported-region binding lives in `ppy_runtime`."""

from ppy_runtime.exported import ExportedBinding, bind_exported

__all__ = ["ExportedBinding", "bind_exported"]
