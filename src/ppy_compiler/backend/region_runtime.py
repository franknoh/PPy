"""Compiler-side alias: ATen-region binding lives in `ppy_runtime`."""

from ppy_runtime.regions import RegionBinding, bind_region

__all__ = ["RegionBinding", "bind_region"]
