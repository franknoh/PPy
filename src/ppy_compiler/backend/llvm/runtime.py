"""Compiler-side alias: the binding trampolines live in `ppy_runtime`."""

from ppy_runtime.binding import GuardFailed, NativeBinding, bind

__all__ = ["GuardFailed", "NativeBinding", "bind"]
