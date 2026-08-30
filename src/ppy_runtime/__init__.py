"""The PPY runtime: everything a built artifact needs, and nothing more.

This package must never import `ppy_compiler`. A compiled application
depends on the interpreter, this runtime, its native library, and its
manifest -- uninstalling the compiler must not break it (spec: runtime
separation).
"""

__version__ = "0.1.0"
