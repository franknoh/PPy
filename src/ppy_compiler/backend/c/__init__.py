"""The C and C++ backends: canonical IR as a translation unit, or a header."""

from .emit import EmitError, HeaderOnlyError, Language, emit_module
from .runtime import SHIMS, support_source

__all__ = ["SHIMS", "EmitError", "HeaderOnlyError", "Language", "emit_module", "support_source"]
