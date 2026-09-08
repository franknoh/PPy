"""From analyzed Python to the canonical IR.

The frontend reads the AST once, with what analysis established about it,
and writes IR; every backend reads the IR and never the AST. `ast_to_ir`
covers the natively lowerable subset -- scalars, fixed tuples, value
classes, borrowed buffers, loops, calls -- and refuses anything else with
the reason, the same reasons the LLVM lowering gave.
"""

from __future__ import annotations

from .ast_to_ir import Frontend, Lowered, lower_function, lower_module_to_ir

__all__ = ["Frontend", "Lowered", "lower_function", "lower_module_to_ir"]
