"""What one module's conversion decided to write (spec 9.4).

A plan is the whole interface between deciding and writing: the planning
step fills it in from the analysis, and the rewriting step reads it and
touches nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ConversionPlan", "mentions"]


@dataclass(slots=True)
class ConversionPlan:
    """Annotations to insert into one module, keyed by source position."""

    params: dict[tuple[int, str], str] = field(default_factory=dict)
    returns: dict[int, str] = field(default_factory=dict)
    assignments: dict[tuple[int, str], str] = field(default_factory=dict)
    #: Instance fields, annotated where `__init__` first assigns them.
    fields: dict[tuple[int, str], str] = field(default_factory=dict)
    #: Class names that are not yet bound where an annotation mentioning them
    #: is written, and so have to be quoted.
    forward: dict[str, int] = field(default_factory=dict)
    #: `ppy` decorators to attach, keyed by the line the `def` starts on.
    decorators: dict[int, tuple[str, ...]] = field(default_factory=dict)
    #: Module-level list constructions to wrap in `array.array`, by line.
    buffers: dict[int, str] = field(default_factory=dict)
    needs_array: bool = False
    #: Modules of this project that the file imports. `import ppy` installs the
    #: loader those need, so it has to be placed ahead of them.
    local_imports: set[str] = field(default_factory=set)
    #: Classes proven safe to move, or None to move any (aggressive mode).
    hoistable: frozenset[str] | None = frozenset()
    #: Top-level definitions a hoist may cross (safe mode only): a crossed
    #: decorator that observes module state would see the moved class early.
    reorder_safe: frozenset[str] | None = None
    typing_imports: set[str] = field(default_factory=set)
    ppy_imports: set[str] = field(default_factory=set)
    needs_ppy: bool = False
    #: Rewrite `int(input())` and friends into `ppy.input[T]()`. Only set for
    #: a module that reads with `input` and never touches `sys.stdin`, since
    #: the typed reader owns the file descriptor.
    rewrite_input: bool = False

    def annotation_counts(self) -> dict[str, int]:
        return {
            "parameters": len(self.params),
            "returns": len(self.returns),
            "module_constants": len(self.assignments),
            "fields": len(self.fields),
            "directives": sum(len(added) for added in self.decorators.values()),
        }

    @property
    def is_empty(self) -> bool:
        return not (
            self.params
            or self.returns
            or self.assignments
            or self.fields
            or self.decorators
            or self.buffers
            or self.needs_ppy
        )


def mentions(text: str, name: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(name)}\b", text) is not None
