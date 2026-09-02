"""What analysis produced, separate from the pass that produced it.

Nine modules read these and none of them run the checker: the backends, the
optimizer, the contracts layer, and the driver all consume its results. They
live here so that dependency reads as what it is.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..diagnostics import DiagnosticBag
from . import types as T
from .aliasing import AliasInfo
from .effects import EffectSet
from .refinements import Facts
from .symbols import FunctionInfo, ModuleSymbols, ProjectSymbols

__all__ = ["FunctionAnalysis", "LoweringNote", "ModuleAnalysis", "ProjectAnalysis"]


@dataclass(frozen=True, slots=True)
class LoweringNote:
    """What a plugin decided for one recognized call (spec 18.2)."""

    qualname: str
    lowering: str
    reason: str
    guards: tuple[str, ...] = ()
    line: int = 0


@dataclass(slots=True)
class FunctionAnalysis:
    info: FunctionInfo
    effects: EffectSet = field(default_factory=EffectSet)
    inferred_ret: T.Type = T.NEVER
    ret_facts: Facts = field(default_factory=Facts)
    locals: dict[str, T.Type] = field(default_factory=dict)
    dynamic: bool = False
    unknown_callees: tuple[str, ...] = ()
    purity_blockers: tuple[str, ...] = ()
    native_blockers: tuple[str, ...] = ()
    parallel_blockers: tuple[str, ...] = ()
    escaping: set[str] = field(default_factory=set)
    mutated_params: set[str] = field(default_factory=set)
    #: Parameters this function hands to a callee that writes through them.
    #: The write lands in the caller's memory just as a direct one would.
    delegated_writes: set[str] = field(default_factory=set)
    #: A write whose target is not a plain local name, so the backend cannot
    #: tell what it reached.
    foreign_writes: bool = False
    #: Every mutation this function performed landed on something it allocated
    #: itself, which spec 11.2 permits inside `@ppy.pure`.
    writes_only_locals: bool = True
    calls: set[str] = field(default_factory=set)
    #: The alias map the facts above were resolved through.
    aliases: AliasInfo | None = None

    @property
    def verified_pure(self) -> bool:
        return self.effects.is_pure and not self.unknown_callees


@dataclass(slots=True)
class ModuleAnalysis:
    symbols: ModuleSymbols
    functions: dict[str, FunctionAnalysis] = field(default_factory=dict)
    node_types: dict[int, T.Type] = field(default_factory=dict)
    node_facts: dict[int, Facts] = field(default_factory=dict)
    module_effects: EffectSet = field(default_factory=EffectSet)
    dynamic_spans: list[tuple[int, int]] = field(default_factory=list)
    lowerings: dict[int, LoweringNote] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.symbols.name

    def type_of(self, node: ast.AST) -> T.Type:
        return self.node_types.get(id(node), T.UNKNOWN)

    def facts_of(self, node: ast.AST) -> Facts:
        return self.node_facts.get(id(node), Facts())


@dataclass(slots=True)
class ProjectAnalysis:
    symbols: ProjectSymbols
    modules: dict[str, ModuleAnalysis] = field(default_factory=dict)
    diagnostics: DiagnosticBag = field(default_factory=DiagnosticBag)

    def function(self, qualname: str) -> FunctionAnalysis | None:
        for module in self.modules.values():
            found = module.functions.get(qualname)
            if found is not None:
                return found
        return None
