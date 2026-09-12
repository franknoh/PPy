"""The pass manager: passes over modules, in order, with what they need.

A pass names the analyses it requires, preserves, and invalidates, and the
context serves analyses from a cache that a pass's own declaration
empties. Between passes the manager can verify the module, so that a pass
which breaks it is named in the error rather than found by the next one.
Plugins add passes at named stages; the manager runs them where the stage
falls in the pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .analysis import ANALYSES
from .dialect import DialectRegistry
from .dialect import registry as default_registry
from .model import IRFunction, IRModule
from .verify import VerifyError, verify

__all__ = [
    "STAGES",
    "FunctionPass",
    "Pass",
    "PassContext",
    "PassManager",
    "PassReport",
    "PassVerificationError",
]

#: Where a plugin may hang a pass of its own; `backend` is where a backend
#: hangs its own, after every shared pass and the plugins' `before-backend`.
STAGES = (
    "after-ir-generation",
    "after-canonicalization",
    "before-optimization",
    "after-optimization",
    "before-backend",
    "backend",
)


class PassContext:
    """What passes share: the registry, options, remarks, and cached analyses."""

    def __init__(
        self,
        registry: DialectRegistry | None = None,
        *,
        verify_after_each: bool = False,
        options: dict[str, object] | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.verify_after_each = verify_after_each
        self.options: dict[str, object] = dict(options or {})
        self.remarks: list[str] = []
        self._analyses: dict[tuple[str, int], object] = {}
        self._functions: dict[int, IRFunction] = {}

    def analysis(self, name: str, function: IRFunction) -> object:
        """The named analysis of `function`, computed once until invalidated."""
        key = (name, id(function))
        if key not in self._analyses:
            compute = ANALYSES.get(name)
            if compute is None:
                raise KeyError(f"no analysis named {name!r}")
            self._analyses[key] = compute(function)
            self._functions[id(function)] = function
        return self._analyses[key]

    def invalidate(self, function: IRFunction | None = None, keep: Iterable[str] = ()) -> None:
        """Drop cached analyses except `keep`, for one function or all."""
        kept = set(keep)
        for key in list(self._analyses):
            name, owner = key
            if (function is None or owner == id(function)) and name not in kept:
                del self._analyses[key]

    def cached(self) -> set[tuple[str, str]]:
        """(analysis, function name) pairs currently cached, for tests."""
        return {(name, self._functions[owner].name) for name, owner in self._analyses}

    def remark(self, message: str) -> None:
        self.remarks.append(message)


class Pass:
    """One transformation of a module."""

    name: str = ""
    #: Analyses this pass reads; the manager computes them ahead of it.
    requires: tuple[str, ...] = ()
    #: Analyses still valid after this pass changed something. "all" keeps every one.
    preserves: tuple[str, ...] | str = ()
    #: Analyses this pass makes stale even when it reports no change.
    invalidates: tuple[str, ...] = ()

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        """Transform `module`; return whether anything changed."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name or type(self).__name__


class FunctionPass(Pass):
    """A pass that works one function at a time."""

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        changed = False
        for function in module.functions.values():
            if function.is_declaration:
                continue
            if self.run_on_function(function, ctx):
                changed = True
                self._invalidate(function, ctx)
            elif self.invalidates:
                ctx.invalidate(function, keep=[n for n in ANALYSES if n not in self.invalidates])
        return changed

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        raise NotImplementedError

    def _invalidate(self, function: IRFunction, ctx: PassContext) -> None:
        if self.preserves == "all":
            keep: Iterable[str] = ANALYSES
        else:
            keep = [n for n in self.preserves if n not in self.invalidates]
        ctx.invalidate(function, keep=keep)


@dataclass(slots=True)
class PassReport:
    """What ran, whether it changed anything, and how long it took."""

    entries: list[tuple[str, bool, float]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(changed for _name, changed, _seconds in self.entries)

    def names(self) -> list[str]:
        return [name for name, _changed, _seconds in self.entries]


class PassVerificationError(Exception):
    """A pass left the module invalid; the pass is named."""

    def __init__(self, pass_name: str, errors: list[VerifyError]) -> None:
        self.pass_name = pass_name
        self.errors = errors
        detail = "\n".join(str(e) for e in errors)
        super().__init__(f"pass {pass_name!r} broke the IR:\n{detail}")


@dataclass(frozen=True, slots=True)
class _Stage:
    name: str


class PassManager:
    def __init__(self, ctx: PassContext | None = None) -> None:
        self.ctx = ctx or PassContext()
        self._items: list[Pass | _Stage] = []
        self._stage_passes: dict[str, list[Callable[[], Pass]]] = {s: [] for s in STAGES}

    def add(self, pass_: Pass) -> PassManager:
        self._items.append(pass_)
        return self

    def add_stage(self, stage: str) -> PassManager:
        """Mark where passes registered for `stage` run."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")
        self._items.append(_Stage(stage))
        return self

    def register_stage_pass(self, stage: str, factory: Callable[[], Pass]) -> None:
        """A pass a plugin adds; it runs wherever the pipeline marks `stage`."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")
        self._stage_passes[stage].append(factory)

    def passes(self) -> list[Pass]:
        """The passes in the order they will run, stage passes expanded."""
        expanded: list[Pass] = []
        for item in self._items:
            if isinstance(item, _Stage):
                expanded.extend(factory() for factory in self._stage_passes[item.name])
            else:
                expanded.append(item)
        return expanded

    def run(self, module: IRModule) -> PassReport:
        report = PassReport()
        for pass_ in self.passes():
            for function in module.functions.values():
                if function.is_declaration:
                    continue
                for name in pass_.requires:
                    self.ctx.analysis(name, function)
            started = time.perf_counter()
            changed = pass_.run(module, self.ctx)
            report.entries.append((repr(pass_), changed, time.perf_counter() - started))
            if changed and not isinstance(pass_, FunctionPass):
                keep = ANALYSES if pass_.preserves == "all" else pass_.preserves
                self.ctx.invalidate(keep=keep)
            if self.ctx.verify_after_each:
                errors = verify(module, self.ctx.registry)
                if errors:
                    raise PassVerificationError(repr(pass_), errors)
        return report
