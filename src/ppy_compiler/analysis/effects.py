"""Effect vocabulary and summaries (spec 11)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["Effect", "EffectSet", "EffectSummary", "PURE_FORBIDDEN", "EMPTY"]


class Effect(enum.StrEnum):
    ALLOC = "Alloc"
    READ_OBJECT = "ReadObject"
    WRITE_OBJECT = "WriteObject"
    READ_GLOBAL = "ReadGlobal"
    WRITE_GLOBAL = "WriteGlobal"
    IO = "IO"
    RANDOM = "Random"
    TIME = "Time"
    THREAD = "Thread"
    PROCESS = "Process"
    SYNC = "Sync"
    MAY_RAISE = "MayRaise"
    PYTHON_CALLBACK = "PythonCallback"
    EXTERNAL_UNKNOWN = "ExternalUnknown"


#: Effects that a `@ppy.pure` function may not have (spec 11.2).
#: `Alloc`, `ReadObject`, and `MayRaise` are compatible with purity.
PURE_FORBIDDEN = frozenset({
    Effect.WRITE_OBJECT,
    Effect.READ_GLOBAL,
    Effect.WRITE_GLOBAL,
    Effect.IO,
    Effect.RANDOM,
    Effect.TIME,
    Effect.THREAD,
    Effect.PROCESS,
    Effect.SYNC,
    Effect.PYTHON_CALLBACK,
    Effect.EXTERNAL_UNKNOWN,
})


@dataclass(frozen=True, slots=True)
class EffectSet:
    effects: frozenset[Effect] = frozenset()
    raises: frozenset[str] = frozenset()

    @staticmethod
    def of(*effects: Effect, raises: Iterable[str] = ()) -> "EffectSet":
        found = set(effects)
        raised = frozenset(raises)
        if raised:
            found.add(Effect.MAY_RAISE)
        return EffectSet(frozenset(found), raised)

    def __or__(self, other: "EffectSet") -> "EffectSet":
        return EffectSet(self.effects | other.effects, self.raises | other.raises)

    def __contains__(self, effect: Effect) -> bool:
        return effect in self.effects

    def add(self, *effects: Effect, raises: Iterable[str] = ()) -> "EffectSet":
        return self | EffectSet.of(*effects, raises=raises)

    @property
    def is_pure(self) -> bool:
        return not (self.effects & PURE_FORBIDDEN)

    @property
    def may_raise(self) -> bool:
        return Effect.MAY_RAISE in self.effects

    @property
    def is_empty(self) -> bool:
        return not self.effects

    def violations(self) -> frozenset[Effect]:
        return frozenset(self.effects & PURE_FORBIDDEN)

    def __str__(self) -> str:
        if not self.effects:
            return "none"
        names = []
        for effect in sorted(self.effects, key=str):
            if effect is Effect.MAY_RAISE and self.raises:
                names.append(f"MayRaise[{', '.join(sorted(self.raises))}]")
            else:
                names.append(str(effect))
        return ", ".join(names)


EMPTY = EffectSet()


@dataclass(slots=True)
class EffectSummary:
    """A cached, externally visible effect summary for a function (spec 11.5)."""

    qualname: str
    effects: EffectSet = field(default_factory=lambda: EMPTY)
    declared_pure: bool = False
    verified_pure: bool = False
    unknown_callees: tuple[str, ...] = ()

    @property
    def is_known(self) -> bool:
        return Effect.EXTERNAL_UNKNOWN not in self.effects

    def abi_hash_input(self) -> str:
        parts = sorted(str(e) for e in self.effects.effects)
        raises = sorted(self.effects.raises)
        return f"{self.qualname}|{','.join(parts)}|{','.join(raises)}|{int(self.verified_pure)}"
