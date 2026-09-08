"""Effect vocabulary and summaries (spec 11).

An effect is a fact about what running a function can touch. The vocabulary
is the one every consumer shares -- purity, native and GPU eligibility,
code motion, inlining, parallelization, autodiff, fusion, the async
lowering, and plugin contracts all read the same names -- and it names
memory apart from objects: `ReadMemory`/`WriteMemory` are native memory a
pointer or buffer reaches, `ReadObject`/`WriteObject` are Python objects.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = [
    "EMPTY",
    "GPU_FORBIDDEN",
    "PURE_FORBIDDEN",
    "REMOVABLE",
    "Effect",
    "EffectSet",
    "EffectSummary",
]


class Effect(enum.StrEnum):
    ALLOC = "Alloc"
    READ_OBJECT = "ReadObject"
    WRITE_OBJECT = "WriteObject"
    READ_MEMORY = "ReadMemory"
    WRITE_MEMORY = "WriteMemory"
    READ_GLOBAL = "ReadGlobal"
    WRITE_GLOBAL = "WriteGlobal"
    IO = "IO"
    NETWORK = "Network"
    RANDOM = "Random"
    TIME = "Time"
    THREAD = "Thread"
    PROCESS = "Process"
    SYNC = "Sync"
    ATOMIC = "Atomic"
    MAY_RAISE = "MayRaise"
    PYTHON_CALLBACK = "PythonCallback"
    PYTHON_DYNAMIC = "PythonDynamic"
    GPU_LAUNCH = "GpuLaunch"
    DEVICE_MEMORY = "DeviceMemory"
    EXTERNAL_UNKNOWN = "ExternalUnknown"

    @property
    def spelled(self) -> str:
        """The lower-case spelling the IR and the docs use: `write_memory`."""
        return _SPELLINGS[self]


_SPELLINGS: dict[str, str] = {}


def _spell(name: str) -> str:
    out = []
    for index, char in enumerate(name):
        if char.isupper() and index and not name[index - 1].isupper():
            out.append("_")
        out.append(char.lower())
    return "".join(out).replace("i_o", "io").replace("gpu_launch", "gpu_launch")


for _effect in Effect:
    _SPELLINGS[_effect] = _spell(_effect.value)

#: Effects that a `@ppy.pure` function may not have (spec 11.2).
#: `Alloc`, `ReadObject`, `ReadMemory`, and `MayRaise` are compatible with purity.
PURE_FORBIDDEN = frozenset(
    {
        Effect.WRITE_OBJECT,
        Effect.WRITE_MEMORY,
        Effect.READ_GLOBAL,
        Effect.WRITE_GLOBAL,
        Effect.IO,
        Effect.NETWORK,
        Effect.RANDOM,
        Effect.TIME,
        Effect.THREAD,
        Effect.PROCESS,
        Effect.SYNC,
        Effect.ATOMIC,
        Effect.PYTHON_CALLBACK,
        Effect.PYTHON_DYNAMIC,
        Effect.GPU_LAUNCH,
        Effect.DEVICE_MEMORY,
        Effect.EXTERNAL_UNKNOWN,
    }
)

#: Effects an unused call may be dropped with: nothing observable, and
#: nothing that raises. A call that only allocates and reads has no result
#: anyone can see once its value is unused.
REMOVABLE = frozenset({Effect.ALLOC, Effect.READ_OBJECT, Effect.READ_MEMORY, Effect.READ_GLOBAL})

#: Effects that keep a function off the GPU: anything that reaches the host
#: interpreter, its files, or its clocks.
GPU_FORBIDDEN = frozenset(
    {
        Effect.READ_OBJECT,
        Effect.WRITE_OBJECT,
        Effect.READ_GLOBAL,
        Effect.WRITE_GLOBAL,
        Effect.IO,
        Effect.NETWORK,
        Effect.TIME,
        Effect.THREAD,
        Effect.PROCESS,
        Effect.SYNC,
        Effect.PYTHON_CALLBACK,
        Effect.PYTHON_DYNAMIC,
        Effect.EXTERNAL_UNKNOWN,
    }
)


@dataclass(frozen=True, slots=True)
class EffectSet:
    effects: frozenset[Effect] = frozenset()
    raises: frozenset[str] = frozenset()

    @staticmethod
    def of(*effects: Effect, raises: Iterable[str] = ()) -> EffectSet:
        found = set(effects)
        raised = frozenset(raises)
        if raised:
            found.add(Effect.MAY_RAISE)
        return EffectSet(frozenset(found), raised)

    def __or__(self, other: EffectSet) -> EffectSet:
        return EffectSet(self.effects | other.effects, self.raises | other.raises)

    def __contains__(self, effect: Effect) -> bool:
        return effect in self.effects

    def add(self, *effects: Effect, raises: Iterable[str] = ()) -> EffectSet:
        return self | EffectSet.of(*effects, raises=raises)

    @property
    def is_pure(self) -> bool:
        overlap = self.effects & PURE_FORBIDDEN
        return not overlap

    @property
    def may_raise(self) -> bool:
        return Effect.MAY_RAISE in self.effects

    @property
    def is_empty(self) -> bool:
        return not self.effects

    def violations(self) -> frozenset[Effect]:
        return frozenset(self.effects & PURE_FORBIDDEN)

    @property
    def is_removable(self) -> bool:
        """May a call with these effects be dropped when its result is unused?"""
        return self.effects <= REMOVABLE

    @property
    def gpu_blockers(self) -> frozenset[Effect]:
        return frozenset(self.effects & GPU_FORBIDDEN)

    def spelled(self) -> tuple[str, ...]:
        """The effects as the IR writes them: sorted, lower-case names."""
        return tuple(sorted(e.spelled for e in self.effects))

    @staticmethod
    def parse(names: Iterable[str]) -> EffectSet:
        """The set the IR spelled; unknown names are an error."""
        by_spelling = {e.spelled: e for e in Effect}
        found: set[Effect] = set()
        for name in names:
            effect = by_spelling.get(str(name))
            if effect is None:
                raise ValueError(f"unknown effect {name!r}")
            found.add(effect)
        return EffectSet(frozenset(found))

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
