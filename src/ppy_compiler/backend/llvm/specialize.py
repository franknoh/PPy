"""Runtime specialization of natively lowered functions (spec 16.9, 27.6).

A `@ppy.jit` function is compiled once generically. When the same argument
shape keeps arriving -- a repeated constant, a repeated buffer length -- a
second copy is compiled with those values pinned, which is what lets LLVM fold,
unroll, and vectorize around them. Every specialization is selected by a guard
on the exact key it was built for, and anything else uses the generic code.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ...analysis.symbols import FunctionInfo
from ...cache import CacheKey, digest
from .lowering import NativeSignature, lower_specialization

__all__ = [
    "DEFAULT_MAX",
    "DEFAULT_THRESHOLD",
    "Specialization",
    "SpecializationKey",
    "SpecializationPolicy",
    "Specializer",
    "key_for",
]

#: Calls with the same key before a specialization is worth compiling.
DEFAULT_THRESHOLD = 8

#: Cap on distinct specializations per function, bounding code growth.
DEFAULT_MAX = 4

#: Calls to keep watching before concluding the arguments never settle. Past
#: this the wrapper stops building keys, so a function whose arguments always
#: differ pays nothing for having asked.
LEARNING_BUDGET = 64

#: Only these ever become part of a key: a value that is cheap to guard.
_PINNABLE = {"int", "float", "bool"}


@dataclass(frozen=True, slots=True)
class SpecializationKey:
    """The argument facts one specialization pins."""

    entries: tuple[tuple[int, str, str, object], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)

    @property
    def constants(self) -> dict[str, object]:
        return {
            (name if kind == "value" else f"len({name})"): value
            for _index, name, kind, value in self.entries
        }

    def suffix(self) -> str:
        return digest(self.entries)[:12]

    def pins(self) -> tuple[tuple[int, int, object], ...]:
        """The pinned facts as `(kind, argument index, value)` triples.

        Kinds match the generated wrapper: 1 an integer value, 2 a float
        value, 3 a length.
        """
        described: list[tuple[int, int, object]] = []
        for index, _name, kind, value in self.entries:
            if kind == "length":
                described.append((3, index, int(value)))  # type: ignore[arg-type]
            elif isinstance(value, float):
                described.append((2, index, value))
            else:
                described.append((1, index, int(value)))  # type: ignore[arg-type]
        return tuple(described)

    def matcher(self) -> Callable[[tuple[object, ...]], bool]:
        """A cheap predicate selecting exactly the calls this was built for.

        This runs on every call once a specialization exists, so it compares
        the pinned values directly rather than rebuilding and hashing a key.
        """
        checks = tuple(
            (index, kind == "length", value) for index, _name, kind, value in self.entries
        )
        if len(checks) == 1:
            index, by_length, value = checks[0]
            if by_length:
                return lambda args: len(args[index]) == value  # type: ignore[arg-type]
            return lambda args: args[index] == value

        def matches(args: tuple[object, ...]) -> bool:
            for index, by_length, value in checks:
                actual = len(args[index]) if by_length else args[index]  # type: ignore[arg-type]
                if actual != value:
                    return False
            return True

        return matches

    def __str__(self) -> str:
        return ", ".join(
            f"{name}=={value!r}" if kind == "value" else f"len({name})=={value}"
            for _index, name, kind, value in self.entries
        )


@dataclass(frozen=True, slots=True)
class SpecializationPolicy:
    """What the directives asked for (spec 6.2)."""

    enabled: bool = False
    threshold: int = DEFAULT_THRESHOLD
    maximum: int = DEFAULT_MAX
    pin_values: bool = True
    pin_lengths: bool = True
    budget: int = LEARNING_BUDGET

    @staticmethod
    def of(info: FunctionInfo) -> SpecializationPolicy:
        jit = info.directive("jit")
        specialize = info.directive("specialize")
        if jit is None and specialize is None:
            return SpecializationPolicy(enabled=False)
        options = dict(jit.options if jit else {})
        options.update(specialize.options if specialize else {})
        maximum = options.get("max_specializations", DEFAULT_MAX)
        threshold = options.get("threshold", 1 if specialize is not None else DEFAULT_THRESHOLD)
        return SpecializationPolicy(
            enabled=True,
            threshold=max(1, int(threshold)) if isinstance(threshold, int) else DEFAULT_THRESHOLD,
            maximum=max(0, int(maximum)) if isinstance(maximum, int) else DEFAULT_MAX,
            pin_values=bool(options.get("values", True)),
            pin_lengths=bool(options.get("lengths", True)),
            budget=int(options.get("budget", LEARNING_BUDGET)),
        )


def key_for(
    signature: NativeSignature,
    args: tuple[object, ...],
    policy: SpecializationPolicy,
) -> SpecializationKey:
    """The key describing what is worth pinning about this call."""
    entries: list[tuple[int, str, str, object]] = []
    for index, (parameter, value) in enumerate(zip(signature.parameters, args, strict=False)):
        if parameter.is_buffer:
            if policy.pin_lengths:
                try:
                    entries.append((index, parameter.name, "length", len(value)))  # type: ignore[arg-type]
                except TypeError:
                    continue
            continue
        if not policy.pin_values or parameter.kind not in _PINNABLE:
            continue
        # A bool pins almost nothing and is not an exact `int` to the
        # generated guard, so it is left alone.
        if type(value) not in (int, float):
            continue
        entries.append((index, parameter.name, "value", value))
    return SpecializationKey(tuple(entries))


@dataclass(slots=True)
class Specialization:
    key: SpecializationKey
    symbol: str
    address: int = 0
    ir: str = ""
    calls: int = 0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.address != 0


@dataclass(slots=True)
class Specializer:
    """Compiles and hands out specializations for one module's functions."""

    engine: object
    module_analysis: object
    cache: object | None = None
    cache_directory: Path | None = None
    layouts: dict = field(default_factory=dict)
    compiled: list[Specialization] = field(default_factory=list)
    refusals: list[tuple[str, str]] = field(default_factory=list)
    _sources: dict[str, tuple[FunctionInfo, ast.FunctionDef]] = field(default_factory=dict)

    def register(self, info: FunctionInfo, node: ast.FunctionDef) -> None:
        self._sources[info.qualname] = (info, node)

    def specialize(
        self,
        info: FunctionInfo,
        key: SpecializationKey,
    ) -> Specialization | None:
        """Compile one specialization, or explain why it was not compiled."""
        entry = self._sources.get(info.qualname)
        if entry is None or not key:
            return None
        _info, node = entry
        symbol = f"ppy_{info.qualname.replace('.', '_')}__spec_{key.suffix()}"
        specialization = Specialization(key=key, symbol=symbol)
        try:
            text = lower_specialization(
                self.module_analysis, info, node, key.constants, symbol, self.layouts
            )
        except Exception as exc:  # noqa: BLE001 - a refusal keeps the generic code
            specialization.reason = f"could not lower: {exc}"
            self.refusals.append((info.qualname, specialization.reason))
            return specialization

        specialization.ir = text
        try:
            self.engine.add(text)  # type: ignore[attr-defined]
            self.engine.finalize()  # type: ignore[attr-defined]
            specialization.address = self.engine.address(symbol)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            specialization.reason = f"could not compile: {exc}"
            self.refusals.append((info.qualname, specialization.reason))
            return specialization

        self._publish(info, specialization)
        self.compiled.append(specialization)
        return specialization

    def _publish(self, info: FunctionInfo, specialization: Specialization) -> None:
        """Record the specialized IR so `ppy inspect` can show it (spec 27.6)."""
        if self.cache is None:
            return
        key = CacheKey.build(
            "jit",
            source_digest=digest(specialization.ir),
            compiler_version="0.1.0",
            opt_level=3,
            target="jit",
            directives=(f"{info.qualname}:{specialization.key}",),
        )
        try:
            self.cache.put(  # type: ignore[attr-defined]
                key,
                specialization.ir,
                kind="jit",
                source=info.qualname,
                suffix=".ll",
            )
        except Exception:  # noqa: BLE001 - caching is never load-bearing
            return
