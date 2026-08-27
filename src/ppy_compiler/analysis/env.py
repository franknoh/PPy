"""Flow-sensitive binding environments."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import types as T
from .refinements import Facts

__all__ = ["Binding", "Env"]


@dataclass(frozen=True, slots=True)
class Binding:
    type: T.Type
    facts: Facts = field(default_factory=Facts)

    def with_type(self, t: T.Type) -> "Binding":
        return Binding(t, self.facts)

    def with_facts(self, facts: Facts) -> "Binding":
        return Binding(self.type, facts)

    def merge(self, other: "Binding") -> "Binding":
        return Binding(T.join(self.type, other.type), self.facts.merge(other.facts))


class Env:
    """A mutable map from local names to bindings, forkable for branches."""

    __slots__ = ("_bindings", "reachable")

    def __init__(self, bindings: dict[str, Binding] | None = None, reachable: bool = True) -> None:
        self._bindings: dict[str, Binding] = dict(bindings or {})
        self.reachable = reachable

    def __contains__(self, name: str) -> bool:
        return name in self._bindings

    def __iter__(self):
        return iter(self._bindings)

    def get(self, name: str) -> Binding | None:
        return self._bindings.get(name)

    def set(self, name: str, binding: Binding) -> None:
        self._bindings[name] = binding

    def remove(self, name: str) -> None:
        self._bindings.pop(name, None)

    def names(self) -> set[str]:
        return set(self._bindings)

    def fork(self) -> "Env":
        return Env(self._bindings, self.reachable)

    def terminate(self) -> None:
        self.reachable = False

    def merge(self, other: "Env") -> "Env":
        """Merge two branch environments, keeping only jointly proven facts."""
        if not self.reachable:
            return other.fork()
        if not other.reachable:
            return self.fork()
        merged: dict[str, Binding] = {}
        for name in self._bindings.keys() | other._bindings.keys():
            left = self._bindings.get(name)
            right = other._bindings.get(name)
            if left is None or right is None:
                # Bound on only one path: keep the type but drop all facts.
                only = left or right
                assert only is not None
                merged[name] = Binding(only.type, Facts())
            else:
                merged[name] = left.merge(right)
        return Env(merged, True)

    def snapshot(self) -> dict[str, Binding]:
        return dict(self._bindings)

    def restore(self, snapshot: dict[str, Binding]) -> None:
        self._bindings = dict(snapshot)

    def equals(self, snapshot: dict[str, Binding]) -> bool:
        return self._bindings == snapshot
