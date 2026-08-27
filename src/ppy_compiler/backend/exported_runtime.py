"""Calling a build-time exported graph region (spec 21.3, 21.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = ["ExportedBinding", "bind_exported"]


@dataclass(slots=True)
class ExportedBinding:
    function: str
    wrapper: Callable[..., object]
    fallback: Callable[..., object]
    calls: int = 0
    fallbacks: int = 0
    reason: str = ""
    last_error: str = ""

    @property
    def routed(self) -> bool:
        return self.wrapper is not self.fallback


def bind_exported(function: str, payload: bytes, fallback: Callable[..., object]) -> ExportedBinding:
    """Rehydrate an exported computation, falling back to the staged function.

    The exported artifact carries its own StableHLO and calling metadata, so
    invoking it does not re-trace the Python source. Shapes outside the
    exported signature fall back, which is what shape polymorphism does not
    cover (spec 21.5).
    """
    try:
        from ..plugins.jax_export import runtime_call
    except ImportError as exc:  # pragma: no cover - jax absent
        return ExportedBinding(function, fallback, fallback, reason=str(exc))

    try:
        call = runtime_call(payload)
    except Exception as exc:  # noqa: BLE001 - a mismatched runtime must not break the program
        return ExportedBinding(
            function, fallback, fallback,
            reason=f"the exported artifact does not load here: {exc}",
        )

    binding = ExportedBinding(function, fallback, fallback)

    def wrapper(*args: object, **kwargs: object) -> object:
        if kwargs:
            binding.fallbacks += 1
            return fallback(*args, **kwargs)
        try:
            result = call(*args)
        except Exception as exc:  # noqa: BLE001 - a mismatch is a guard failure
            # Recorded rather than swallowed: an artifact that never routes is
            # something the developer needs to be able to see.
            binding.last_error = f"{type(exc).__name__}: {exc}"
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        return result

    wrapper.__name__ = function
    wrapper.__ppy_exported__ = True  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding
