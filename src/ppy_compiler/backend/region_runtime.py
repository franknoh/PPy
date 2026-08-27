"""Guarded dispatch into a compiled ATen region (spec 20.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = ["RegionBinding", "bind_region"]


@dataclass(slots=True)
class RegionBinding:
    function: str
    wrapper: Callable[..., object]
    fallback: Callable[..., object]
    calls: int = 0
    fallbacks: int = 0
    reason: str = ""

    @property
    def routed(self) -> bool:
        return self.wrapper is not self.fallback


def bind_region(
    function: str,
    compiled: Callable[..., object] | None,
    fallback: Callable[..., object],
) -> RegionBinding:
    """Wrap a compiled region in the guards its curated domain requires.

    Every `at::` call inside the region goes through the dispatcher, so
    autograd and device selection need no guard. What does need one is the
    Python override machinery the region bypasses (spec 20.6).
    """
    if compiled is None:
        return RegionBinding(function, fallback, fallback, reason="the region was not compiled")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch absent
        return RegionBinding(function, fallback, fallback, reason=str(exc))

    tensor = torch.Tensor
    has_override = torch.overrides.has_torch_function
    binding = RegionBinding(function, fallback, fallback)

    def wrapper(*args: object, **kwargs: object) -> object:
        if kwargs:
            binding.fallbacks += 1
            return fallback(*args, **kwargs)
        for argument in args:
            if isinstance(argument, tensor) and type(argument) is not tensor:
                binding.fallbacks += 1
                return fallback(*args)
        if has_override(args):
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        return compiled(*args)

    wrapper.__name__ = function
    wrapper.__ppy_region__ = True  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding
