"""Passes over the IR that every pipeline shares."""

from __future__ import annotations

from .canonicalize import Canonicalize, ConstantFold, canonicalization_patterns
from .dce import DeadCodeElimination
from .simplify_cfg import SimplifyCFG

__all__ = [
    "Canonicalize",
    "ConstantFold",
    "DeadCodeElimination",
    "SimplifyCFG",
    "canonicalization_patterns",
    "default_pipeline",
]


def default_pipeline(level: int = 1, ctx=None):  # type: ignore[no-untyped-def]
    """The passes a module goes through before a backend, by optimization level."""
    from ..passes import PassManager

    manager = PassManager(ctx)
    manager.add_stage("after-ir-generation")
    manager.add(Canonicalize())
    manager.add_stage("after-canonicalization")
    manager.add_stage("before-optimization")
    if level >= 1:
        manager.add(SimplifyCFG())
        manager.add(DeadCodeElimination())
        manager.add(Canonicalize())
        manager.add(SimplifyCFG())
    manager.add_stage("after-optimization")
    manager.add(DeadCodeElimination())
    manager.add_stage("before-backend")
    return manager
