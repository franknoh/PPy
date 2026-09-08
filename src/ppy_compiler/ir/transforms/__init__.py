"""Passes over the IR that every pipeline shares."""

from __future__ import annotations

from .autodiff import AutodiffError, differentiate
from .canonicalize import Canonicalize, ConstantFold, canonicalization_patterns
from .dce import DeadCodeElimination
from .fuse_tensor import FuseTensor, TensorCanonicalize
from .lower_async import AsyncLoweringError, LowerAsync, lower_async
from .lower_parallel import BACKENDS as PARALLEL_BACKENDS
from .lower_parallel import LowerParallel
from .lower_tensor import LoweringError, LowerTensor
from .promote_slots import PromoteSlots, promote_slots
from .sanitize import Sanitize, sanitize, sanitizer_kinds
from .simplify_cfg import SimplifyCFG
from .whole_program import GlobalDCE, Inline, internalize, whole_program

__all__ = [
    "PARALLEL_BACKENDS",
    "AsyncLoweringError",
    "AutodiffError",
    "Canonicalize",
    "ConstantFold",
    "DeadCodeElimination",
    "FuseTensor",
    "GlobalDCE",
    "Inline",
    "LowerAsync",
    "LowerParallel",
    "LowerTensor",
    "LoweringError",
    "PromoteSlots",
    "Sanitize",
    "SimplifyCFG",
    "TensorCanonicalize",
    "canonicalization_patterns",
    "default_pipeline",
    "differentiate",
    "internalize",
    "lower_async",
    "promote_slots",
    "sanitize",
    "sanitizer_kinds",
    "whole_program",
]


def default_pipeline(level: int = 1, ctx=None, parallel=None, sanitize=(), until=None):  # type: ignore[no-untyped-def]
    """The passes a module goes through before a backend, by optimization level.

    `parallel` is the `LowerParallel` pass for the build's configuration;
    it runs after the optimizations and the result is cleaned up again.
    `sanitize` names the sanitizers a build asked for; their checks go in
    before the optimizations, so what the optimizer removes they still
    hold. `until` stops the pipeline at a point -- `none` before any pass,
    `after-canonicalization`, `after-fusion` -- for `ppy inspect --stage`.
    """
    from ..passes import PassManager

    manager = PassManager(ctx)
    if until == "none":
        return manager
    manager.add_stage("after-ir-generation")
    manager.add(LowerAsync())
    manager.add(Canonicalize())
    manager.add_stage("after-canonicalization")
    if until == "after-canonicalization":
        return manager
    if sanitize:
        manager.add(Sanitize(frozenset(sanitize)))
    manager.add_stage("before-optimization")
    if level >= 1:
        manager.add(SimplifyCFG())
        manager.add(DeadCodeElimination())
        manager.add(Canonicalize())
        manager.add(SimplifyCFG())
    manager.add_stage("after-optimization")
    manager.add(DeadCodeElimination())
    if level >= 1:
        manager.add(TensorCanonicalize())
        manager.add(FuseTensor())
    if until == "after-fusion":
        return manager
    manager.add(LowerTensor())
    if parallel is not None:
        manager.add(parallel)
        manager.add(Canonicalize())
        manager.add(SimplifyCFG())
        manager.add(DeadCodeElimination())
    manager.add_stage("before-backend")
    return manager
