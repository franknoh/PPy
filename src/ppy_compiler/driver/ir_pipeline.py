"""The shared road from the typed program to the IR every backend receives.

`canonical_ir_modules` is the canonical IR of every module in a project after
the shared passes; `optimize_shared_ir` is those passes over one module.
Neither knows a backend: the LLVM backend, the source backends, the device
backends, `ppy emit ir`, and an external backend all take what they hand
back. What happens after -- `prepare_for_backend` -- is the backend's own
passes and its validation, and only then its emission.

The order, for one module:

1. canonical IR generation (`ppy_compiler.lowering`), then the
   `after-ir-generation` plugin passes;
2. canonicalization, then `after-canonicalization`;
3. `before-optimization`, the shared optimization passes, `after-optimization`;
4. tensor and parallel lowering, then `before-backend`;
5. the `backend` stage, where a backend's `register_passes` hangs its own;
6. the backend's `validate`;
7. the backend's `emit` or `build`.

The module is verified before the passes, after each of them when a plugin
or a backend contributed one, and after the backend's, so a pass that
leaves the IR invalid is named rather than handed on.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator

from ..ir import IRModule, PassContext, PassManager, verify_or_raise
from ..ir.transforms import default_pipeline

__all__ = [
    "canonical_ir_modules",
    "definitions",
    "optimize_shared_ir",
    "parallel_pass",
    "prepare_for_backend",
    "value_class_layouts",
]


def _verify_between_passes() -> bool:
    return bool(os.environ.get("PPY_IR_VERIFY"))


def parallel_pass(parallel):  # type: ignore[no-untyped-def]
    """The `lower-parallel` pass for a `ParallelConfig`, or None for the default."""
    from ..backend.llvm.parallel import _requested_threads
    from ..ir.transforms import LowerParallel

    if parallel is None:
        return LowerParallel("threads", threads=_requested_threads("auto"))
    backend = parallel.backend if parallel.enabled else "serial"
    return LowerParallel(backend, threads=_requested_threads(parallel.threads))


def optimize_shared_ir(  # type: ignore[no-untyped-def]
    module: IRModule,
    level: int,
    plugins=None,
    parallel=None,
    sanitize=(),
    until=None,
    instrument: bool = False,
    profile=None,
    backend=None,
) -> PassContext:
    """Verify the frontend's IR, run the shared passes, verify again.

    With a project's plugins, the registry is the project's -- its dialects,
    patterns, and lowerings -- and the plugins' passes run at their stages,
    each verified so that one which breaks the IR is named. `parallel` is
    the project's `ParallelConfig`, which decides how a parallel loop is
    lowered. A `backend` registers its own passes at the `backend` stage,
    which the pipeline marks last.
    """
    registry = plugins.dialect_registry() if plugins is not None else None
    verify_or_raise(module, registry)
    external = (plugins is not None and len(plugins) > 0) or backend is not None
    ctx = PassContext(registry, verify_after_each=_verify_between_passes() or external)
    manager = default_pipeline(
        level,
        ctx,
        parallel_pass(parallel),
        sanitize=sanitize,
        until=until,
        instrument=instrument,
        profile=profile,
    )
    if plugins is not None:
        plugins.register_passes(manager)
    if backend is not None:
        backend.register_passes(manager)
    manager.run(module)
    verify_or_raise(module, registry)
    return ctx


def definitions(tree: ast.Module) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function a backend may lower, with its owning class if any."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield "", node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield node.name, child


#: Methods that could observe a value class being flattened into scalars.
_INTERCEPTORS = frozenset({"__getattr__", "__getattribute__", "__setattr__", "__init_subclass__"})


def value_class_layouts(bundle) -> dict[str, tuple[tuple[str, str], ...]]:  # type: ignore[no-untyped-def]
    """Classes whose instances can be flattened into scalar arguments.

    A value class here is one with a fixed set of scalar fields and no base
    beyond `object`: nothing about it needs a Python object to represent
    (spec 13.2, 25.4).
    """
    from ..analysis import types as T

    scalars = {"int", "float", "bool"}
    layouts: dict[str, tuple[tuple[str, str], ...]] = {}
    for qualname, info in bundle.symbols.classes.items():
        if info.is_protocol or info.is_enum or info.is_pydantic:
            continue
        if tuple(entry for entry in info.mro if entry != "object") != (qualname,):
            continue
        # Reading a field must be a plain attribute read: anything that can
        # intercept it could observe the flattening.
        if _INTERCEPTORS & set(info.methods):
            continue
        if set(info.fields) & set(info.methods):
            continue
        fields: list[tuple[str, str]] = []
        for name, declared in info.fields.items():
            if name in info.class_vars:
                continue
            base = T.strip_literal(declared)
            if not isinstance(base, T.Instance) or base.name not in scalars:
                fields = []
                break
            fields.append((name, base.name))
        if fields:
            layouts[qualname] = tuple(fields)
    return layouts


def canonical_ir_modules(  # type: ignore[no-untyped-def]
    bundle, launches: bool = False, until=None, backend=None
) -> dict[str, IRModule]:
    """The canonical IR of every module in the project, after the shared passes.

    With `launches`, a function launching a kernel lowers with its launch,
    as the source backends, `ppy emit ir`, and an external backend want it;
    without, it stays in Python, since the CPU backends have no launch
    runtime. With a `backend`, its passes run at the `backend` stage of
    every module; its validation is `prepare_for_backend`'s.
    """
    from ..backend.llvm import prover_for
    from ..lowering import lower_module_to_ir
    from .profile import profile_for

    config = bundle.project.config
    layouts = value_class_layouts(bundle)
    modules: dict[str, IRModule] = {}
    available: dict[str, tuple] = {}  # type: ignore[type-arg]
    for module in bundle.graph.order():
        analysis = bundle.analysis.modules.get(module.name)
        symbols = bundle.symbols.modules.get(module.name)
        if analysis is None or symbols is None:
            continue
        candidates: dict = {}
        for owner, node in definitions(module.tree):
            info = (
                symbols.classes[owner].methods.get(node.name)
                if owner and owner in symbols.classes
                else symbols.functions.get(node.name)
            )
            if info is None:
                continue
            function_analysis = analysis.functions.get(info.qualname)
            if function_analysis is not None:
                candidates[info.qualname] = (info, function_analysis, node)
        if not candidates:
            continue
        lowered = lower_module_to_ir(
            analysis,
            candidates,
            layouts,
            safeguards=config.llvm.safeguards or "hoisted",
            prover=prover_for(config),
            root=bundle.project.root,
            launches=launches,
            imports=available.get,
        )
        if not lowered.functions:
            continue
        for qualname, entry in lowered.functions.items():
            available[qualname] = (entry.info, entry.signature)
        optimize_shared_ir(
            lowered.module,
            config.opt_level,
            bundle.project.plugins,
            config.parallel,
            sanitize=config.llvm.sanitize,
            until=until,
            instrument=config.llvm.instrument,
            profile=profile_for(config),
            backend=backend,
        )
        modules[module.name] = lowered.module
    return modules


def prepare_for_backend(module: IRModule, backend, context) -> None:  # type: ignore[no-untyped-def]
    """Hand `module`, already through the shared passes and the backend's own,
    to the backend's validation: the last word before it emits or builds.

    A backend refuses what it cannot take here, with the operation, type, or
    function named, rather than emitting it wrongly; the driver reports the
    refusal as `E1904`.
    """
    verify_or_raise(module, context.registry)
    backend.validate(module, context)


def backend_pass_manager(backend, context) -> PassManager:  # type: ignore[no-untyped-def]
    """A manager holding only the backend's passes, verified after each: what a
    backend runs over IR it received already optimized (a `.ppyir`, say)."""
    ctx = PassContext(context.registry, verify_after_each=True)
    manager = PassManager(ctx)
    manager.add_stage("backend")
    backend.register_passes(manager)
    return manager
