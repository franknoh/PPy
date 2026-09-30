"""Which settled module globals native code passes each function.

A settled global (`ModuleSymbols.settled_globals`) is bound once by the
module's body and never rebound, so the object a name refers to when a
function is called is the one it will refer to throughout the call. Native
code takes each one the function reads as one more parameter: Python's
boundary reads the global when it calls the function, and a native caller
passes the one it was given. A caller therefore takes every global its native
callees read, and this pass closes each function's set over the calls it
makes.
"""

from __future__ import annotations

from .results import FunctionAnalysis, ImplicitGlobal, ProjectAnalysis

__all__ = ["close_settled_globals", "implicit_name", "implicit_parameter_name"]


def implicit_name(module: str, name: str) -> str:
    """The parameter a global the function does not read by name is passed as."""
    return f"__global_{module.replace('.', '_')}_{name}"


def _plain(analysis: FunctionAnalysis) -> bool:
    """A function of the module that can take more parameters than it spells:
    not a method, a closure, a generic, a generator, or a coroutine, whose
    callers do not pass arguments the way a native call to a function does."""
    info = analysis.info
    return (
        info.owner is None
        and info.enclosing is None
        and not info.type_params
        and not info.is_generator
        and not info.is_async
    )


def close_settled_globals(project: ProjectAnalysis) -> None:
    """Fill `implicit_globals` and `globals_native` for every function."""
    functions: dict[str, FunctionAnalysis] = {}
    for module in project.modules.values():
        functions.update(module.functions)
    found: dict[str, dict[tuple[str, str], ImplicitGlobal]] = {}
    native: dict[str, bool] = {}
    for qualname, analysis in functions.items():
        module = analysis.info.module
        writes = analysis.mutated_params | analysis.delegated_writes
        found[qualname] = {
            (module, name): ImplicitGlobal(module, name, kind, name in writes)
            for name, kind in analysis.settled_globals.items()
        }
        native[qualname] = not analysis.unsettled_global
    changed = True
    while changed:
        changed = False
        for qualname, analysis in functions.items():
            own = found[qualname]
            ok = native[qualname]
            for callee in analysis.calls:
                if callee == qualname or callee not in found:
                    continue
                ok = ok and native[callee]
                for key, wanted in found[callee].items():
                    held = own.get(key)
                    if held is None or (wanted.written and not held.written):
                        own[key] = ImplicitGlobal(
                            wanted.module,
                            wanted.name,
                            wanted.type,
                            wanted.written or (held is not None and held.written),
                        )
                        changed = True
            if own and not _plain(analysis):
                ok = False
            if ok != native[qualname]:
                native[qualname] = ok
                changed = True
    for qualname, analysis in functions.items():
        analysis.implicit_globals = tuple(found[qualname][key] for key in sorted(found[qualname]))
        analysis.globals_native = native[qualname]


def implicit_parameter_name(analysis: FunctionAnalysis, held: ImplicitGlobal) -> str:
    """The name `held` has inside the function: the global's own when the body
    reads it by that name, a spelled-out one when only a callee does."""
    if held.module == analysis.info.module and held.name in analysis.settled_globals:
        return held.name
    return implicit_name(held.module, held.name)
