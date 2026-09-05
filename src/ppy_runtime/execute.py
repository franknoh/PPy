"""Execution of generated Python with `.ppy` traceback mapping (spec 15.6).

Runtime-only: a built artifact executes its generated modules through this
module with no compiler installed.
"""

from __future__ import annotations

import builtins
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .generated import (
    BINDER_NAME,
    EXPORTED_BINDER,
    FUSED_BINDER,
    REGION_BINDER,
    GeneratedModule,
)

__all__ = ["ExecutionResult", "NativeBinder", "execute", "install_loader"]


class NativeBinder(Protocol):
    """Supplies guarded native entry points for one module."""

    def names(self, module: str) -> frozenset[str]: ...

    def bind(self, module: str, function: str, fallback: object) -> object: ...

    def fused(self, module: str, symbol: str, fallback: object) -> object: ...

    def exported_names(self, module: str) -> frozenset[str]: ...

    def exported(self, module: str, function: str, fallback: object) -> object: ...

    def region_names(self, module: str) -> frozenset[str]: ...

    def region(self, module: str, function: str, fallback: object) -> object: ...


def _prepare_natives(
    namespace: dict,
    module_name: str,
    natives: NativeBinder | None,
    generated: GeneratedModule | None = None,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Install the binding hooks a generated module calls as it loads."""
    if natives is None:
        return frozenset(), frozenset(), frozenset()
    if generated is not None and generated.needs_fused_binder:
        namespace[FUSED_BINDER] = lambda symbol, fallback: natives.fused(
            module_name, symbol, fallback
        )
    exported = frozenset()
    if getattr(natives, "exported_names", None) is not None:
        exported = natives.exported_names(module_name)
        if exported:
            namespace[EXPORTED_BINDER] = lambda function, value: natives.exported(
                module_name, function, value
            )
    regions = frozenset()
    if getattr(natives, "region_names", None) is not None:
        regions = natives.region_names(module_name)
        if regions:
            namespace[REGION_BINDER] = lambda function, value: natives.region(
                module_name, function, value
            )
    names = natives.names(module_name)
    if names:
        namespace[BINDER_NAME] = lambda function, value: natives.bind(module_name, function, value)
    return (names, exported, regions)


@dataclass(slots=True)
class ExecutionResult:
    exit_code: int
    exception: BaseException | None = None


class _GeneratedFinder:
    """Serves generated modules to `import` while keeping `.ppy` identities."""

    def __init__(self, modules: dict[str, GeneratedModule], natives=None) -> None:
        self.modules = modules
        self.natives = natives

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[no-untyped-def]
        from importlib.machinery import ModuleSpec

        generated = self.modules.get(fullname)
        if generated is None:
            return None
        spec = ModuleSpec(
            fullname, GeneratedLoader(generated, self.natives), origin=str(generated.source_path)
        )
        spec.has_location = True
        return spec


class GeneratedLoader:
    """Executes a generated module with its native bindings in place."""

    def __init__(self, generated: GeneratedModule, natives=None) -> None:
        self.generated = generated
        self.natives = natives

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        module.__file__ = str(self.generated.source_path)
        names, exported, regions = _prepare_natives(
            module.__dict__, self.generated.name, self.natives, self.generated
        )
        exec(self.generated.compile(names, exported, regions), module.__dict__)

    def get_source(self, fullname: str) -> str:
        return self.generated.source_path.read_text(encoding="utf-8")


def install_loader(modules: dict[str, GeneratedModule], natives=None) -> _GeneratedFinder:
    """Put the generated finder ahead of everything, and keep it there.

    A program that does `import ppy` installs the runtime's own `.ppy` finder
    at the front of `sys.meta_path`, which would then serve a sibling module
    from source and lose its native bindings. Installing that hook first means
    the program's own `import ppy` finds it already present and leaves the
    order alone.
    """
    try:
        import ppy
        from ppy import _native

        ppy.install()
        _native.managed()
    except ImportError:  # pragma: no cover - the runtime is a hard dependency
        pass
    finder = _GeneratedFinder(modules, natives)
    sys.meta_path.insert(0, finder)
    return finder


def execute(
    entry: GeneratedModule,
    modules: dict[str, GeneratedModule],
    argv: list[str],
    *,
    search_paths: list[Path] | None = None,
    natives: NativeBinder | None = None,
    entry_name: str | None = None,
) -> ExecutionResult:
    """Run the generated entry module as `__main__`."""
    finder = install_loader({name: m for name, m in modules.items() if m is not entry}, natives)
    saved_argv = sys.argv[:]
    saved_path = sys.path[:]
    for extra in reversed(search_paths or []):
        sys.path.insert(0, str(extra))
    sys.argv = [str(entry.source_path), *argv]

    namespace = {
        "__name__": "__main__",
        "__file__": str(entry.source_path),
        "__builtins__": builtins,
        "__package__": None,
        "__spec__": None,
        "__doc__": None,
    }
    module = types.ModuleType("__main__")
    module.__dict__.update(namespace)
    saved_main = sys.modules.get("__main__")
    sys.modules["__main__"] = module
    try:
        names, exported, regions = _prepare_natives(
            module.__dict__, entry_name or entry.name, natives, entry
        )
        exec(entry.compile(names, exported, regions), module.__dict__)
        return ExecutionResult(0)
    except SystemExit as exit_request:
        code = exit_request.code
        return ExecutionResult(code if isinstance(code, int) else (0 if code is None else 1))
    except BaseException as exc:  # noqa: BLE001 - reported to the user verbatim
        return ExecutionResult(1, exc)
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path
        if saved_main is not None:
            sys.modules["__main__"] = saved_main
        else:
            sys.modules.pop("__main__", None)
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


def format_traceback(exc: BaseException) -> str:
    import traceback

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
