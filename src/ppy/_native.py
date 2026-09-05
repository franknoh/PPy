"""A `.ppy` module served natively, when a build of it exists or can be made.

`import ppy` installs a finder that loads `.ppy` modules as Python source.
This is the step after that: when the finder has found `kernel.ppy`, it asks
here whether a native build of it can be served instead -- the same artifact
`ppy run` builds and launches, made once into the project's cache and bound
through the runtime's prebuilt binder. A program that carves its kernels into
`.ppy` modules then gets them native with `import ppy` and nothing else: no
bootstrap, no launcher, and the rest of the program is the Python it was.

Every failure is soft. No compiler installed, no LLVM, a check error in the
kernel, a build that needs the in-process JIT: the module loads as Python
source, and one line on stderr says why. `PPY_IMPORT=python` turns the native
path off for a process; `native-import = false` under `[tool.ppy]` turns it
off for a project; `PPY_QUIET=1` silences the notes.

The runtime never *requires* the compiler. This module asks for it, lazily,
and takes no for an answer; `ppy_runtime` itself does not import it.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

__all__ = ["managed", "native_import", "native_imports", "spec_for"]

_ENV = "PPY_IMPORT"
_QUIET = "PPY_QUIET"
#: Where the compiler's half lives; a test may point it elsewhere.
_BRIDGE = "PPY_NATIVE_BRIDGE"


class _State:
    """What this process has decided and attached so far."""

    __slots__ = ("attached", "enabled", "manifests", "off", "said")

    def __init__(self) -> None:
        #: realpath of a `.ppy` source -> (its generated module, the binder
        #: that serves its natives), from every manifest attached so far.
        self.attached: dict[str, tuple[object, object]] = {}
        self.manifests: set[str] = set()
        #: Why native imports are off for this process, once decided.
        self.off: str | None = None
        self.enabled: bool | None = None
        self.said: set[str] = set()


_state = _State()


def native_import(enabled: bool | None = None) -> bool:
    """Whether `.ppy` imports may be served natively; pass a bool to set it.

    The default is on, unless `PPY_IMPORT=python`; a program may decide for
    itself before its first `.ppy` import.
    """
    if enabled is not None:
        _state.enabled = bool(enabled)
    if _state.enabled is not None:
        return _state.enabled
    return os.environ.get(_ENV, "native") != "python"


def native_imports() -> dict[str, tuple[str, ...]]:
    """What has been served natively so far: module name -> its bound functions.

    A program that wants to know whether its kernels came in native -- a
    benchmark, a parity test -- reads this after importing them.
    """
    served: dict[str, tuple[str, ...]] = {}
    for generated, binder in _state.attached.values():
        name = generated.name  # type: ignore[attr-defined]
        if name in served:
            continue
        names = getattr(binder, "names", None)
        served[name] = tuple(sorted(names(name))) if names is not None else ()
    return served


def managed(reason: str = "the compiler's own loader serves this process") -> None:
    """The compiler is running this program (`ppy run`): stand down."""
    _state.off = reason


def spec_for(fullname: str, filename: str) -> ModuleSpec | None:
    """A spec that loads `filename` natively, or None to load it as source."""
    if _state.off is not None or not native_import():
        return None
    real = os.path.realpath(filename)
    found = _state.attached.get(real)
    if found is None:
        manifest = _build(Path(real))
        if manifest is None:
            return None
        _attach(manifest)
        found = _state.attached.get(real)
        if found is None:
            _note(f"{os.path.basename(real)}: not part of its own build; serving Python")
            return None
    generated, binder = found
    from ppy_runtime.execute import GeneratedLoader

    spec = ModuleSpec(fullname, GeneratedLoader(generated, binder), origin=filename)
    spec.has_location = True
    return spec


def _build(path: Path) -> Path | None:
    """The manifest of `path`'s native build: found in the cache, or made now.

    The work is the compiler's (`ppy_compiler.driver.native_import`), asked
    for by name so that this package never depends on it: no compiler, and
    every `.ppy` loads as source.
    """
    try:
        # The name is read at run time so that the analyzer does not follow
        # the import: `ppy` does not depend on the compiler, it asks for it.
        bridge = importlib.import_module(
            os.environ.get(_BRIDGE, "ppy_compiler.driver.native_import")
        )
    except ImportError:
        _state.off = "the compiler is not installed; .ppy modules load as Python"
        _note(_state.off)
        return None
    try:
        outcome = bridge.manifest_for(path, _note)
    except Exception as error:  # noqa: BLE001 - a failed build must never fail an import
        _note(f"{path.name}: {type(error).__name__}: {error}; serving Python")
        return None
    if outcome.off is not None:
        _state.off = outcome.off
        _note(outcome.off)
        return None
    return outcome.manifest


def _attach(manifest_path: Path) -> None:
    """Open a build and remember which sources it serves."""
    key = str(manifest_path)
    if key in _state.manifests:
        return
    from ppy_runtime.launch import PrebuiltBinder, generated_modules
    from ppy_runtime.manifest import ManifestError, load

    try:
        manifest = load(manifest_path)
    except ManifestError as error:
        _note(f"{error}; serving Python")
        return
    library = None
    if manifest.library is not None:
        try:
            library = ctypes.CDLL(str(manifest.library))
        except OSError as error:
            _note(f"cannot load {manifest.library.name}: {error}; serving Python")
            return
    binder = PrebuiltBinder(manifest, library)
    modules = generated_modules(manifest)
    for generated in modules.values():
        _state.attached.setdefault(os.path.realpath(generated.source_path), (generated, binder))
    _state.manifests.add(key)
    served = ", ".join(sorted(m.name for m in modules.values()))
    _note(f"native: {served}")


def _note(message: str) -> None:
    if os.environ.get(_QUIET) or message in _state.said:
        return
    _state.said.add(message)
    sys.stderr.write(f"[ppy] {message}\n")
