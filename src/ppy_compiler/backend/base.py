"""The backend interface: what a code generator is to the compiler.

A backend consumes the canonical IR after the shared passes and nothing
else -- never the Python AST, never the checker's tables -- and answers
with text, bytes, or built artifacts. It may hang passes of its own at the
`backend` stage, must refuse IR it cannot take in `validate`, and says
whether its toolchain is here. Plugins are the other extension: a plugin
models a library's semantics and lowering; a backend makes code. Neither
does the other's job.

An external backend is a package with an entry point in the `ppy.backends`
group whose factory takes the backend's options from `pyproject.toml`
(`[tool.ppy.backends.<name>]`) and returns a `Backend`::

    [project.entry-points."ppy.backends"]
    toy = "ppy_toy:create_backend"

The interface is versioned by `BACKEND_API_VERSION`, independently of the
compiler's version: a backend written against another one is refused with
the reason, never loaded and hoped about.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..cache.keys import digest
from ..driver.config import BackendConfig

if TYPE_CHECKING:
    from ..ir import DialectRegistry, IRModule, PassManager, SourceLocation

__all__ = [
    "BACKEND_API_VERSION",
    "Backend",
    "BackendConfig",
    "BackendContext",
    "BackendError",
    "BackendUnavailable",
    "BackendValidationError",
    "BuildResult",
    "EmitFormat",
    "ToolchainStatus",
]

#: The interface a backend is written against. Bumped when a method's
#: meaning or signature changes; a backend declaring another version is
#: refused at load time.
BACKEND_API_VERSION = 1


class BackendError(Exception):
    """A backend's refusal; the message is the reason."""


class BackendUnavailable(BackendError):
    """The backend's toolchain is not here; what is missing is named."""


class BackendValidationError(BackendError):
    """IR the backend does not take: the backend, what it cannot take, the
    capability it lacks, and the source location when the IR carries one."""

    def __init__(
        self,
        backend: str,
        what: str,
        *,
        capability: str = "",
        location: SourceLocation | None = None,
    ) -> None:
        self.backend = backend
        self.what = what
        self.capability = capability
        self.location = location
        where = f" at {location}" if location is not None else ""
        lacking = f" (needs {capability})" if capability else ""
        super().__init__(f"backend {backend!r} cannot take {what}{where}{lacking}")


@dataclass(frozen=True, slots=True)
class EmitFormat:
    """One thing a backend can write for `ppy emit <name>`."""

    #: The word after `ppy emit`; unique across every installed backend.
    name: str
    #: The file suffix `-o DIR` writes with, dot included.
    suffix: str
    #: Whether `emit` answers bytes rather than text.
    binary: bool = False
    #: One line for `--help` and `ppy doctor`.
    description: str = ""


@dataclass(frozen=True, slots=True)
class ToolchainStatus:
    """Whether the backend can build here, and what it found or missed."""

    available: bool
    #: `SDK 1.2.3`, or `missing foo`: the words `ppy doctor` prints.
    detail: str = ""


@dataclass(slots=True)
class BuildResult:
    """What `build` produced."""

    #: Every file the build wrote, for the driver to list.
    outputs: tuple[Path, ...] = ()
    #: Remarks for the driver to print as notes (not warnings).
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class BackendContext:
    """Everything a backend is told besides the IR.

    The IR is already through the shared passes and the backend's own;
    nothing here reaches back into the frontend. `identity` is the cache
    key the driver computed for the module (compiler, IR, backend, its
    fingerprint and configuration, target, optimization level, plugins),
    for a backend that keeps artifacts of its own.
    """

    #: The project root; relative paths in the backend's options resolve here.
    root: Path
    #: The whole project configuration, read-only.
    config: object
    #: The backend's own table from `pyproject.toml`.
    backend_config: BackendConfig
    #: The optimization level the run asked for.
    opt_level: int
    #: The target the backend's configuration names (`target = "rngd"`), or "".
    target: str
    #: The dialect registry the IR was verified against: the builtin dialects
    #: plus whatever the project's plugins registered.
    registry: DialectRegistry | None
    #: Plugin fingerprints for the modules this compilation involves.
    plugin_fingerprints: tuple[str, ...] = ()
    #: The artifact identity per module name, once computed.
    identity: dict[str, str] = field(default_factory=dict)
    #: The program's entry file, for a build of one; None for a directory.
    entry: Path | None = None
    #: What the backend wants the driver to print after `notes:`.
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        self.notes.append(text)


class Backend:
    """The class an external backend extends. Every method has a default: a
    backend that emits registers formats and implements `emit`; one that
    builds implements `build`; the rest is optional."""

    #: The backend's name: the word after `--backend`, the entry point's name.
    name: str = ""
    #: The interface version this backend was written against.
    api_version: int = BACKEND_API_VERSION

    def __init__(self, options: Mapping[str, object] | None = None) -> None:
        self.options: dict[str, object] = dict(options or {})

    def fingerprint(self) -> str:
        """What identifies this backend's code generation for the cache: its
        version, its SDK's, anything whose change makes old artifacts wrong.
        The default is the class and the interface version -- a backend with
        a toolchain under it should include that toolchain's version."""
        return digest(type(self).__module__, type(self).__qualname__, self.api_version)

    def emit_formats(self) -> tuple[EmitFormat, ...]:
        """The formats `ppy emit` may ask this backend for."""
        return ()

    def register_passes(self, manager: PassManager) -> None:
        """Hang the backend's passes at the `backend` stage:
        `manager.register_stage_pass("backend", MyPass)`."""

    def validate(self, module: IRModule, context: BackendContext) -> None:
        """Refuse IR the backend cannot take, with a `BackendValidationError`
        naming what and why. Called after every pass, before `emit` or `build`."""

    def emit(  # pylint: disable=redefined-builtin
        self, module: IRModule, format: str, context: BackendContext
    ) -> str | bytes:
        """`format` (one of `emit_formats`) for one module: text, or bytes for
        a binary format."""
        raise BackendError(f"backend {self.name!r} does not emit {format!r}")

    def build(
        self, modules: Mapping[str, IRModule], output: Path, context: BackendContext
    ) -> BuildResult:
        """Build every module into `output`, which exists, and say what was written."""
        raise BackendError(f"backend {self.name!r} does not build")

    def toolchain_status(self) -> ToolchainStatus:
        """Whether the backend can work here; `ppy doctor` prints the detail."""
        return ToolchainStatus(True)

    def __repr__(self) -> str:
        return f"<backend {self.name}>"
