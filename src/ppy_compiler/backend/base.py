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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..cache.keys import digest
from ..driver.config import BackendConfig

if TYPE_CHECKING:
    from ..ir import DialectRegistry, IRModule, Pass, PassManager, SourceLocation

__all__ = [
    "BACKEND_API_VERSION",
    "UNDECLARED_API_VERSION",
    "Backend",
    "BackendConfig",
    "BackendContext",
    "BackendError",
    "BackendPassRegistrar",
    "BackendUnavailable",
    "BackendValidationError",
    "BuildResult",
    "EmitFormat",
    "ToolchainStatus",
]

#: The interface a backend is written against. Bumped when a method's
#: meaning or signature changes; a backend declaring another version is
#: refused at load time.
#:
#: An external backend declares the version it implements as a literal of
#: its own -- `api_version = 1` -- and never as this constant: a package
#: that spells the constant is rewritten by every compiler upgrade it is
#: imported into, which is exactly the compatibility break the number is
#: there to catch.
BACKEND_API_VERSION = 1

#: `Backend.api_version` unless a subclass declares one. An external backend
#: that leaves it is refused rather than assumed current.
UNDECLARED_API_VERSION: int | None = None


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
    #: `module`: one artifact per module, from `emit`; a target resolving to
    #: several modules needs `-o DIR`, since two artifacts are not one file.
    #: `program`: one artifact for the whole program, from `emit_program`,
    #: which is handed every module at once.
    scope: str = "module"

    def __post_init__(self) -> None:
        if self.scope not in ("module", "program"):
            raise ValueError(
                f"the emit format {self.name!r} has scope {self.scope!r}; "
                "a format is either 'module' or 'program' scoped"
            )


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


class BackendPassRegistrar:
    """Where a backend hangs its passes: the `backend` stage, and nowhere else.

    The stages before it belong to the shared pipeline and to the plugins --
    a pass of a backend's own running among them would decide for every other
    backend what the canonical IR is. A backend is handed this rather than the
    `PassManager` so that the invalid thing cannot be spelled.
    """

    __slots__ = ("_backend", "_manager")

    #: The one stage a backend may register at.
    STAGE = "backend"

    def __init__(self, manager: PassManager, backend: str) -> None:
        self._manager = manager
        self._backend = backend

    def add(self, factory: Callable[[], Pass]) -> None:
        """Run `factory()`'s pass at the `backend` stage, after every shared pass."""
        self._manager.register_stage_pass(self.STAGE, factory)

    def register_stage_pass(self, stage: str, factory: Callable[[], Pass]) -> None:
        """`add(factory)`, for the one stage a backend owns; any other is refused."""
        if stage != self.STAGE:
            raise BackendError(
                f"backend {self._backend!r} registers a pass at the {stage!r} stage: a "
                f"backend's passes run at the {self.STAGE!r} stage, after the shared "
                f"pipeline and the plugins, and `BackendPassRegistrar.add(factory)` is "
                f"how one is hung there. A transformation that must run earlier belongs "
                f"to a plugin or to the shared pipeline, not to one backend"
            )
        self.add(factory)

    def __repr__(self) -> str:
        return f"<backend passes for {self._backend}>"


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
    #: The interface version this backend implements, which an external
    #: backend declares as a literal of its own::
    #:
    #:     class MyBackend(Backend):
    #:         api_version = 1
    #:
    #: Left undeclared it is refused: a package that inherits whatever the
    #: compiler it is imported into happens to say would be called current
    #: forever, and the number would catch nothing.
    api_version: int | None = UNDECLARED_API_VERSION
    #: The distribution this backend came from, as (name, version), filled in
    #: by the registry; None for a builtin. It is part of every artifact's
    #: identity, so a new release of the package is new artifacts whether or
    #: not its author touched `fingerprint`.
    distribution: tuple[str, str] | None = None

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

    def register_passes(self, passes: BackendPassRegistrar) -> None:
        """Hang the backend's own passes: `passes.add(MyPass)`.

        They run at the `backend` stage, after every shared pass and every
        plugin's, and the module is verified after each of them.
        """

    def validate(self, module: IRModule, context: BackendContext) -> None:
        """Refuse IR the backend cannot take, with a `BackendValidationError`
        naming what and why. Called after every pass, before `emit` or `build`."""

    def emit(  # pylint: disable=redefined-builtin
        self, module: IRModule, format: str, context: BackendContext
    ) -> str | bytes:
        """`format` (one of `emit_formats`) for one module: text, or bytes for
        a binary format."""
        raise BackendError(f"backend {self.name!r} does not emit {format!r}")

    def emit_program(  # pylint: disable=redefined-builtin
        self, modules: Mapping[str, IRModule], format: str, context: BackendContext
    ) -> str | bytes:
        """A `program`-scoped format: every module at once, one artifact back.

        Only a format that declared `scope="program"` arrives here; a
        module-scoped one goes to `emit`, once per module.
        """
        raise BackendError(f"backend {self.name!r} does not emit {format!r} for a whole program")

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
