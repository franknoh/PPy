"""Which backends there are, and loading one when it is asked for.

The builtin backends are always here. An external one is a distribution
with an entry point in the `ppy.backends` group; discovery reads the entry
points without importing anything, and the package is imported only when
its backend is selected -- by `--backend`, by a format it registers, or by
`ppy doctor`. One broken package does not break the others: its failure
is a reported problem, or the error of the command that asked for it. A
name registered twice is never settled by luck: it is reported, and asking
for it is an error naming both distributions.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass, field

from .base import (
    BACKEND_API_VERSION,
    Backend,
    BackendConfig,
    BackendError,
    EmitFormat,
)
from .builtin import BUILTIN_BACKENDS, builtin_backend

__all__ = [
    "ENTRY_POINT_GROUP",
    "BackendCatalog",
    "BackendInfo",
    "BackendLoadError",
    "FormatOwner",
    "available_backends",
    "discover_external_backends",
    "emit_format_owner",
    "load_backend",
]

ENTRY_POINT_GROUP = "ppy.backends"


class BackendLoadError(BackendError):
    """A backend that cannot be used: unknown, duplicated, incompatible, or
    failing to load; the message says which and names what is available."""


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """One backend as discovery sees it, loaded or not."""

    name: str
    builtin: bool
    #: `builtin`, or the distribution and the object the entry point names.
    origin: str
    #: The `importlib.metadata.EntryPoint`; `object` for the compiler's own checker.
    entry: object | None = None


@dataclass(slots=True)
class BackendCatalog:
    """Every backend by name, and what discovery had to refuse."""

    backends: dict[str, BackendInfo] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    #: Names registered more than once, with every claimant; asking for one is an error.
    duplicates: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.backends))


def _origin(entry: object) -> str:
    dist = getattr(entry, "dist", None)
    where = f"{dist.name} {dist.version}" if dist is not None else "an unnamed distribution"
    return f"{where} ({getattr(entry, 'value', '?')})"


def discover_external_backends() -> tuple[dict[str, object], list[str]]:
    """Every installed external backend by name, without importing any, and
    the names that were registered more than once (each is a problem)."""
    # `object` for the compiler's own checker, which has no model of an entry point.
    found: dict[str, object] = {}
    claimants: dict[str, list[str]] = {}
    try:
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as error:  # noqa: BLE001 - a broken distribution must not break discovery
        return found, [f"backend discovery failed: {error}"]
    for entry in entries:
        claimants.setdefault(entry.name, []).append(_origin(entry))
        found.setdefault(entry.name, entry)
    problems = [
        f"backend {name!r} is registered by more than one distribution: {', '.join(where)}"
        for name, where in sorted(claimants.items())
        if len(where) > 1
    ]
    return found, problems


def available_backends() -> BackendCatalog:
    """The builtin backends and every discovered external one, nothing imported."""
    catalog = BackendCatalog()
    for name in BUILTIN_BACKENDS:
        catalog.backends[name] = BackendInfo(name, True, "builtin")
    external, problems = discover_external_backends()
    catalog.problems.extend(problems)
    claimants: dict[str, list[str]] = {}
    try:
        for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            claimants.setdefault(entry.name, []).append(_origin(entry))
    except Exception:  # noqa: BLE001 - already reported by discovery
        claimants = {}
    for name, entry in sorted(external.items()):
        if name in BUILTIN_BACKENDS:
            catalog.problems.append(
                f"backend {name!r} from {_origin(entry)} has a builtin backend's name; "
                "the builtin is used"
            )
            continue
        where = tuple(claimants.get(name, ()))
        if len(where) > 1:
            catalog.duplicates[name] = where
        catalog.backends[name] = BackendInfo(name, False, _origin(entry), entry)
    return catalog


def load_backend(name: str, options: Mapping[str, object] | BackendConfig | None = None) -> Backend:
    """The backend called `name`, constructed with its options: a builtin one
    directly, an external one through its entry point, imported now.

    Refused with a `BackendLoadError` that says why: no such backend (and
    which there are), registered twice, a factory that fails or answers
    something that is not a `Backend`, or an interface version other than
    `BACKEND_API_VERSION`.
    """
    settings = options.options if isinstance(options, BackendConfig) else dict(options or {})
    if name in BUILTIN_BACKENDS:
        return builtin_backend(name, settings)
    catalog = available_backends()
    info = catalog.backends.get(name)
    if info is None or info.entry is None:
        raise BackendLoadError(
            f"no backend is called {name!r}; the backends are {', '.join(catalog.names())}"
        )
    if name in catalog.duplicates:
        raise BackendLoadError(
            f"backend {name!r} is registered by more than one distribution "
            f"({', '.join(catalog.duplicates[name])}); uninstall all but one"
        )
    try:
        factory = info.entry.load()  # type: ignore[attr-defined]
    except Exception as error:
        raise BackendLoadError(
            f"backend {name!r} could not be imported from {info.origin}: {error}"
        ) from error
    try:
        backend = factory(dict(settings))
    except Exception as error:
        raise BackendLoadError(
            f"backend {name!r} could not be created by {info.origin}: {error}"
        ) from error
    if not isinstance(backend, Backend):
        raise BackendLoadError(
            f"backend {name!r} ({info.origin}) answered {type(backend).__name__}, "
            "not a ppy_compiler.backend.Backend"
        )
    version = getattr(backend, "api_version", None)
    if version != BACKEND_API_VERSION:
        raise BackendLoadError(
            f"backend {name!r} is written against backend API version {version!r}; "
            f"this compiler speaks version {BACKEND_API_VERSION}"
        )
    if not backend.name:
        backend.name = name
    elif backend.name != name:
        raise BackendLoadError(
            f"backend {name!r} ({info.origin}) calls itself {backend.name!r}; "
            "the entry point's name and the backend's must agree"
        )
    return backend


@dataclass(frozen=True, slots=True)
class FormatOwner:
    """The backend that emits a format, and the format as it declared it."""

    backend: Backend
    format: EmitFormat


def _builtin_formats() -> dict[str, FormatOwner]:
    owners: dict[str, FormatOwner] = {}
    for name in BUILTIN_BACKENDS:
        backend = builtin_backend(name)
        for spec in backend.emit_formats():
            owners[spec.name] = FormatOwner(backend, spec)
    return owners


def emit_format_owner(kind: str, options_for=None) -> FormatOwner:  # type: ignore[no-untyped-def]
    """The backend that emits `kind`.

    A builtin format is answered without loading anything. For any other
    name every external backend is loaded (each with the options
    `options_for(name)` gives it) and asked; a backend that fails to load
    is reported in the error only when the format is not found elsewhere.
    Two backends claiming one format is an error naming both.
    """
    builtin = _builtin_formats()
    if kind in builtin:
        return builtin[kind]
    catalog = available_backends()
    owners: list[tuple[str, FormatOwner]] = []
    known: list[str] = sorted(builtin)
    failures: list[str] = list(catalog.problems)
    for name, info in sorted(catalog.backends.items()):
        if info.builtin:
            continue
        try:
            backend = load_backend(name, options_for(name) if options_for is not None else None)
        except BackendLoadError as error:
            failures.append(str(error))
            continue
        for spec in backend.emit_formats():
            if spec.name in builtin:
                failures.append(
                    f"backend {name!r} registers the format {spec.name!r}, which is builtin"
                )
                continue
            known.append(spec.name)
            if spec.name == kind:
                owners.append((name, FormatOwner(backend, spec)))
    if len(owners) > 1:
        names = ", ".join(repr(name) for name, _owner in owners)
        raise BackendLoadError(f"the format {kind!r} is registered by backends {names}")
    if owners:
        return owners[0][1]
    detail = f"; also: {'; '.join(failures)}" if failures else ""
    raise BackendLoadError(
        f"no backend emits {kind!r}; the formats are {', '.join(sorted(set(known)))}{detail}"
    )
