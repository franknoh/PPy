"""Import support for `.ppy` modules under plain CPython.

Importing `ppy` installs a `sys.meta_path` finder so that sibling `.ppy`
modules load as ordinary Python source -- or natively, when a build of them
exists or can be made (`_native`). No compiler is involved here.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Iterable, Sequence
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import spec_from_file_location

__all__ = [
    "SUFFIX",
    "PPyAmbiguousModuleWarning",
    "PPyPathFinder",
    "PPySourceLoader",
    "add_import_root",
    "import_roots",
    "install",
    "is_installed",
    "uninstall",
]

SUFFIX = ".ppy"


class PPyAmbiguousModuleWarning(ImportWarning):
    """Both `name.py` and `name.ppy` are importable under the same name."""


_roots: list[str] = []


def add_import_root(path: str | os.PathLike[str]) -> None:
    """Confine `.ppy` precedence to the given root (spec 5.2).

    With no root registered, any `sys.path` entry may provide `.ppy` modules.
    """
    resolved = os.path.realpath(os.fspath(path))
    if resolved not in _roots:
        _roots.append(resolved)


def import_roots() -> tuple[str, ...]:
    return tuple(_roots)


def _within_roots(path: str) -> bool:
    if not _roots:
        return True
    real = os.path.realpath(path)
    return any(real == r or real.startswith(r + os.sep) for r in _roots)


class PPySourceLoader(SourceFileLoader):
    """Loads a `.ppy` file as ordinary Python source.

    `SourceFileLoader` already does everything: the suffix only matters to the
    finder, so no method needs overriding.
    """


def _candidates(directory: str, tail: str) -> Iterable[tuple[str, bool]]:
    yield os.path.join(directory, tail + SUFFIX), False
    yield os.path.join(directory, tail, "__init__" + SUFFIX), True


class PPyPathFinder:
    """Meta-path finder for `.ppy` modules and packages."""

    @classmethod
    def find_spec(
        cls,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> ModuleSpec | None:
        tail = fullname.rpartition(".")[2]
        search = list(path) if path is not None else list(sys.path)
        for entry in search:
            directory = entry or os.getcwd()
            if not os.path.isdir(directory):
                continue
            for filename, is_package in _candidates(directory, tail):
                if not os.path.isfile(filename):
                    continue
                if not _within_roots(filename):
                    continue
                cls._warn_if_ambiguous(fullname, directory, tail, is_package)
                from . import _native

                native = _native.spec_for(fullname, filename)
                if native is not None:
                    if is_package:
                        native.submodule_search_locations = [os.path.dirname(filename)]
                    return native
                loader = PPySourceLoader(fullname, filename)
                return spec_from_file_location(
                    fullname,
                    filename,
                    loader=loader,
                    submodule_search_locations=[os.path.dirname(filename)] if is_package else None,
                )
        return None

    @staticmethod
    def _warn_if_ambiguous(fullname: str, directory: str, tail: str, is_package: bool) -> None:
        shadowed = (
            os.path.join(directory, tail, "__init__.py")
            if is_package
            else os.path.join(directory, tail + ".py")
        )
        if os.path.isfile(shadowed):
            warnings.warn(
                f"module {fullname!r} is provided by both a .py and a .ppy source in "
                f"{directory!r}; the .ppy source takes precedence",
                PPyAmbiguousModuleWarning,
                stacklevel=2,
            )

    @classmethod
    def invalidate_caches(cls) -> None:
        return None


def is_installed() -> bool:
    return any(finder is PPyPathFinder for finder in sys.meta_path)


def install() -> None:
    """Install the `.ppy` finder ahead of the standard path finder."""
    if not is_installed():
        sys.meta_path.insert(0, PPyPathFinder)


def uninstall() -> None:
    sys.meta_path[:] = [f for f in sys.meta_path if f is not PPyPathFinder]
