"""`ppy lint`: run an installed type checker or linter over PPY sources.

External tools key off the `.py` extension, so the sources are mirrored into a
staging tree, the tool runs there, and the paths in its output are mapped back
to the `.ppy` files the user actually has.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..diagnostics import Diagnostic, Severity
from .pipeline import collect_sources, open_project
from .reporting import Reporter

__all__ = ["run_lint", "BACKENDS", "available_backends"]


@dataclass(frozen=True, slots=True)
class Backend:
    name: str
    module: str
    #: Arguments that put the tool in its strictest useful mode.
    strict: tuple[str, ...]
    relaxed: tuple[str, ...] = ()


BACKENDS: dict[str, Backend] = {
    "pyright": Backend("pyright", "pyright", (), ()),
    "pylint": Backend("pylint", "pylint", ("--enable=all",), ()),
    # A type checker's strict mode is a coherent setting; a linter's "every
    # rule" is not the same thing. Ruff runs the project's own configuration,
    # and `--select ALL` is available separately as `--all-rules`.
    "ruff": Backend("ruff", "ruff", ("check",), ("check",)),
    "mypy": Backend("mypy", "mypy", ("--strict",), ()),
}


def available_backends() -> list[str]:
    """The backends whose tool is importable in this environment."""
    import importlib.util

    return [name for name, b in BACKENDS.items() if importlib.util.find_spec(b.module)]


def run_lint(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.path
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2

    chosen = options.backend
    if chosen == "auto":
        found = available_backends()
        if not found:
            reporter.note(
                "no linter is installed; add one with `uv sync --group dev` "
                f"(supported: {', '.join(sorted(BACKENDS))})"
            )
            return 0
        chosen = found[0]
    backend = BACKENDS[chosen]

    import importlib.util

    if importlib.util.find_spec(backend.module) is None:
        reporter.emit(
            Diagnostic(
                "E1801",
                Severity.ERROR,
                f"`{chosen}` is not installed",
                help=f"install it, or pass --backend with one of {', '.join(available_backends()) or 'none'}",
            )
        )
        return 2

    project = open_project(target)
    sources = [p for p in collect_sources(target, ppy_only=target.is_dir()) if p.suffix == ".ppy"]
    if not sources:
        reporter.note(f"no PPY sources found under {target}")
        return 0

    with tempfile.TemporaryDirectory(prefix="ppy-lint-") as scratch:
        staged, mapping = _stage(sources, project.root, Path(scratch))
        _configure(backend, Path(scratch), strict=options.strict)
        arguments = backend.strict if options.strict else (backend.relaxed or backend.strict)
        if chosen == "ruff" and getattr(options, "all_rules", False):
            arguments = (*arguments, "--select", "ALL")
        done = subprocess.run(
            [sys.executable, "-m", backend.module, *arguments, *[str(p) for p in staged]],
            cwd=scratch, capture_output=True, text=True, check=False, timeout=1800,
        )
        output = _restore_paths(done.stdout + done.stderr, mapping, Path(scratch))

    if output.strip():
        print(output.rstrip())
    if done.returncode == 0:
        reporter.note(f"{chosen}: no findings in {len(sources)} file(s)")
    return 1 if done.returncode else 0


def _stage(sources: list[Path], root: Path, scratch: Path) -> tuple[list[Path], dict[str, Path]]:
    """Copy each `.ppy` to a `.py` of the same relative name."""
    staged: list[Path] = []
    mapping: dict[str, Path] = {}
    for source in sources:
        try:
            relative = source.resolve().relative_to(root.resolve())
        except ValueError:
            relative = Path(source.name)
        target = scratch / relative.with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        staged.append(target)
        mapping[str(target)] = source
        mapping[str(relative.with_suffix(".py"))] = source
    for extra in ("pyproject.toml", "setup.cfg", ".pylintrc", "ruff.toml"):
        candidate = root / extra
        if candidate.is_file():
            shutil.copyfile(candidate, scratch / extra)
    return staged, mapping


def _configure(backend: Backend, scratch: Path, *, strict: bool) -> None:
    """Write whatever config file the tool needs to run in the wanted mode."""
    if backend.name != "pyright":
        return
    mode = "strict" if strict else "standard"
    (scratch / "pyrightconfig.json").write_text(
        '{"typeCheckingMode": "%s", "reportMissingImports": false}\n' % mode,
        encoding="utf-8",
    )


def _restore_paths(output: str, mapping: dict[str, Path], scratch: Path) -> str:
    """Point every reported path back at the `.ppy` the user wrote."""
    for staged, original in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        output = output.replace(staged, str(original))
    return re.sub(rf"{re.escape(str(scratch))}[/\\]?", "", output)


def run_pytest(options: argparse.Namespace, reporter: Reporter) -> int:
    """Run a pytest suite with the `.ppy` import hook already installed.

    A test suite is ordinary `.py`, so it is not staged; what it needs is for
    `import ppy` to have happened before it imports the `.ppy` modules under
    test, which the injected plugin arranges.
    """
    import importlib.util

    if importlib.util.find_spec("pytest") is None:
        reporter.emit(
            Diagnostic("E1801", Severity.ERROR, "pytest is not installed", help="`uv sync`")
        )
        return 2

    target: Path = options.path
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2

    project = open_project(target)
    with tempfile.TemporaryDirectory(prefix="ppy-pytest-") as scratch:
        plugin = Path(scratch) / "ppy_pytest_hook.py"
        plugin.write_text(
            "import ppy\n\nppy.install()\n"
            + "".join(f"ppy.add_import_root({str(p)!r})\n" for p in project.search_paths),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [scratch, *(str(p) for p in project.search_paths), environment.get("PYTHONPATH", "")]
        )
        passed = [a for a in getattr(options, "args", []) if a != "--"]
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "ppy_pytest_hook", str(target), *passed],
            cwd=project.root, env=environment, check=False,
        )
    return done.returncode
