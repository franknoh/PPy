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

__all__ = ["BACKENDS", "available_backends", "run_lint"]


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
                help="install it, or pass --backend with one of "
                + (", ".join(available_backends()) or "none"),
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
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        output = _restore_paths(done.stdout + done.stderr, mapping, Path(scratch))

    if output.strip():
        print(output.rstrip())
    if done.returncode == 0:
        reporter.note(f"{chosen}: no findings in {len(sources)} file(s)")
    return 1 if done.returncode else 0


#: Every file a supported tool reads its settings from. A config left behind
#: means the tool runs in a mode the project never chose, and reports findings
#: the project had already decided about.
_CONFIGS = (
    "pyproject.toml",
    "setup.cfg",
    ".pylintrc",
    "pylintrc",
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    ".mypy.ini",
    "setup.py",
    "pyrightconfig.json",
)


def _stage(sources: list[Path], root: Path, scratch: Path) -> tuple[list[Path], dict[str, Path]]:
    """Mirror the project: each `.ppy` as `.py`, and everything it imports.

    A checker's answer depends on what it can see. Staging only the `.ppy`
    files would leave every `.py` in the project unresolvable, and the missing
    imports would take real findings down with them.
    """
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

    converted = {p.with_suffix(".py") for p in staged}
    for companion in root.rglob("*.py"):
        if any(part in _SKIP_DIRECTORIES for part in companion.parts):
            continue
        target = scratch / companion.resolve().relative_to(root.resolve())
        if target in converted or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(companion, target)

    for extra in _CONFIGS:
        candidate = root / extra
        if candidate.is_file():
            shutil.copyfile(candidate, scratch / extra)
    return staged, mapping


#: Directories whose `.py` files are not the project's own source.
_SKIP_DIRECTORIES = frozenset(
    {".venv", "venv", ".git", "__pycache__", "build", "dist", ".ppy-cache", ".tox", "node_modules"}
)


def _configure(backend: Backend, scratch: Path, *, strict: bool) -> None:
    """Set the wanted mode on top of the project's own settings.

    The project's `extraPaths`, `stubPath`, per-directory execution
    environments and rule overrides are what make its type-check meaningful.
    Overwriting the config with two keys would replace the answer the user
    wanted with an answer about a project that does not exist.
    """
    if backend.name != "pyright":
        return
    import json

    config: dict[str, object] = {}
    existing = scratch / "pyrightconfig.json"
    if existing.is_file():
        try:
            loaded = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            config = loaded
    config["typeCheckingMode"] = "strict" if strict else "standard"
    existing.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


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
            cwd=project.root,
            env=environment,
            check=False,
        )
    return done.returncode
