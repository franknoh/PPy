from __future__ import annotations

import os
import textwrap
from pathlib import Path

# JAX preallocates most of the accelerator's memory by default, which starves
# any other runtime the suite loads. Tests share one device, so they share it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest  # pylint: disable=wrong-import-position

from ppy_compiler.driver.pipeline import (  # pylint: disable=wrong-import-position
    analyze_paths,
    open_project,
)


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\nopt-level = 2\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def write(project_dir: Path):
    def _write(name: str, source: str) -> Path:
        path = project_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def analyze(project_dir: Path):
    def _analyze(path: Path, *, backend: str = "python", **overrides):
        project = open_project(path, config_overrides=overrides)
        return analyze_paths(project, [path], backend=backend)

    return _analyze


@pytest.fixture
def codes(analyze):
    def _codes(path: Path, **kwargs) -> list[str]:
        bundle = analyze(path, **kwargs)
        return [d.code for d in bundle.diagnostics.sorted()]

    return _codes
