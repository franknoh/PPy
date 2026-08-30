"""`[tool.ppy]` project configuration (spec 30)."""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = [
    "Config",
    "ConvertConfig",
    "DiagnosticsConfig",
    "FormatConfig",
    "InferenceConfig",
    "LlvmConfig",
    "ParallelConfig",
    "PluginConfig",
    "PythonBackendConfig",
    "find_project_root",
    "load_config",
]

_MARKERS = ("pyproject.toml", "ppy.toml", ".git")


@dataclass(slots=True)
class PythonBackendConfig:
    enabled: bool = True
    interpreter: str = "python"


@dataclass(slots=True)
class LlvmConfig:
    enabled: bool = True
    target: str = "native"
    jit: bool = True
    lto: str = "thin"
    cpython_api: str = "version-specific"
    #: "hoisted" (default) proves loop guards once before the loop and runs a
    #: clean body; "inline" keeps the per-operation guards in the body.
    safeguards: str = "hoisted"


@dataclass(slots=True)
class ParallelConfig:
    enabled: bool = True
    threads: str | int = "auto"


@dataclass(slots=True)
class InferenceConfig:
    interprocedural: bool = True
    write_local_annotations: bool = True
    implicit_any: str = "error"


@dataclass(slots=True)
class PluginConfig:
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConvertConfig:
    #: Whether `ppy convert` hands its result to the project's formatter.
    format: bool = False
    #: `safe` moves only classes whose definition is provably inert;
    #: `aggressive` moves any, `off` moves none.
    hoist_classes: str = "safe"


@dataclass(slots=True)
class FormatConfig:
    #: Which external formatter runs after the built-in normalizer. `auto`
    #: picks the one the project configures; `none` is built-in only.
    backend: str = "auto"


@dataclass(slots=True)
class DiagnosticsConfig:
    optimization_remarks: bool = False


@dataclass(slots=True)
class Config:
    root: Path = field(default_factory=Path.cwd)
    python: str = ">=3.12,<3.15"
    strict: bool = True
    opt_level: int = 2
    cache_dir: str = ".ppy-cache"
    dynamic_boundaries: str = "explicit"
    build_execution: str = "deny"
    source_roots: tuple[str, ...] = ("src", ".")
    python_backend: PythonBackendConfig = field(default_factory=PythonBackendConfig)
    llvm: LlvmConfig = field(default_factory=LlvmConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    plugins: dict[str, PluginConfig] = field(default_factory=dict)
    convert: ConvertConfig = field(default_factory=ConvertConfig)
    format: FormatConfig = field(default_factory=FormatConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    @property
    def cache_path(self) -> Path:
        override = os.environ.get("PPY_CACHE_DIR")
        if override:
            # The slow-filesystem escape hatch: a repo on a Windows-mounted
            # drive under WSL can keep its cache on the fast side. One tree
            # per project root, so two projects never share a store.
            stamp = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:12]
            return Path(override).expanduser() / f"{self.root.name}-{stamp}"
        path = Path(self.cache_dir)
        return path if path.is_absolute() else self.root / path

    def plugin(self, name: str) -> PluginConfig:
        return self.plugins.get(name, PluginConfig())

    def with_overrides(self, **overrides: Any) -> Config:
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean) if clean else self


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start] if start.is_dir() else [start.parent]
    for directory in [*candidates, *candidates[0].parents]:
        for marker in _MARKERS:
            if (directory / marker).exists():
                return directory
    return candidates[0]


def _as_bool(value: Any, default: bool) -> bool:
    return bool(value) if isinstance(value, bool) else default


def load_config(root: Path) -> Config:
    """Read `[tool.ppy]` from `pyproject.toml`, falling back to defaults."""
    config = Config(root=root)
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return config
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return config
    table = data.get("tool", {}).get("ppy")
    if not isinstance(table, Mapping):
        return config
    return _apply(config, table)


def _apply(config: Config, table: Mapping[str, Any]) -> Config:
    config.python = table.get("python", config.python)
    config.strict = _as_bool(table.get("strict"), config.strict)
    level = table.get("opt-level", config.opt_level)
    config.opt_level = level if isinstance(level, int) and 0 <= level <= 3 else config.opt_level
    config.cache_dir = table.get("cache-dir", config.cache_dir)
    config.dynamic_boundaries = table.get("dynamic-boundaries", config.dynamic_boundaries)
    config.build_execution = table.get("build-execution", config.build_execution)
    roots = table.get("source-roots")
    if isinstance(roots, list) and roots:
        config.source_roots = tuple(str(r) for r in roots)

    if isinstance(sub := table.get("python-backend"), Mapping):
        config.python_backend = PythonBackendConfig(
            enabled=_as_bool(sub.get("enabled"), True),
            interpreter=sub.get("interpreter", "python"),
        )
    if isinstance(sub := table.get("llvm"), Mapping):
        config.llvm = LlvmConfig(
            enabled=_as_bool(sub.get("enabled"), True),
            target=sub.get("target", "native"),
            jit=_as_bool(sub.get("jit"), True),
            lto=sub.get("lto", "thin"),
            cpython_api=sub.get("cpython-api", "version-specific"),
            safeguards=sub.get("safeguards", "hoisted"),
        )
    if isinstance(sub := table.get("parallel"), Mapping):
        config.parallel = ParallelConfig(
            enabled=_as_bool(sub.get("enabled"), True),
            threads=sub.get("threads", "auto"),
        )
    if isinstance(sub := table.get("inference"), Mapping):
        config.inference = InferenceConfig(
            interprocedural=_as_bool(sub.get("interprocedural"), True),
            write_local_annotations=_as_bool(sub.get("write-local-annotations"), True),
            implicit_any=sub.get("implicit-any", "error"),
        )
    if isinstance(sub := table.get("convert"), Mapping):
        mode = sub.get("hoist-classes", "safe")
        config.convert = ConvertConfig(
            format=_as_bool(sub.get("format"), False),
            hoist_classes=mode if mode in {"safe", "aggressive", "off"} else "safe",
        )
    if isinstance(sub := table.get("format"), Mapping):
        backend = sub.get("backend", "auto")
        config.format = FormatConfig(
            backend=backend if backend in {"auto", "ruff", "black", "none"} else "auto"
        )
    if isinstance(sub := table.get("diagnostics"), Mapping):
        config.diagnostics = DiagnosticsConfig(
            optimization_remarks=_as_bool(sub.get("optimization-remarks"), False),
        )
    if isinstance(sub := table.get("plugins"), Mapping):
        for name, options in sub.items():
            if isinstance(options, Mapping):
                config.plugins[name] = PluginConfig(
                    enabled=_as_bool(options.get("enabled"), True),
                    options={k: v for k, v in options.items() if k != "enabled"},
                )
    return config
