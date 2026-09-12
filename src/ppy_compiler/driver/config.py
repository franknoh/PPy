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


#: The one road through the LLVM backend. `"ast"` -- in the setting or as
#: `PPY_LOWERING=ast` -- named the direct AST lowering 0.2.0 removed; it is
#: still read, and a build that asks for it is told (`W2004`).
PIPELINES = ("ir",)
PIPELINE_ENV = "PPY_LOWERING"
REMOVED_PIPELINE = "ast"


def selected_pipeline(configured: str | None) -> str:
    """The IR road, whatever was asked for: there is no other."""
    del configured
    return "ir"


def asks_for_removed_pipeline(configured: str | None) -> bool:
    """Whether the project or the environment names the direct road that is gone."""
    return configured == REMOVED_PIPELINE or os.environ.get(PIPELINE_ENV) == REMOVED_PIPELINE


@dataclass(slots=True)
class LlvmConfig:
    enabled: bool = True
    target: str = "native"
    jit: bool = True
    lto: str = "thin"
    cpython_api: str = "version-specific"
    #: "hoisted" proves loop guards once before the loop and runs a clean
    #: body; "inline" keeps the per-operation guards in the body; "off" drops
    #: the overflow guards on data arithmetic (64-bit wrap) while keeping
    #: every bounds check. None means "the command decides": `ppy run`
    #: and `ppy build` both default to hoisted; `--unsafe` is off.
    safeguards: str | None = None
    #: "z3" asks the solver to prove overflow guards away where the ranges
    #: and relations the analysis established allow it; None or "off" emits
    #: every guard as always. Needs `ppy-lang[solver]`.
    prover: str | None = None
    #: Compile object code for the CPU that builds it rather than the
    #: portable baseline. Off by default: an artifact is meant to be shipped,
    #: and host code faults on an older machine. In-process JIT code always
    #: targets the host, because it never leaves it.
    host_cpu: bool = False
    #: The road through the backend: the canonical IR and its passes. "ast",
    #: the direct lowering 0.2.0 removed, is still read and answered.
    pipeline: str = "ir"
    #: The sanitizers a build instruments the IR with (`bounds`, `overflow`,
    #: `pointer`, `alignment`); a failed check raises rather than falls back.
    sanitize: tuple[str, ...] = ()
    #: A `.ppyprof` from `ppy run --profile` that guides the build (`--pgo FILE`).
    pgo: str | None = None
    #: Place profile counters in the IR; `ppy run --profile` sets it for its run.
    instrument: bool = False


@dataclass(slots=True)
class ParallelConfig:
    enabled: bool = True
    threads: str | int = "auto"
    #: How a parallel loop is lowered: "threads" splits it across the worker
    #: count, "serial" runs it on the calling thread, "simd" hands the serial
    #: loop to the vectorizer, "openmp" spells it as OpenMP regions in the C
    #: backend's output. Every choice gives the same answer.
    backend: str = "threads"


@dataclass(slots=True)
class InferenceConfig:
    interprocedural: bool = True
    write_local_annotations: bool = True
    implicit_any: str = "error"


@dataclass(slots=True)
class PluginConfig:
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """The `[tool.ppy.backends.<name>]` table, as the backend receives it."""

    name: str
    options: Mapping[str, object] = field(default_factory=dict)

    def get(self, key: str, default: object = None) -> object:
        return self.options.get(key, default)

    def fingerprint(self) -> str:
        """A digest of the options: part of every artifact key, so a changed
        setting is a different artifact."""
        from ..cache.keys import digest

        return digest(self.name, tuple(sorted((k, repr(v)) for k, v in self.options.items())))


@dataclass(slots=True)
class GenericsConfig:
    """How far monomorphization may go before it is refused."""

    #: Distinct type-argument tuples one generic may be called with.
    max_specializations: int = 64
    #: Nesting of one generic's type argument inside its own type parameter.
    max_depth: int = 8


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
    #: Whether `import ppy` may serve a `.ppy` module from its native build
    #: when the compiler is installed. Off, every `.ppy` loads as source.
    native_import: bool = True
    source_roots: tuple[str, ...] = ("src", ".")
    python_backend: PythonBackendConfig = field(default_factory=PythonBackendConfig)
    llvm: LlvmConfig = field(default_factory=LlvmConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    plugins: dict[str, PluginConfig] = field(default_factory=dict)
    #: `[tool.ppy.backends.<name>]`: each backend's own table, handed to that
    #: backend and to no other; a table for a backend that is not installed
    #: is not an error.
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    generics: GenericsConfig = field(default_factory=GenericsConfig)
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

    def backend(self, name: str) -> BackendConfig:
        return self.backends.get(name, BackendConfig(name))

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
    config.native_import = _as_bool(table.get("native-import"), config.native_import)
    config.build_execution = table.get("build-execution", config.build_execution)
    generics = table.get("generics", {})
    if isinstance(generics, dict):
        config.generics = GenericsConfig(
            max_specializations=int(generics.get("max-specializations", 64)),
            max_depth=int(generics.get("max-depth", 8)),
        )
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
            safeguards=sub.get("safeguards"),
            prover=sub.get("prover"),
            pipeline=str(sub.get("pipeline", "ir")),
            host_cpu=_as_bool(sub.get("host-cpu"), False),
            sanitize=tuple(str(kind) for kind in (sub.get("sanitize") or ())),
            pgo=str(sub["pgo"]) if sub.get("pgo") else None,
        )
    if isinstance(sub := table.get("parallel"), Mapping):
        config.parallel = ParallelConfig(
            enabled=_as_bool(sub.get("enabled"), True),
            threads=sub.get("threads", "auto"),
            backend=str(sub.get("backend", "threads")),
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
    if isinstance(sub := table.get("backends"), Mapping):
        for name, options in sub.items():
            if isinstance(options, Mapping):
                config.backends[str(name)] = BackendConfig(str(name), dict(options))
    return config
