"""The compiler's half of a native `import`: the build of one `.ppy` module.

`import ppy` serves a `.ppy` module from its native build when one exists or
can be made; deciding that, and making it, is the compiler's business, and
this is where it lives. The runtime asks for this module by name, lazily,
and takes its absence as "no compiler installed".

`manifest_for` is `ppy run` up to the launch: the warm-run directory keyed by
everything that could change the artifact, a check of the module, and
`compile_for_run` into that directory. It never raises for the caller's
sake: what it cannot build it explains through `note`, and says in `off`
when nothing in this process will build.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Outcome", "manifest_for"]


@dataclass(frozen=True, slots=True)
class Outcome:
    """The manifest to serve, or why not -- and whether to stop asking."""

    manifest: Path | None = None
    #: Set when no `.ppy` in this process will be served natively: the LLVM
    #: backend is missing, or the project turned native imports off.
    off: str | None = None


def manifest_for(path: Path, note: Callable[[str], None]) -> Outcome:
    """The native build of the module at `path`, found or made now."""
    from ..backend.llvm import available, compile_for_run
    from .pipeline import analyze_paths, open_project
    from .reporting import Reporter
    from .warm import locate

    if not available():
        return Outcome(off="llvmlite is not installed; .ppy modules load as Python")
    options = argparse.Namespace(
        safeguards=None, unsafe=False, safe=False, opt_level=None, no_strict=False, prover=None
    )
    located = locate(path, options)
    if not located.config.native_import:
        return Outcome(off="native-import = false in [tool.ppy]; .ppy modules load as Python")
    if located.manifest is not None:
        return Outcome(manifest=located.manifest)
    if located.needs_jit:
        note(f"{path.name}: needs the in-process JIT; serving Python")
        return Outcome()
    project = open_project(path)
    bundle = analyze_paths(project, [path], backend="llvm")
    reporter = Reporter(quiet=True)
    errors = reporter.report(bundle.diagnostics)
    if errors:
        note(f"{path.name}: {errors} check error(s) above; serving Python")
        return Outcome()
    started = time.perf_counter()
    manifest = compile_for_run(bundle, reporter, located.directory, path.resolve())
    if manifest is None:
        note(f"{path.name}: needs the in-process JIT; serving Python")
        return Outcome()
    note(f"built {path.name} natively in {time.perf_counter() - started:.1f}s")
    return Outcome(manifest=manifest)
