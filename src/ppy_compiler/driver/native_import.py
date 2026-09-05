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

`warm` is the same build done ahead of time, by `ppy build --warm`: a
program launched many times at once -- every rank of a `torchrun` -- then
finds the artifact instead of each rank building it.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Outcome", "build_for_import", "manifest_for", "warm"]

#: An import takes no flags: the key is the project configuration alone, so
#: `import ppy`, `ppy run FILE`, and `ppy build --warm` all name one artifact.
_OPTIONS = argparse.Namespace(
    safeguards=None, unsafe=False, safe=False, opt_level=None, no_strict=False, prover=None
)


@dataclass(frozen=True, slots=True)
class Outcome:
    """The manifest to serve, or why not -- and whether to stop asking."""

    manifest: Path | None = None
    #: Set when no `.ppy` in this process will be served natively: the LLVM
    #: backend is missing, or the project turned native imports off.
    off: str | None = None
    #: What happened: `found`, `built`, `jit` (the module needs the in-process
    #: path), `errors` (it does not check clean), or `off`.
    state: str = ""
    errors: int = 0


def manifest_for(path: Path, note: Callable[[str], None]) -> Outcome:
    """The native build of the module at `path`, found or made now."""
    from .reporting import Reporter

    return build_for_import(path, Reporter(quiet=True), note)


def build_for_import(path: Path, reporter, note: Callable[[str], None]) -> Outcome:  # type: ignore[no-untyped-def]
    """Find or make the artifact that serves `path`, reporting check findings through `reporter`."""
    from ..backend.llvm import available, compile_for_run
    from .pipeline import analyze_paths, open_project
    from .warm import locate

    if not available():
        return Outcome(off="llvmlite is not installed; .ppy modules load as Python", state="off")
    located = locate(path, _OPTIONS)
    if not located.config.native_import:
        return Outcome(
            off="native-import = false in [tool.ppy]; .ppy modules load as Python", state="off"
        )
    if located.manifest is not None:
        return Outcome(manifest=located.manifest, state="found")
    if located.needs_jit:
        note(f"{path.name}: needs the in-process JIT; serving Python")
        return Outcome(state="jit")
    project = open_project(path)
    bundle = analyze_paths(project, [path], backend="llvm")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        note(f"{path.name}: {errors} check error(s) above; serving Python")
        return Outcome(state="errors", errors=errors)
    started = time.perf_counter()
    manifest = compile_for_run(bundle, reporter, located.directory, path.resolve())
    if manifest is None:
        note(f"{path.name}: needs the in-process JIT; serving Python")
        return Outcome(state="jit")
    note(f"built {path.name} natively in {time.perf_counter() - started:.1f}s")
    return Outcome(manifest=manifest, state="built")


def warm(target: Path, reporter) -> int:  # type: ignore[no-untyped-def]
    """`ppy build --warm TARGET`: build ahead what `import ppy` and `ppy run` build on first use.

    Every `.ppy` under a directory target is built into the project cache
    under the key an import of it will look for. A module that does not
    check clean is an error here -- the place to learn it is before the
    launch, not in a note on every rank's stderr.
    """
    from ..diagnostics import Diagnostic, Severity
    from .pipeline import collect_sources

    files = collect_sources(target, ppy_only=True)
    if not files:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"no .ppy modules under {target}"))
        return 2
    tally: Counter[str] = Counter()
    for path in files:
        shown = os.path.relpath(path)
        outcome = build_for_import(path, reporter, reporter.note)
        if outcome.off is not None:
            reporter.emit(Diagnostic("E1801", Severity.ERROR, outcome.off))
            return 2
        tally[outcome.state] += 1
        if outcome.state == "found":
            reporter.note(f"{shown}: already built")
    reporter.note(
        f"warm: {tally['built']} built, {tally['found']} already built, "
        f"{tally['jit']} need the in-process JIT, {tally['errors']} with check errors"
    )
    return 1 if tally["errors"] else 0
