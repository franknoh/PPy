"""Differential fuzzing: generated programs, every path, one answer.

    uv run python scripts/fuzz.py --seed 0 --count 25
    uv run python scripts/fuzz.py --seed 400 --count 10 --paths run,standalone
    uv run python scripts/fuzz.py --state --count 25 # module globals and objects
    uv run python scripts/fuzz.py --seed 0 --count 25 --stdlib   # the standard library
    uv run python scripts/fuzz.py --seed 0 --count 25 --calls    # keywords and defaults
    uv run python scripts/fuzz.py --unannotated --count 25   # inferred parameter types
    uv run python scripts/fuzz.py --unannotated --inference --count 25  # decorators, operators
    uv run python scripts/fuzz.py --unannotated --decorators --count 25  # decorators that act
    uv run python scripts/fuzz.py --replay           # every saved regression
    uv run python scripts/fuzz.py --prints --seed 0 --count 25 --paths python,run
    uv run python scripts/fuzz.py --boundary --count 25  # writes through shared containers
    uv run python scripts/fuzz.py --structures --count 25  # linked structures edited in place
    uv run python scripts/fuzz.py --resident --count 25  # Python writes between native calls
    uv run python scripts/fuzz.py --shapes --count 25  # shapes the corpus kept in Python
    uv run python scripts/fuzz.py --optional --count 25  # numbers and strings that may be None

Each seed is a program from `ppy_compiler.testing.fuzz.generate_program`. It
runs under CPython (the reference) and each path asked for, one at a time,
and any path that differs is minimized and saved under
`tests/fuzz_regressions/` with the path it failed on in its first line.
`tests/test_fuzz.py` replays every file there. With `--prints`, functions
also print between checks that may fall back, and a path that prints a
line more often than CPython does is a failure of its own. With `--state`,
each program also reads and writes module globals and walks objects Python
made, and runs on the paths with a Python boundary (CPython, `ppy`, and
`ppy run`). With `--boundary`, a function Python calls natively writes
through containers and objects Python made, shared and nested, which the
generated wrapper copies in and back; those run on the same paths.
With `--structures`, classes whose fields have no annotations (a search
tree with parent links, a doubly linked list) are relinked in place by
methods Python calls natively, on the same paths and without strict mode.
With `--resident`, the same structures are edited by native methods called
again and again, whose objects stay resident between the calls, and written
to from Python between them: attributes set, deleted, and given another
type, links cut, nodes linked in, structures made and dropped.
With `--shapes`, each program also has the shapes the corpus kept in
Python (returns on every side, list parameters, string constants, tuple
assignments, `*args`), on every path; with `--state` too, a function that
falls off its end and a nested function handed its cells.
With `--optional`, each program also holds `int | None`, `float | None`,
`bool | None`, and `str | None` in parameters, results, fields, list and
dict elements, and locals, prints them, and meets `None` in arithmetic
through a field a call reset after it was tested, on every path.
With `--unannotated`, the functions have no annotations, run
without strict mode, and are called from Python with arguments of other
types than the ones their types were inferred from, on the paths with a
Python boundary. `--inference` adds what inference reads beside plain calls:
a `functools.wraps` decorator, operators on a value class, a parameter
declared `list`, a function called with an `int` and a `float`, an
`argparse` option, and functions typed only by how their body uses a
parameter. `--decorators` adds project decorators that change what a call
does (scale the result, print, count, cache, swap the arguments, hand back
another function), called by name from Python and from native loops.

Run it through the shared memory cap in a batch at a time; each program's
paths run one after another, each under its own timeout and memory cap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ppy_compiler.testing.fuzz import (
    ALL_PATHS,
    STATE_PATHS,
    compare,
    generate_program,
    minimize,
    printed_twice,
    run_program,
)

REGRESSIONS = Path(__file__).resolve().parent.parent / "tests" / "fuzz_regressions"


def _paths(text: str) -> tuple[str, ...]:
    wanted = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [p for p in wanted if p not in ALL_PATHS]
    if unknown:
        raise SystemExit(f"unknown path(s): {', '.join(unknown)}; choose from {ALL_PATHS}")
    return ("python", *[p for p in wanted if p != "python"])


def _still_fails(path: str, reason: str):  # type: ignore[no-untyped-def]
    def check(source: str) -> bool:
        # A hang is found again well inside the fuzzing timeout.
        results = run_program(source, ("python", path), timeout=30.0)
        reference = results["python"]
        if "NameError" in reference.last_error or "SyntaxError" in reference.last_error:
            return False
        found = printed_twice(results) + compare(results)
        return any(m.path == path and m.reason == reason for m in found)

    return check


def _save(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    seed: int,
    path: str,
    reason: str,
    source: str,
    state: bool = False,
    boundary: bool = False,
    structures: bool = False,
    shapes: bool = False,
    bools: bool = False,
    optional: bool = False,
    resident: bool = False,
) -> Path:
    REGRESSIONS.mkdir(parents=True, exist_ok=True)
    domain = "_state" if state else "_boundary" if boundary else "_structures" if structures else ""
    domain = "_resident" if resident else domain
    domain += "_shapes" if shapes else ""
    domain += "_bools" if bools else ""
    domain += "_optional" if optional else ""
    target = REGRESSIONS / f"seed{seed}{domain}_{path}.ppy"
    target.write_text(f"# fuzz: path={path} seed={seed} ({reason})\n{source}", encoding="utf-8")
    return target


def fuzz(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    seed: int,
    count: int,
    paths: tuple[str, ...],
    shrink: bool,
    prints: bool = False,
    state: bool = False,
    stdlib: bool = False,
    calls: bool = False,
    unannotated: bool = False,
    boundary: bool = False,
    structures: bool = False,
    shapes: bool = False,
    inference: bool = False,
    bools: bool = False,
    decorators: bool = False,
    optional: bool = False,
    resident: bool = False,
) -> int:
    failures = 0
    started = time.monotonic()
    for current in range(seed, seed + count):
        source = generate_program(
            current,
            prints,
            state,
            stdlib,
            calls,
            unannotated,
            boundary=boundary,
            structures=structures,
            shapes=shapes,
            inference=inference,
            bools=bools,
            decorators=decorators,
            optional=optional,
            resident=resident,
        )
        results = run_program(source, paths, timeout=60.0)
        mismatches = printed_twice(results) + compare(results)
        if not mismatches:
            print(f"ok    seed {current}", flush=True)
            continue
        failures += 1
        for mismatch in mismatches:
            print(
                f"FAIL  seed {current} [{mismatch.path}] {mismatch.reason}: "
                f"expected {mismatch.expected.status} {mismatch.expected.last_error!r}, "
                f"got {mismatch.found.status} {mismatch.found.last_error!r}",
                flush=True,
            )
        first = mismatches[0]
        reduced = source
        if shrink and first.reason != "did not build":
            reduced = minimize(source, _still_fails(first.path, first.reason), attempts=60)
        saved = _save(
            current,
            first.path,
            first.reason,
            reduced,
            state,
            boundary,
            structures,
            shapes,
            bools,
            optional,
            resident,
        )
        print(f"      saved {saved.relative_to(REGRESSIONS.parent.parent)}", flush=True)
    elapsed = time.monotonic() - started
    print(f"{count - failures}/{count} programs agree on {', '.join(paths)} ({elapsed:.0f}s)")
    return 1 if failures else 0


def replay() -> int:
    failures = 0
    for file in sorted(REGRESSIONS.glob("*.ppy")):
        header, _, source = file.read_text(encoding="utf-8").partition("\n")
        path = header.split("path=", 1)[1].split()[0]
        mismatches = compare(run_program(source, ("python", path)))
        status = "FAIL" if mismatches else "ok  "
        failures += bool(mismatches)
        print(f"{status}  {file.name}", flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--paths", default=None)
    parser.add_argument(
        "--state", action="store_true", help="add module globals and objects (paths with Python)"
    )
    parser.add_argument(
        "--boundary",
        action="store_true",
        help="write through shared containers and objects Python passes (paths with Python)",
    )
    parser.add_argument(
        "--structures",
        action="store_true",
        help="edit linked structures with unannotated fields in place (paths with Python)",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="call native methods on the same structures again and again, and write to them "
        "from Python between the calls (paths with Python)",
    )
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="add the shapes the corpus kept in Python: returns on every side, list "
        "parameters, string constants, tuple assignments, *args (with --state, nested cells)",
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="hold int | None, float | None, bool | None, and str | None in parameters, "
        "results, fields, and elements, and meet None in arithmetic",
    )
    parser.add_argument("--no-minimize", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--prints", action="store_true", help="functions print between checks")
    parser.add_argument("--show", type=int, help="print the program for this seed and exit")
    parser.add_argument(
        "--calls",
        action="store_true",
        help="functions take defaults and keyword-only parameters; calls name and omit them",
    )
    parser.add_argument(
        "--stdlib",
        action="store_true",
        help=(
            "draw seeded random numbers and call math, heapq, bisect, itertools, functools,"
            " operator, and collections' containers"
        ),
    )
    parser.add_argument(
        "--unannotated",
        action="store_true",
        help="functions without annotations, typed from their calls under --no-strict",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help=(
            "with --unannotated: a wraps decorator, operators on a value class, a `list`"
            " parameter, int and float calls, argparse, and parameters typed by their use"
        ),
    )
    parser.add_argument(
        "--bools",
        action="store_true",
        help=(
            "store bools where an int is declared (locals, lists, dicts, tuples, fields,"
            " returns, Callable arguments) and print them (paths with Python)"
        ),
    )
    parser.add_argument(
        "--decorators",
        action="store_true",
        help=(
            "with --unannotated: project decorators that scale results, print, count, cache,"
            " swap arguments, or replace the function, called from native loops too"
        ),
    )
    options = parser.parse_args(argv)
    if options.show is not None:
        shown = generate_program(
            options.show,
            options.prints,
            options.state,
            options.stdlib,
            options.calls,
            options.unannotated,
            boundary=options.boundary,
            structures=options.structures,
            shapes=options.shapes,
            inference=options.inference,
            bools=options.bools,
            decorators=options.decorators,
            optional=options.optional,
            resident=options.resident,
        )
        print(shown, end="")
        return 0
    if options.replay:
        return replay()
    # A program with module state, one Python calls by name, or one that
    # writes through what Python passes runs where there is a Python boundary.
    python_only = (
        options.state
        or options.unannotated
        or options.boundary
        or options.structures
        or options.resident
        or options.bools
    )
    paths = options.paths or ",".join(STATE_PATHS if python_only else ALL_PATHS)
    return fuzz(
        options.seed,
        options.count,
        _paths(paths),
        not options.no_minimize,
        options.prints,
        options.state,
        options.stdlib,
        options.calls,
        options.unannotated,
        options.boundary,
        options.structures,
        options.shapes,
        options.inference,
        options.bools,
        options.decorators,
        options.optional,
        options.resident,
    )


if __name__ == "__main__":
    sys.exit(main())
