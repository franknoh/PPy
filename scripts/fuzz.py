"""Differential fuzzing: generated programs, every path, one answer.

    uv run python scripts/fuzz.py --seed 0 --count 25
    uv run python scripts/fuzz.py --seed 400 --count 10 --paths run,standalone
    uv run python scripts/fuzz.py --state --count 25 # module globals and objects
    uv run python scripts/fuzz.py --seed 0 --count 25 --stdlib   # the standard library
    uv run python scripts/fuzz.py --seed 0 --count 25 --calls    # keywords and defaults
    uv run python scripts/fuzz.py --unannotated --count 25   # inferred parameter types
    uv run python scripts/fuzz.py --replay           # every saved regression
    uv run python scripts/fuzz.py --prints --seed 0 --count 25 --paths python,run

Each seed is a program from `ppy_compiler.testing.fuzz.generate_program`. It
runs under CPython (the reference) and each path asked for, one at a time,
and any path that differs is minimized and saved under
`tests/fuzz_regressions/` with the path it failed on in its first line.
`tests/test_fuzz.py` replays every file there. With `--prints`, functions
also print between checks that may fall back, and a path that prints a
line more often than CPython does is a failure of its own. With `--state`,
each program also reads and writes module globals and walks objects Python
made, and runs on the paths with a Python boundary (CPython, `ppy`, and
`ppy run`). With `--unannotated`, the functions have no annotations, run
without strict mode, and are called from Python with arguments of other
types than the ones their types were inferred from, on the paths with a
Python boundary.

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


def _save(seed: int, path: str, reason: str, source: str, state: bool = False) -> Path:
    REGRESSIONS.mkdir(parents=True, exist_ok=True)
    target = REGRESSIONS / f"seed{seed}{'_state' if state else ''}_{path}.ppy"
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
) -> int:
    failures = 0
    started = time.monotonic()
    for current in range(seed, seed + count):
        source = generate_program(current, prints, state, stdlib, calls, unannotated)
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
        saved = _save(current, first.path, first.reason, reduced, state)
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
    options = parser.parse_args(argv)
    if options.show is not None:
        shown = generate_program(
            options.show,
            options.prints,
            options.state,
            options.stdlib,
            options.calls,
            options.unannotated,
        )
        print(shown, end="")
        return 0
    if options.replay:
        return replay()
    # A program with module state, or one Python calls by name, runs where
    # there is a Python boundary.
    python_only = options.state or options.unannotated
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
    )


if __name__ == "__main__":
    sys.exit(main())
