"""Run each counterpart several times, hold them all to one answer, then tabulate their timings.

Every program prints its answers on the lines that do not start with `# `
and one `# <label>: <ms> ms` line per kernel (the eight-kernel programs print
`label   12.3 ms   -> answer` instead); each program already takes the best
of a few warm calls, so a run here is a fresh process, and the table reports
the mean and standard deviation across processes. The counterparts live in
each example's `compare/` folder and want their own environments -- Numba,
CuPy, and Taichi from one Python, Mojo and Codon from their toolchains --
so the commands are given whole:

    python examples/compare.py 5 \\
        "ppy=ppy run compare/ranges_bench.ppy" \\
        "numba=/path/to/python compare/ranges_numba.py" \\
        "mojo=./ranges_mojo"

Correctness comes before timing. Each program must print the same answers
on every run of its own, and every program must print the answers the
reference does -- the first tool named, or the one `--reference NAME`
names. A program that fails, answers differently from itself, answers
differently from the reference, or prints no timing for a kernel the
reference timed, is an error: the table is not printed and the exit status
is 1. A tool that consistently prints its own wrong answer is still wrong.

The tables in the READMEs of `05_numpy`, `07_parallel`, `09_torch`,
`12_buffers_and_jit`, `13_value_classes`, `14_tuples`, `15_algorithms`,
`21_training_torch`, `35_parallel_range`, `36_autodiff`, `37_aio`,
`38_cuda`, `40_generics`, `41_columnar`, `43_regex`, and `44_tile` came from
this, through `scripts/compare_docs.py`, run from a checkout on a native
filesystem with nothing else running; the site collects them on one page.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field

TIMING = re.compile(r"^# ([^:]+): ([0-9.]+) ms$")
#: The eight-kernel programs' own format: `label   12.3 ms   -> answer`.
KERNEL = re.compile(r"^(\S.*?)\s+([0-9.]+) ms\s+-> (\S+)$")


@dataclass
class Outcome:
    """One process's answers and timings, or the reason it gave none."""

    answers: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    failure: str | None = None


def parse(stdout: str) -> Outcome:
    """The answers and the per-kernel timings a program's output carries."""
    outcome = Outcome()
    for line in stdout.splitlines():
        found = TIMING.match(line)
        kernel = KERNEL.match(line)
        if found:
            outcome.timings[found.group(1)] = float(found.group(2))
        elif kernel:
            outcome.timings[kernel.group(1)] = float(kernel.group(2))
            outcome.answers.append(f"{kernel.group(1)}={kernel.group(3)}")
        elif line.startswith("# ") or not line.strip():
            continue
        else:
            outcome.answers.append(line.strip())
    return outcome


def run(command: str) -> Outcome:
    done = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "no output"
        return Outcome(failure=f"exit status {done.returncode}: {tail}")
    return parse(done.stdout)


def check(names: list[str], outcomes: dict[str, list[Outcome]], reference: str) -> list[str]:
    """Every way the outcomes fall short of one agreed answer; empty when they agree."""
    problems: list[str] = []
    for name in names:
        for index, outcome in enumerate(outcomes[name]):
            if outcome.failure is not None:
                problems.append(f"{name}: run {index + 1} failed: {outcome.failure}")
        runs = [outcome for outcome in outcomes[name] if outcome.failure is None]
        if not runs:
            continue
        first = runs[0]
        for index, outcome in enumerate(runs[1:], start=2):
            if outcome.answers != first.answers:
                problems.append(
                    f"{name}: not deterministic: run 1 printed {first.answers}, "
                    f"run {index} printed {outcome.answers}"
                )
        if not first.answers:
            problems.append(f"{name}: printed no answers")
    if any(outcome.failure is not None for outcome in outcomes[reference]):
        return problems
    expected = outcomes[reference][0]
    for name in names:
        if name == reference:
            continue
        runs = [outcome for outcome in outcomes[name] if outcome.failure is None]
        if not runs:
            continue
        given = runs[0]
        if given.answers != expected.answers:
            problems.append(
                f"{name}: answers differ from {reference}: {given.answers} vs {expected.answers}"
            )
        missing = sorted(set(expected.timings) - set(given.timings))
        if missing:
            problems.append(f"{name}: no timing for {', '.join(missing)}")
    return problems


def table(names: list[str], outcomes: dict[str, list[Outcome]]) -> str:
    labels: list[str] = []
    for name in names:
        for outcome in outcomes[name]:
            for label in outcome.timings:
                if label not in labels:
                    labels.append(label)
    lines = ["| kernel | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    # Milliseconds to two places, or four where a kernel is a few microseconds.
    smallest = min(
        (t for name in names for o in outcomes[name] for t in o.timings.values()), default=1.0
    )
    places = 2 if smallest >= 0.1 else 4
    for label in labels:
        cells = []
        for name in names:
            values = [o.timings[label] for o in outcomes[name] if label in o.timings]
            if not values:
                cells.append("—")
                continue
            mean = statistics.mean(values)
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{mean:.{places}f} ± {spread:.{places}f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="hold the counterparts to one answer, then time them"
    )
    parser.add_argument("runs", type=int, help="how many fresh processes per tool")
    parser.add_argument("tools", nargs="+", metavar="label=command")
    parser.add_argument(
        "--reference",
        default=None,
        help="the tool whose answers are the answers (the first, unless named)",
    )
    options = parser.parse_args(argv)
    if options.runs < 1:
        parser.error("at least one run per tool")
    tools: list[tuple[str, str]] = []
    for argument in options.tools:
        name, separator, command = argument.partition("=")
        if not separator or not name or not command:
            parser.error(f"a tool is spelled label=command, not {argument!r}")
        tools.append((name, command))
    names = [name for name, _ in tools]
    if len(set(names)) != len(names):
        parser.error("each tool needs its own label")
    reference = options.reference or names[0]
    if reference not in names:
        parser.error(f"--reference names a tool that is not listed: {reference!r}")
    outcomes: dict[str, list[Outcome]] = {name: [] for name in names}
    for name, command in tools:
        for _ in range(options.runs):
            outcomes[name].append(run(command))
    problems = check(names, outcomes, reference)
    print("answers:")
    for name in names:
        given = next((o.answers for o in outcomes[name] if o.failure is None), None)
        print(f"  {name:<10} {' '.join(given) if given is not None else '(failed)'}")
    if problems:
        print()
        for problem in problems:
            print(f"error: {problem}")
        print(f"\n{len(problems)} problem(s); no timings are reported until every tool agrees")
        return 1
    print(f"\nevery tool agrees with {reference} on {options.runs} run(s) each\n")
    print(table(names, outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
