"""Run each counterpart several times, hold them to one answer, and tabulate their timings.

Every program prints its answers on the lines that do not start with `# `
and one `# <label>: <ms> ms` line per kernel (the eight-kernel programs print
`label   12.3 ms   -> answer` instead); each program already takes the best
of a few warm calls, so a run here is a fresh process, and the table reports
the mean and standard deviation across processes. The counterparts live in
each example's `compare/` folder and want their own environments -- Numba,
CuPy, and Taichi from one Python, Mojo and Codon from their toolchains --
so the commands are given whole:

    python examples/compare.py 5 \
        "ppy=ppy run compare/ranges_bench.ppy" \
        "numba=/path/to/python compare/ranges_numba.py" \
        "mojo=./ranges_mojo"

The tables in the READMEs of `15_algorithms`, `35_parallel_range`, and
`38_cuda` came from this, run from a checkout on a native filesystem with
nothing else running.
"""

import re
import statistics
import subprocess
import sys

TIMING = re.compile(r"^# ([^:]+): ([0-9.]+) ms$")
#: The eight-kernel programs' own format: `label   12.3 ms   -> answer`.
KERNEL = re.compile(r"^(\S.*?)\s+([0-9.]+) ms\s+-> (\S+)$")


def run(command: str) -> tuple[list[str], dict[str, float]]:
    done = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{command}\n{done.stderr[-2000:]}")
    answers: list[str] = []
    timings: dict[str, float] = {}
    for line in done.stdout.splitlines():
        found = TIMING.match(line)
        kernel = KERNEL.match(line)
        if found:
            timings[found.group(1)] = float(found.group(2))
        elif kernel:
            timings[kernel.group(1)] = float(kernel.group(2))
            answers.append(f"{kernel.group(1)}={kernel.group(3)}")
        elif line.startswith("# ") or not line.strip():
            continue
        else:
            answers.append(line.strip())
    return answers, timings


def main() -> None:
    runs = int(sys.argv[1])
    tools = [argument.split("=", 1) for argument in sys.argv[2:]]
    answers: dict[str, list[str]] = {}
    samples: dict[str, dict[str, list[float]]] = {}
    for name, command in tools:
        for _ in range(runs):
            given, timings = run(command)
            answers.setdefault(name, given)
            if given != answers[name]:
                print(f"{name}: answers differ between runs: {given} vs {answers[name]}")
            for label, value in timings.items():
                samples.setdefault(label, {}).setdefault(name, []).append(value)
    print("answers:")
    for name, given in answers.items():
        print(f"  {name:<10} {' '.join(given)}")
    names = [name for name, _ in tools]
    print()
    print("| kernel | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for label, by_tool in samples.items():
        cells = []
        for name in names:
            values = by_tool.get(name)
            if not values:
                cells.append("—")
                continue
            mean = statistics.mean(values)
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{mean:.2f} ± {spread:.2f}")
        print(f"| {label} | " + " | ".join(cells) + " |")


main()
