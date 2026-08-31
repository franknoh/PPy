"""Time every problem here the way a judge would: input included.

Each problem is fed the same input on standard input and run on all four
paths, and both halves are reported -- reading the input and solving it --
because for a submission the two are one wall clock. Not a CI gate.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROUNDS = 5
PPY = [sys.executable, "-m", "ppy_compiler"]
TIMES = re.compile(r"read ([0-9.]+) ms\s+solve ([0-9.]+) ms")
SOLVE_ONLY = re.compile(r"solve ([0-9.]+) ms")


def _queens() -> str:
    return "12\n"


def _dijkstra() -> str:
    count, degree = 200000, 6
    out = [f"{count} {count * degree}", "0"]
    out.extend(
        f"{i} {(i * 7919 + k * 104729 + 1) % count} {(i * 31 + k * 17) % 1000 + 1}"
        for i in range(count)
        for k in range(degree)
    )
    return "\n".join(out) + "\n"


def _kmp() -> str:
    text = "".join(chr(97 + (i * 7919) % 4) for i in range(4000000))
    return f"{text}\n{text[:12]}\n"


def _segment_tree() -> str:
    size, rounds = 1 << 18, 200000
    out = [f"{size} {rounds} {rounds}"]
    out.append(" ".join(str((i * 7919) % 1000) for i in range(size)))
    for step in range(rounds):
        left = (step * 104729) % size
        right = min(left + (step % 512) + 1, size)
        out.append(f"1 {(step * 7919) % size} {(step * 31) % 1000}")
        out.append(f"2 {left} {right}")
    return "\n".join(out) + "\n"


def _lis() -> str:
    count = 1000000
    values = " ".join(str((i * 7919 + (i % 977) * 104729) % 1000003) for i in range(count))
    return f"{count}\n{values}\n"


def _inversions() -> str:
    count = 500000
    return f"{count}\n" + " ".join(str((i * 7919) % 1000003) for i in range(count)) + "\n"


PROBLEMS = [
    ("15a_nqueens", "nqueens", _queens),
    ("15b_dijkstra", "dijkstra", _dijkstra),
    ("15c_kmp", "kmp", _kmp),
    ("15d_segment_tree", "segment_tree", _segment_tree),
    ("15e_lis", "lis", _lis),
    ("15f_input", "inversions", _inversions),
]


def _paths(stem: str) -> list[tuple[str, list[str]]]:
    return [
        ("plain", [sys.executable, f"{stem}.ppy"]),
        ("ppy run", [*PPY, "run", f"{stem}.ppy"]),
        ("ppy build", [f"./dist/{stem}"]),
        ("C scanf", [f"./{stem}_c"]),
    ]


def _measure(command: list[str], folder: Path, data: Path) -> tuple[float, float]:
    with data.open("rb") as stream:
        done = subprocess.run(
            command, cwd=folder, stdin=stream, capture_output=True, text=True, check=True
        )
    found = TIMES.search(done.stdout)
    if found:
        return float(found.group(1)), float(found.group(2))
    only = SOLVE_ONLY.search(done.stdout)
    if only:
        return 0.0, float(only.group(1))
    raise SystemExit(f"no timing from {command}: {done.stdout[:300]!r}")


def main() -> int:
    only = sys.argv[1:]
    print(f"{'problem':<18}{'phase':<8}" + "".join(f"{name:>18}" for name, _ in _paths("x")))
    for folder_name, stem, generate in PROBLEMS:
        if only and not any(token in folder_name for token in only):
            continue
        folder = HERE / folder_name
        if not (folder / "dist" / stem).is_file():
            subprocess.run(
                [*PPY, "build", f"{stem}.ppy", "-o", "dist"],
                cwd=folder,
                capture_output=True,
                text=True,
                check=True,
            )
        with tempfile.TemporaryDirectory() as scratch:
            data = Path(scratch) / "input.txt"
            data.write_text(generate(), encoding="utf-8")
            reads: list[str] = []
            solves: list[str] = []
            for _name, command in _paths(stem):
                samples = [_measure(command, folder, data) for _ in range(ROUNDS)]
                read = [s[0] for s in samples]
                solve = [s[1] for s in samples]
                reads.append(f"{statistics.mean(read):8.1f} ±{statistics.stdev(read):5.1f}")
                solves.append(f"{statistics.mean(solve):8.1f} ±{statistics.stdev(solve):5.1f}")
        print(f"{folder_name:<18}{'read':<8}" + "".join(f"{cell:>18}" for cell in reads))
        print(f"{'':<18}{'solve':<8}" + "".join(f"{cell:>18}" for cell in solves), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
