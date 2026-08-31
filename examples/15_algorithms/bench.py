"""Time every problem here the way a judge does: the whole process.

Nothing inside the programs is instrumented -- they read input and print an
answer -- so what is measured is wall time from launch to exit, input and
interpreter startup included. Not a CI gate; `scripts/refresh.py` runs this
and says when a number in the documentation has drifted.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
ROUNDS = 5
PPY = [sys.executable, "-m", "ppy_compiler"]


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

PATHS = ["plain", "ppy run", "ppy build", "C scanf"]


def _commands(stem: str) -> list[tuple[str, list[str]]]:
    return [
        ("plain", [sys.executable, f"{stem}.ppy"]),
        ("ppy run", [*PPY, "run", f"{stem}.ppy"]),
        ("ppy build", [f"./dist/{stem}"]),
        ("C scanf", [f"./{stem}_c"]),
    ]


def _wall(command: list[str], folder: Path, data: Path) -> tuple[float, str]:
    """One run, timed from outside: what the judge's clock would show."""
    with data.open("rb") as stream:
        started = time.perf_counter()
        done = subprocess.run(
            command, cwd=folder, stdin=stream, capture_output=True, text=True, check=True
        )
        elapsed = (time.perf_counter() - started) * 1000.0
    return elapsed, done.stdout.strip()


def measure(only: list[str] | None = None, rounds: int = ROUNDS) -> dict:
    """Every problem on every path: milliseconds of wall time, and the answer."""
    results: dict = {}
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
            row: dict = {}
            answers = set()
            for label, command in _commands(stem):
                samples = []
                for _ in range(rounds):
                    elapsed, answer = _wall(command, folder, data)
                    samples.append(elapsed)
                    answers.add(answer)
                row[label] = {
                    "mean": statistics.mean(samples),
                    "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                }
            row["answers"] = sorted(answers)
        results[folder_name] = row
    return results


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    results = measure(only or None)
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1))
        return 0
    print(f"{'problem':<18}" + "".join(f"{name:>18}" for name in PATHS))
    for folder_name, row in results.items():
        cells = "".join(f"{row[name]['mean']:9.1f} ±{row[name]['stdev']:5.1f}" for name in PATHS)
        print(f"{folder_name:<18}{cells}")
        if len(row["answers"]) != 1:
            print(f"  DISAGREE: {row['answers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
