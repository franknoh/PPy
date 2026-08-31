"""How fast is reading input, on each path?

Three ways of reading the same numbers, run on plain CPython and on
`ppy run`. Reading is an IO effect, so it stays on the interpreter and the
two paths should agree -- that agreement is the point being measured.
Not a CI gate; run it when you want the numbers on your own machine.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
COUNT = 500000
ROUNDS = 5

READERS = {
    "input()": """
import sys
import time

start = time.perf_counter()
count = int(input())
total = 0
for _ in range(count):
    total += int(input())
print(f"{(time.perf_counter() - start) * 1000.0:.1f} {total}")
""",
    "sys.stdin.readline()": """
import sys
import time

start = time.perf_counter()
stream = sys.stdin
count = int(stream.readline())
total = 0
for _ in range(count):
    total += int(stream.readline())
print(f"{(time.perf_counter() - start) * 1000.0:.1f} {total}")
""",
    "sys.stdin.read().split()": """
import sys
import time

start = time.perf_counter()
fields = sys.stdin.read().split()
count = int(fields[0])
total = 0
for i in range(1, count + 1):
    total += int(fields[i])
print(f"{(time.perf_counter() - start) * 1000.0:.1f} {total}")
""",
}

PATHS = [("plain", []), ("ppy run", ["-m", "ppy_compiler", "run"])]


def _numbers(path: Path) -> None:
    lines = [str(COUNT)]
    lines.extend(str((i * 7919) % 1000003) for i in range(COUNT))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _time(program: Path, args: list[str], data: Path) -> float:
    with data.open("rb") as stream:
        done = subprocess.run(
            [sys.executable, *args, program.name],
            cwd=program.parent,
            stdin=stream,
            capture_output=True,
            text=True,
            check=True,
        )
    return float(done.stdout.split()[-2])


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        room = Path(scratch)
        (room / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
        data = room / "numbers.txt"
        _numbers(data)
        print(f"reading {COUNT} integers, {ROUNDS} runs each\n")
        print(f"{'how':<26}" + "".join(f"{label:>18}" for label, _ in PATHS))
        for name, source in READERS.items():
            program = room / "reader.ppy"
            program.write_text(source.lstrip(), encoding="utf-8")
            cells = []
            for _label, args in PATHS:
                samples = [_time(program, args, data) for _ in range(ROUNDS)]
                cells.append(f"{statistics.mean(samples):9.1f} ± {statistics.stdev(samples):4.1f}")
            print(f"{name:<26}" + "".join(f"{cell:>18}" for cell in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
