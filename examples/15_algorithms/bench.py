"""Time every problem here the way a judge does: the whole process.

Nothing inside the programs is instrumented -- they read input and print an
answer -- so what is measured is wall time from launch to exit, input and
interpreter startup included. Not a CI gate; `scripts/refresh.py` runs this
and says when a number in the documentation has drifted.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import standalone_toolchain_status, toolchain_status

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

PATHS = ["plain", "ppy run", "ppy build", "standalone", "C scanf"]

#: The CPython-free variant of a problem, where the subset reaches it. It is a
#: separate source -- no `array.array`, no `try`/`except EOFError` -- so it is
#: built and timed as itself rather than credited to the file beside it. It
#: still answers the same input, and the run below holds it to that.
STANDALONE = {
    "15a_nqueens": "nqueens",
    "15b_dijkstra": "dijkstra",
    "15d_segment_tree": "segment_tree",
    "15e_lis": "lis",
    "15f_input": "inversions",
}


def available() -> dict[str, str]:
    """Which paths this machine can actually build, and why not when it cannot.

    Asked once, before anything is timed. Without it a missing toolchain
    surfaces as a `FileNotFoundError` on an executable nobody wrote, which
    says nothing about what is missing.
    """
    missing: dict[str, str] = {}
    if shutil.which("gcc") is None:
        missing["C scanf"] = "gcc is not on PATH"
    if not llvm_available():
        reason = "llvmlite is not installed"
        missing["ppy run"] = missing["ppy build"] = missing["standalone"] = reason
        return missing
    usable, detail = toolchain_status()
    if not usable:
        missing["ppy build"] = detail
    usable, detail = standalone_toolchain_status()
    if not usable:
        missing["standalone"] = detail
    return missing


def _commands(stem: str, standalone: str | None) -> list[tuple[str, list[str]]]:
    rows = [
        ("plain", [sys.executable, f"{stem}.ppy"]),
        ("ppy run", [*PPY, "run", f"{stem}.ppy"]),
        ("ppy build", [f"./dist/{stem}"]),
    ]
    if standalone is not None:
        rows.append(("standalone", [f"./native/{standalone}"]))
    rows.append(("C scanf", [f"./{stem}_c"]))
    return rows


def collapse(answers: dict[str, list[str]]) -> tuple[list[str], dict]:
    """Every answer seen, and each path's -- one value only if it never varied.

    A path that answers right four times out of five is not a path that
    answers right, so a varying path keeps the list rather than a sample.
    """
    seen = sorted({answer for given in answers.values() for answer in given})
    by_path = {
        label: given[0] if len(set(given)) == 1 else sorted(set(given))
        for label, given in answers.items()
    }
    return seen, by_path


def _wall(command: list[str], folder: Path, data: Path, env: dict) -> tuple[float, str]:
    """One run, timed from outside: what the judge's clock would show."""
    program = folder / command[0] if command[0].startswith(".") else None
    if program is not None and not program.is_file():
        raise FileNotFoundError(f"{shlex.join(command)}: the build wrote no executable here")
    with data.open("rb") as stream:
        started = time.perf_counter()
        done = subprocess.run(
            command,
            cwd=folder,
            stdin=stream,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
    return elapsed, done.stdout.strip()


#: The runtime a launcher imports at every start, staged beside the program.
_RUNTIME = ("ppy", "ppy_runtime")


def _staged(
    folder: Path, stem: str, room: Path, standalone: str | None, skipped: dict[str, str]
) -> tuple[Path, dict]:
    """Copy the program and the runtime somewhere the filesystem is fast.

    A checkout on a mounted Windows drive answers `stat` and `open` several
    times slower than a native one, and a launch is mostly imports -- 51 ms
    against 214 ms for the same binary and the same machine. Measuring from
    a staged copy reports what an ordinary install does, not what the mount
    costs.
    """
    source_root = Path(__file__).resolve().parents[2] / "src"
    staged_src = room / "src"
    for package in _RUNTIME:
        shutil.copytree(source_root / package, staged_src / package)
    work = room / folder.name
    work.mkdir()
    for name in (f"{stem}.ppy", f"{stem}.c", "pyproject.toml"):
        if (folder / name).is_file():
            shutil.copy2(folder / name, work / name)
    if not (work / "pyproject.toml").is_file():
        (work / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    where = {**os.environ, "PYTHONPATH": str(staged_src)}
    if "C scanf" not in skipped:
        subprocess.run(
            ["gcc", "-O3", "-o", f"{stem}_c", f"{stem}.c"],
            cwd=work,
            capture_output=True,
            check=True,
        )
    if "ppy build" not in skipped:
        subprocess.run(
            [*PPY, "build", f"{stem}.ppy", "-o", "dist"],
            cwd=work,
            capture_output=True,
            text=True,
            check=True,
            env=where,
        )
    if standalone is not None:
        shutil.copy2(HERE / "standalone" / f"{standalone}.ppy", work / f"{standalone}.ppy")
        subprocess.run(
            [*PPY, "build", "--standalone", f"{standalone}.ppy", "-o", "native"],
            cwd=work,
            capture_output=True,
            text=True,
            check=True,
            env=where,
        )
    return work, where


def environment() -> dict:
    """What the numbers were measured on.

    A wall time means nothing without this: a different CPU, interpreter, or
    compiler is a different measurement, not a regression.
    """
    compiler = subprocess.run(
        ["cc", "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_name(),
        "cores": os.cpu_count(),
        "c_compiler": compiler[0].strip() if compiler else "unknown",
        "ppy": _ppy_version(),
        "rounds": ROUNDS,
        "timing": "wall time of the whole process, measured from outside",
        "staged": "program and runtime copied to a native filesystem first",
        "semantics": "ppy build defaults to wrap semantics; ppy run keeps Python integers",
        "missing": available(),
    }


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ppy_version() -> str:
    done = subprocess.run([*PPY, "--version"], capture_output=True, text=True, check=False)
    return (done.stdout or done.stderr).strip() or "unknown"


def measure(only: list[str] | None = None, rounds: int = ROUNDS) -> dict:
    """Every problem on every path: milliseconds of wall time, and the answer."""
    skipped = available()
    for path, reason in sorted(skipped.items()):
        print(f"skipping {path}: {reason}", file=sys.stderr)
    results: dict = {}
    for folder_name, stem, generate in PROBLEMS:
        if only and not any(token in folder_name for token in only):
            continue
        folder = HERE / folder_name
        with tempfile.TemporaryDirectory() as scratch:
            room = Path(scratch)
            variant = STANDALONE.get(folder_name)
            if "standalone" in skipped:
                variant = None
            work, where = _staged(folder, stem, room, variant, skipped)
            data = room / "input.txt"
            data.write_text(generate(), encoding="utf-8")
            row: dict = {}
            #: Every repetition's answer, not the last one: a path that is
            #: right four times out of five is not a path that is right.
            answers: dict[str, list[str]] = {}
            for label, command in _commands(stem, variant):
                if label in skipped:
                    continue
                samples = []
                for _ in range(rounds):
                    elapsed, answer = _wall(command, work, data, where)
                    samples.append(elapsed)
                    answers.setdefault(label, []).append(answer)
                row[label] = {
                    "mean": statistics.mean(samples),
                    "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                }
            row["answers"], row["by_path"] = collapse(answers)
        results[folder_name] = row
    return results


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    results = measure(only or None)
    disagreed = any(len(row["answers"]) != 1 for row in results.values())
    if "--json" in sys.argv:
        print(json.dumps({"environment": environment(), "problems": results}, indent=1))
        return 1 if disagreed else 0
    print(f"{'problem':<18}" + "".join(f"{name:>18}" for name in PATHS))
    for folder_name, row in results.items():
        cells = "".join(
            f"{row[name]['mean']:9.1f} ±{row[name]['stdev']:5.1f}" if name in row else f"{'—':>16}"
            for name in PATHS
        )
        print(f"{folder_name:<18}{cells}")
        if len(row["answers"]) != 1:
            print(f"  DISAGREE: {row['by_path']}")
    return 1 if disagreed else 0


if __name__ == "__main__":
    raise SystemExit(main())
