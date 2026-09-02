"""Keep the examples, their conversions, and the measured numbers current.

Three things drift apart as the compiler changes: a hand-written example can
fall behind the syntax, a generated `.ppy` can fall behind its `.py`, and a
number written into a README can fall behind the machine. This checks all
three, and with `--write` fixes the first two and records the third.

    python scripts/refresh.py            # report what has drifted
    python scripts/refresh.py --write    # regenerate and record
    python scripts/refresh.py --quick    # skip the benchmark
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
ALGORITHMS = EXAMPLES / "15_algorithms"
RECORDED = ALGORITHMS / "measurements.json"
PPY = [sys.executable, "-m", "ppy_compiler"]

#: A measurement this far from the recorded one is drift rather than noise.
TOLERANCE = 0.25

#: How each problem README names the path `bench.py` measured.
ROWS = [
    ("plain CPython", "plain"),
    ("`ppy run`", "ppy run"),
    ("`ppy build`", "ppy build"),
    ("`ppy build --standalone`", "standalone"),
    ("C (`gcc -O3`, `scanf`)", "C scanf"),
]

TABLE = re.compile(r"\| path \| wall \|\n\|---\|---:\|\n(?:\|.*\n)+")


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def check_syntax() -> list[str]:
    """Every example still checks under today's compiler."""
    failures = []
    entries = sorted(EXAMPLES.glob("[0-9]*/*.ppy")) + sorted(EXAMPLES.glob("[0-9]*/*/*.ppy"))
    for entry in entries:
        if ".ppy-cache" in str(entry):
            continue
        done = _run([*PPY, "check", entry.name], entry.parent)
        if done.returncode != 0:
            failures.append(
                f"{entry.relative_to(EXAMPLES)}: {done.stderr.strip().splitlines()[-1]}"
            )
    return failures


def refresh_conversions(write: bool) -> list[str]:
    """Every generated `.ppy` still matches what `ppy convert` writes today."""
    from importlib import import_module

    verify = import_module("verify_conversions")
    stale = []
    for source, generated in verify._pairs():
        flags = verify.FLAGS.get(source.parent.name, [])
        command = verify.COMMANDS.get(source.parent.name, "convert")
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            ok, produced = verify._convert(source, Path(scratch))
        if not ok:
            stale.append(f"{generated.relative_to(EXAMPLES)}: conversion failed")
            continue
        if produced != generated.read_text(encoding="utf-8"):
            stale.append(f"{generated.relative_to(EXAMPLES)} (ppy {command} {' '.join(flags)})")
            if write:
                generated.write_text(produced, encoding="utf-8")
    return stale


def refresh_measurements(write: bool) -> tuple[list[str], bool]:
    """Wall times, against what the documentation says they are.

    Returns what drifted and whether that is a failure. A wrong answer always
    is -- `--write` records nothing and the run fails, because a timing for a
    program that prints the wrong number is not a baseline. Drift on a machine
    that is not the recorded one never is: two computers are two measurements,
    not a regression.
    """
    sys.path.insert(0, str(ALGORITHMS))
    from importlib import import_module

    bench = import_module("bench")
    fresh = bench.measure()
    stored = json.loads(RECORDED.read_text(encoding="utf-8")) if RECORDED.is_file() else {}
    recorded = stored.get("problems", stored)
    before_environment = stored.get("environment", {})
    here = bench.environment()

    wrong = [
        f"{problem}: the paths disagree -- {row['answers']}"
        for problem, row in fresh.items()
        if len(row.get("answers", [])) != 1
    ]
    if wrong:
        return [*wrong, "  nothing was recorded: a wrong answer has no useful timing"], True

    drifted = []
    moved = [
        f"  the machine changed: {key} was {before_environment[key]!r}, now {here[key]!r}"
        for key in ("processor", "python", "platform", "c_compiler")
        if before_environment.get(key) not in (None, here.get(key))
    ]
    for problem, row in fresh.items():
        before = recorded.get(problem, {})
        for path, measured in row.items():
            if path == "answers":
                if before.get("answers") and before["answers"] != measured:
                    drifted.append(f"{problem}: the answer changed to {measured}")
                continue
            if path == "by_path":
                continue
            was = before.get(path, {}).get("mean")
            if was is None:
                drifted.append(f"{problem} / {path}: new, {measured['mean']:.1f} ms")
                continue
            if abs(measured["mean"] - was) > TOLERANCE * was:
                drifted.append(f"{problem} / {path}: {was:.1f} ms -> {measured['mean']:.1f} ms")
    if drifted and moved:
        drifted.extend(moved)
        drifted.append("  drift across machines is not a regression; rerun with --write here")
    if write:
        payload = {
            "environment": here,
            "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "problems": fresh,
        }
        RECORDED.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return drifted, bool(drifted) and not moved and not write


def _table(row: dict) -> str:
    """The wall-time table a problem README shows, from what was measured."""
    timed = {label: row[path] for label, path in ROWS if path in row}
    best = min(timed, key=lambda label: timed[label]["mean"])
    lines = ["| path | wall |", "|---|---:|"]
    for label, measured in timed.items():
        cell = f"{measured['mean']:.1f} ± {measured['stdev']:.1f} ms"
        lines.append(f"| {label} | {'**' + cell + '**' if label == best else cell} |")
    return "\n".join(lines) + "\n"


def refresh_tables(write: bool) -> list[str]:
    """Every problem README's table, against the recorded measurements.

    The numbers live in `measurements.json`; a README is a rendering of it,
    so the two cannot disagree for longer than one run of this script.
    """
    if not RECORDED.is_file():
        return []
    problems = json.loads(RECORDED.read_text(encoding="utf-8")).get("problems", {})
    behind = []
    for name, row in problems.items():
        document = ALGORITHMS / name / "README.md"
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        wanted = _table(row)
        if TABLE.search(text) is None:
            behind.append(f"{name}/README.md: no wall-time table to fill")
            continue
        updated = TABLE.sub(wanted.replace("\\", "\\\\"), text, count=1)
        if updated != text:
            behind.append(f"{name}/README.md: the table is behind measurements.json")
            if write:
                document.write_text(updated, encoding="utf-8")
    return behind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="fix what has drifted")
    parser.add_argument("--quick", action="store_true", help="skip the benchmark")
    options = parser.parse_args()
    sys.path.insert(0, str(EXAMPLES))

    failed = False
    syntax = check_syntax()
    print(f"syntax: {'ok' if not syntax else f'{len(syntax)} example(s) do not check'}")
    for line in syntax:
        print(f"  {line}")
    failed |= bool(syntax)

    stale = refresh_conversions(options.write)
    state = "rewritten" if options.write else "stale"
    print(f"conversions: {'ok' if not stale else f'{len(stale)} {state}'}")
    for line in stale:
        print(f"  {line}")
    failed |= bool(stale) and not options.write

    if not options.quick:
        drifted, fatal = refresh_measurements(options.write)
        state = "recorded" if options.write else "drifted"
        print(f"measurements: {'ok' if not drifted else f'{len(drifted)} {state}'}")
        for line in drifted:
            print(f"  {line}")
        if fatal:
            print("\n  rerun with --write once the numbers are the ones to keep")
        failed |= fatal

    behind = refresh_tables(options.write)
    state = "rewritten" if options.write else "behind"
    print(f"tables: {'ok' if not behind else f'{len(behind)} {state}'}")
    for line in behind:
        print(f"  {line}")
    failed |= bool(behind) and not options.write

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
