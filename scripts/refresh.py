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

#: The overview table in the folder README, which shows every problem at once.
OVERVIEW = re.compile(
    r"\| \| problem \| plain \| `ppy build` \| `--standalone` \| C \(`scanf`\) \|\n"
    r"\|---\|---\|---:\|---:\|---:\|---:\|\n(?:\|.*\n)+"
)

#: How the overview names each problem, in the order it lists them.
TITLES = {
    "15a_nqueens": "N-Queens",
    "15b_dijkstra": "shortest path",
    "15c_kmp": "substring search",
    "15d_segment_tree": "range sums",
    "15e_lis": "longest increasing subsequence",
    "15f_input": "counting inversions",
}


def _same_file(left: Path, right: Path) -> bool:
    """Whether two paths name one file, whether or not it exists yet."""
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve() == right.resolve()


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


def refresh_measurements(write: bool, record: Path | None = None) -> tuple[list[str], bool]:
    """Wall times, against what the documentation says they are.

    Returns what drifted and whether that is a failure. A wrong answer always
    is -- `--write` records nothing and the run fails, because a timing for a
    program that prints the wrong number is not a baseline. So is a machine
    that cannot build every path: a record with columns missing would replace
    a complete one, and the gap would read as a result. Drift on a machine
    that is not the recorded one is neither: two computers are two
    measurements, not a regression.
    """
    sys.path.insert(0, str(ALGORITHMS))
    from importlib import import_module

    bench = import_module("bench")
    missing = bench.available()
    fresh = bench.measure()
    if record is not None:
        payload = {"environment": bench.environment(), "problems": fresh}
        record.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    if missing:
        # Fatal whether or not `--write` was asked for. A run that could not
        # build every path has nothing to compare and nothing to record, and
        # a scheduled job that reported success would leave a partial artifact
        # looking like a measurement.
        return [
            *(f"{path}: {reason}" for path, reason in sorted(missing.items())),
            "  nothing was recorded: a partial run is not a baseline",
            "  install what is missing, or measure somewhere that has it",
        ], True
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
    elif drifted and not write:
        drifted.append("  rerun with --write once these are the numbers to keep")
    if write and not missing:
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


def _overview(problems: dict) -> str:
    """The folder README's table: every problem, one row each.

    Bold marks a cell that beat the C reference, which is a claim the numbers
    have to keep making rather than one written down once.
    """
    header = [
        "| | problem | plain | `ppy build` | `--standalone` | C (`scanf`) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rows = []
    for name, title in TITLES.items():
        row = problems.get(name)
        if row is None:
            continue
        # Bold means "beat C", so without a C column there is nothing to beat
        # and every cell stays plain rather than the table failing to render.
        reference = row.get("C scanf", {}).get("mean")
        cells = []
        for path in ("plain", "ppy build", "standalone", "C scanf"):
            if path not in row:
                cells.append("—")
                continue
            cell = f"{row[path]['mean']:.1f} ms"
            beats = path != "C scanf" and reference is not None and row[path]["mean"] < reference
            cells.append(f"**{cell}**" if beats else cell)
        rows.append(f"| [{name[:3]}]({name}/) | {title} | " + " | ".join(cells) + " |")
    return "\n".join(header + rows) + "\n"


def refresh_tables(write: bool) -> list[str]:
    """Every problem README's table, against the recorded measurements.

    The numbers live in `measurements.json`; a README is a rendering of it,
    so the two cannot disagree for longer than one run of this script.
    """
    if not RECORDED.is_file():
        return []
    problems = json.loads(RECORDED.read_text(encoding="utf-8")).get("problems", {})
    behind = _rendered(ALGORITHMS / "README.md", OVERVIEW, _overview(problems), write)
    for name, row in problems.items():
        document = ALGORITHMS / name / "README.md"
        if not document.is_file():
            continue
        behind.extend(_rendered(document, TABLE, _table(row), write))
    return behind


def _rendered(document: Path, pattern: re.Pattern[str], wanted: str, write: bool) -> list[str]:
    """Put `wanted` where `pattern` matches, and say whether it had to move."""
    where = document.relative_to(EXAMPLES)
    text = document.read_text(encoding="utf-8")
    if pattern.search(text) is None:
        return [f"{where}: no table here to fill"]
    updated = pattern.sub(wanted.replace("\\", "\\\\"), text, count=1)
    if updated == text:
        return []
    if write:
        document.write_text(updated, encoding="utf-8")
    return [f"{where}: the table is behind measurements.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="fix what has drifted")
    parser.add_argument("--quick", action="store_true", help="skip the benchmark")
    parser.add_argument(
        "--record",
        type=Path,
        metavar="FILE",
        help="also write this run's raw measurements there, drift or not",
    )
    options = parser.parse_args()
    if options.record is not None and _same_file(options.record, RECORDED):
        # `--record` writes whatever this run measured; the baseline is only
        # written once the run is judged worth keeping. Pointing them at one
        # file would let the unjudged numbers land there first.
        parser.error(f"--record cannot be the recorded baseline ({RECORDED})")
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
        drifted, fatal = refresh_measurements(options.write, options.record)
        state = "recorded" if options.write else "drifted"
        print(f"measurements: {'ok' if not drifted else f'{len(drifted)} {state}'}")
        for line in drifted:
            print(f"  {line}")
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
