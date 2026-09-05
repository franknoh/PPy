"""Run the converter over the compiler's own source, and hold the line.

Thirty thousand lines of real Python is the largest corpus this project has,
and it is the one that found the empty-identifier crash, the fifty-six
missing exceptions, and the cascades of `<unknown>`. So it is checked: every
migration must finish without a traceback, must never show `<unknown>` in a
message, and may not report more errors than the last recorded run. The
count only ratchets down; `--write` records a lower one.

    python scripts/dogfood.py           # check against the recorded ceiling
    python scripts/dogfood.py --write   # record today's counts as the ceiling
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
RECORDED = ROOT / "scripts" / "dogfood.json"
PPY = [sys.executable, "-m", "ppy_compiler"]

#: What is migrated: the compiler itself, its runtime, and the `ppy` package.
#: Each is a project root's worth of real code with a different shape --
#: dispatch-heavy analysis, ctypes and manifests, decorators and markers.
TARGETS = ("src/ppy_compiler", "src/ppy_runtime", "src/ppy")

_ERROR = re.compile(r"^error\[(E\d{4})\]", re.MULTILINE)
_WITHHELD = re.compile(r"^warning\[W2006\]: (\d+) further error", re.MULTILINE)


def migrate(target: str) -> dict:
    """One dry-run migration, measured and summarized."""
    started = time.perf_counter()
    done = subprocess.run(
        [*PPY, "migrate", "--dry-run", target],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Diagnostics go to stderr; stdout carries the converted source, and the
    # compiler's own source spells "<unknown>" on purpose in three places.
    report = done.stderr
    codes = _own_errors(report, (ROOT / target).resolve())
    withheld = _WITHHELD.search(report)
    return {
        "seconds": round(time.perf_counter() - started, 1),
        "errors": len(codes),
        "by_code": dict(sorted(_count(codes).items())),
        "withheld": int(withheld.group(1)) if withheld else 0,
        "unknown_in_output": report.count("<unknown>"),
        "traceback": "Traceback (most recent call last)" in report + done.stdout,
        "crashed": done.returncode not in (0, 1),
    }


def _own_errors(report: str, root: Path) -> list[str]:
    """The codes of the errors located in the target's own files.

    A module reached through an import -- `ppy` reaching `ppy_runtime` -- is
    another target's business and counted there; an error with no location
    is the target's.
    """
    codes: list[str] = []
    lines = report.splitlines()
    for index, line in enumerate(lines):
        found = _ERROR.match(line)
        if found is None:
            continue
        below = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if below.startswith("-->"):
            path = below[3:].strip().rsplit(":", 2)[0]
            try:
                inside = Path(path).resolve().is_relative_to(root)
            except (OSError, ValueError):
                inside = True
            if not inside:
                continue
        codes.append(found.group(1))
    return codes


def _count(codes: list[str]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for code in codes:
        counted[code] = counted.get(code, 0) + 1
    return counted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="record today's counts")
    options = parser.parse_args()
    recorded = json.loads(RECORDED.read_text(encoding="utf-8")) if RECORDED.is_file() else {}

    failed = False
    fresh: dict[str, dict] = {}
    for target in TARGETS:
        result = migrate(target)
        fresh[target] = result
        ceiling = recorded.get(target, {}).get("errors")
        verdict = []
        if result["traceback"] or result["crashed"]:
            verdict.append("crashed")
        if result["unknown_in_output"]:
            verdict.append(f"`<unknown>` shown {result['unknown_in_output']} time(s)")
        if ceiling is not None and result["errors"] > ceiling:
            verdict.append(f"{result['errors']} errors, ceiling {ceiling}")
        state = "ok" if not verdict else "FAIL: " + "; ".join(verdict)
        print(
            f"{target:<18} {result['seconds']:6.1f}s  errors {result['errors']:4d}"
            f"  withheld {result['withheld']:4d}  {state}"
        )
        failed |= bool(verdict)

    if options.write:
        RECORDED.write_text(json.dumps(fresh, indent=1) + "\n", encoding="utf-8")
        print(f"recorded in {RECORDED.relative_to(ROOT)}")
    elif not failed:
        lowered = [t for t in TARGETS if recorded.get(t, {}).get("errors", 0) > fresh[t]["errors"]]
        if lowered:
            print(f"the ceiling can come down for {', '.join(lowered)}: rerun with --write")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
