"""Run every example three ways and report where the answers differ."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PATHS = [("plain", []), ("python", ["-m", "ppy_compiler"]), ("native", ["-m", "ppy_compiler", "run"])]
NOISE = ("compiling ", "SyntaxWarning", "  if ", "staged ", "warning:")
#: An example prefixes a line with this when it reports which path it is on,
#: which is the one thing that is *supposed* to differ between the three.
PATH_SPECIFIC = "# "


#: Examples that print their own timings differ between paths by design.
_MEASURED = [
    (re.compile(r"\d+\.\d+\s*(ms|us|µs|ns|s)\b"), "<time>"),   # wall clock
    (re.compile(r"\d+\.\d+x\b"), "<ratio>"),                    # speedup
    (re.compile(r"\d+\.\d+e[-+]\d+"), "<delta>"),               # float difference
    (re.compile(r"\s{2,}"), " "),                               # column padding
]


def _clean(text: str) -> str:
    kept = (
        l for l in text.splitlines()
        if not l.startswith(NOISE) and not l.startswith(PATH_SPECIFIC) and l.strip()
    )
    return "\n".join(_normalize(line) for line in kept)


def _normalize(line: str) -> str:
    """Erase what is measured rather than computed, so only answers are compared."""
    for pattern, placeholder in _MEASURED:
        line = pattern.sub(placeholder, line)
    return line.strip()


def _run(args: list[str], cwd: Path) -> tuple[int, str]:
    done = subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=240, check=False
    )
    return done.returncode, _clean(done.stdout)


def main() -> int:
    entries = sorted(ROOT.glob("[0-9]*/[a-z]*.ppy")) + sorted(ROOT.glob("*/src/app.ppy"))
    only = sys.argv[1:]
    if only:
        entries = [e for e in entries if any(token in str(e) for token in only)]
    failures = 0
    for entry in entries:
        cwd, name = entry.parent, entry.name
        checked = subprocess.run(
            [sys.executable, "-m", "ppy_compiler", "check", name],
            cwd=cwd, capture_output=True, text=True, check=False,
        )
        status = "ok" if checked.returncode == 0 else "CHECK FAILED"
        results: dict[str, tuple[int, str]] = {}
        for label, args in PATHS:
            try:
                results[label] = _run([*args, name], cwd)
            except subprocess.TimeoutExpired:
                results[label] = (-1, "<timeout>")
        outputs = {label: out for label, (_, out) in results.items()}
        codes = {label: code for label, (code, _) in results.items()}
        agree = len(set(outputs.values())) == 1 and set(codes.values()) == {0}
        if status != "ok" or not agree:
            failures += 1
        mark = "PASS" if status == "ok" and agree else "FAIL"
        rel = entry.relative_to(ROOT)
        print(f"[{mark}] {rel}  check={status}  exit={codes}")
        if not agree:
            for label in outputs:
                print(f"        {label}: {outputs[label]!r}"[:300])
    print(f"\n{len(entries) - failures}/{len(entries)} examples agree on all three paths")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
