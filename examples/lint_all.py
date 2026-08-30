"""Run pylint over every example, source and converted alike.

`.ppy` is ordinary Python, so the same linter applies. Files are linted in
place under their own directory so the module name pylint sees is the real one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
SKIP = {"run_all.py", "verify_conversions.py", "lint_all.py", "bench_startup.py", "bench_boundary.py"}

#: Folders whose `.py` is the deliberately messy "before" picture of a
#: migration; the claim is that the `.ppy` lints clean, not the input.
BEFORE_PICTURES = {"30_migrate"}


def _lint(folder: Path, names: list[str]) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-m", "pylint", "--rcfile", str(ROOT / ".pylintrc"), *names],
        cwd=folder,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    return [line for line in done.stdout.splitlines() if ".py:" in line and ": " in line]


def main() -> int:
    failures = 0
    checked = 0
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        sources = sorted(
            p for p in folder.rglob("*.py") if p.name not in SKIP and ".ppy-cache" not in str(p)
        )
        if folder.name in BEFORE_PICTURES:
            sources = []
        converted = sorted(p for p in folder.rglob("*.ppy") if ".ppy-cache" not in str(p))
        if not sources and not converted:
            continue
        with tempfile.TemporaryDirectory() as scratch:
            staged = Path(scratch)
            names: list[str] = []
            for path in sources + converted:
                target = staged / (path.stem + ".py")
                if target.exists():
                    target = staged / (path.stem + "_converted.py")
                shutil.copyfile(path, target)
                names.append(target.name)
            found = _lint(staged, names)
        checked += len(names)
        if found:
            failures += len(found)
            print(f"[FAIL] {folder.name}")
            for line in found:
                print(f"        {line}")
        else:
            print(f"[OK]   {folder.name}  ({len(names)} file(s))")

    print(f"\n{checked - failures}/{checked} files pass pylint with examples/.pylintrc")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
