"""Check that every file claiming to be `ppy convert` output really is.

Convention, so a filename cannot mislead:

  * `<name>.py`   the input, ordinary untyped Python
  * `<name>.ppy`  exactly what `ppy convert <name>.py` writes

This regenerates every `<name>.ppy` from its `.py` and fails on any difference,
so a hand edit to a file that claims to be generated is caught here rather than
believed. It also checks that each folder's README says which of the two its
`.ppy` is, and that the claim matches what is on disk.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent

#: Directories whose conversions use extra flags.
FLAGS: dict[str, list[str]] = {
    "20_inventory": ["--promote-buffers"],
    "21_training_torch": ["--promote-buffers"],
    "22_training_jax": ["--promote-buffers"],
}


def _pairs() -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for source in sorted(ROOT.glob("*/*.py")):
        if source.name.startswith(("run_all", "verify_", "consumer")):
            continue
        generated = source.with_suffix(".ppy")
        if generated.exists():
            found.append((source, generated))
    return found


def _convert(source: Path, into: Path) -> tuple[bool, str]:
    """Convert `source` inside a scratch copy so nothing on disk is touched."""
    staged = into / source.name
    staged.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config = source.parent / "pyproject.toml"
    if config.exists():
        (into / "pyproject.toml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "convert",
            staged.name,
            *FLAGS.get(source.parent.name, []),
        ],
        cwd=into,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    produced = staged.with_suffix(".ppy")
    if not produced.exists():
        return False, done.stderr
    return True, produced.read_text(encoding="utf-8")


def _lint(path: Path, scratch: Path) -> set[str]:
    """Pylint findings for a file, as message symbols with line numbers dropped."""
    staged = scratch / (path.stem + "_lint.py")
    staged.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pylint",
            "--enable=all",
            "--score=n",
            "--rcfile",
            str(ROOT / ".pylintrc"),
            staged.name,
        ],
        cwd=scratch,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    found: set[str] = set()
    for line in done.stdout.splitlines():
        if ": " in line and line.startswith(staged.name):
            found.add(line.rsplit("(", 1)[-1].rstrip(")"))
    return found


def _missing_libraries(folder: Path) -> set[str]:
    """Optional libraries this example needs and this machine lacks."""
    import importlib.util

    needed: set[str] = set()
    sources = [*folder.rglob("*.py"), *folder.rglob("*.ppy")]
    for source in sources:
        if "__pycache__" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        for library in ("torch", "jax", "uvicorn", "numpy", "pydantic"):
            if f"import {library}" in text:
                needed.add(library)
    return {lib for lib in needed if importlib.util.find_spec(lib) is None}


def main() -> int:
    failures = 0
    skipped = 0
    pairs = _pairs()
    for source, generated in pairs:
        missing = _missing_libraries(source.parent)
        if missing:
            # A skip is not a verification; say what was not proven and why.
            print(f"[SKIP] {generated.relative_to(ROOT)} (needs {', '.join(sorted(missing))})")
            skipped += 1
            continue
        with tempfile.TemporaryDirectory() as scratch:
            ok, produced = _convert(source, Path(scratch))
        rel = generated.relative_to(ROOT)
        if not ok:
            print(f"[FAIL] {rel}  conversion did not run\n{produced}")
            failures += 1
            continue
        on_disk = generated.read_text(encoding="utf-8")
        if on_disk != produced:
            print(f"[FAIL] {rel}  differs from `ppy convert` output")
            failures += 1
            continue
        with tempfile.TemporaryDirectory() as scratch:
            before = _lint(source, Path(scratch))
            after = _lint(generated, Path(scratch))
        introduced = after - before
        if introduced:
            print(f"[FAIL] {rel}  conversion introduced {sorted(introduced)}")
            failures += 1
            continue
        print(f"[OK]   {rel}  is `ppy convert` output and adds no lint finding")

    checked = len(pairs)
    for folder in sorted(ROOT.glob("[0-9]*/")):
        readme = folder / "README.md"
        if not readme.exists():
            continue
        checked += 1
        text = readme.read_text(encoding="utf-8")
        claims_generated = "Generated, not hand-written" in text
        has_source = any(
            ppy.with_suffix(".py").exists()
            and not ppy.with_suffix(".py").name.startswith(("run_all", "verify_", "consumer"))
            for ppy in folder.rglob("*.ppy")
            if ".ppy-cache" not in str(ppy)
        )
        if claims_generated == has_source:
            continue
        wrong = (
            "claims generated but has no .py source"
            if claims_generated
            else ("has a .py source but does not say its .ppy is generated")
        )
        print(f"[FAIL] {readme.relative_to(ROOT)}  {wrong}")
        failures += 1

    print(f"\n{checked - skipped - failures}/{checked - skipped} conversion claims verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
