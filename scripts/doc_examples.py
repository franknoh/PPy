"""Hold the documentation's Python blocks to the project's own formatter.

The docs promise `ppy convert --format` hands its output to ruff; a snippet
in those same docs that ruff would rewrite makes the promise a claim about
someone else's code. Every ```python fence in `README.md` and `docs/` must
parse and already be formatted.

    python scripts/doc_examples.py           # report
    python scripts/doc_examples.py --write   # reformat in place
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"(?<=\n)```python\n(.*?)```", re.DOTALL)


def _sources() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _formatted(block: str) -> str | None:
    """What ruff would write, or None if the block is not parseable Python."""
    try:
        ast.parse(block)
    except SyntaxError:
        return None
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "block.py"
        path.write_text(block, encoding="utf-8")
        done = subprocess.run(
            ["ruff", "format", "-q", str(path)], capture_output=True, text=True, check=False
        )
        if done.returncode != 0:
            return None
        return path.read_text(encoding="utf-8")


def main() -> int:
    write = "--write" in sys.argv
    findings: list[str] = []
    for document in _sources():
        text = document.read_text(encoding="utf-8")
        blocks = FENCE.findall(text)
        for index, block in enumerate(blocks):
            wanted = _formatted(block)
            where = f"{document.relative_to(ROOT)} block {index + 1}"
            if wanted is None:
                findings.append(f"{where}: does not parse as Python")
                continue
            if wanted != block:
                findings.append(f"{where}: not formatted")
                if write:
                    text = text.replace(f"```python\n{block}```", f"```python\n{wanted}```", 1)
        if write:
            document.write_text(text, encoding="utf-8")
    print(f"doc examples: {'ok' if not findings else f'{len(findings)} to fix'}")
    for line in findings:
        print(f"  {line}")
    return 1 if findings and not write else 0


if __name__ == "__main__":
    raise SystemExit(main())
