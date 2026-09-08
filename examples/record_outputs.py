"""Run the commands each example README lists and write what they print back into it.

Every README has a `## Run it` block. This runs each of its commands in the
example's folder and keeps the output under `## What it prints`, so the
README shows the command and its answer side by side. Timings differ from
machine to machine and are shown as one run; answers do not, and a run that
changes one is a change to the example.

    python examples/record_outputs.py             # every example
    python examples/record_outputs.py 15_ 40_     # folders matching a token
    python examples/record_outputs.py --check     # report READMEs that are behind
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
START = "<!-- outputs:start -->"
END = "<!-- outputs:end -->"
HEADING = "## What it prints"
RUN_IT = re.compile(r"^## Run it\n\n```bash\n(.*?)```", re.DOTALL | re.MULTILINE)
MAX_LINES = 40
TIMEOUT = 600

#: Tools an example may name that are not part of this repository's
#: environment; a command needing one is shown as not run rather than failed.
OPTIONAL_TOOLS = ("torchrun", "accelerate", "nvcc", "hipcc")


def readmes(only: list[str]) -> list[Path]:
    found = sorted(ROOT.glob("[0-9]*/README.md")) + sorted(ROOT.glob("[0-9]*/[0-9]*/README.md"))
    if only:
        found = [p for p in found if any(token in str(p.relative_to(ROOT)) for token in only)]
    return found


def commands(text: str) -> list[str]:
    """The commands of the `## Run it` block, comments stripped, in order."""
    block = RUN_IT.search(text)
    if block is None:
        return []
    found = []
    for line in block.group(1).splitlines():
        command = line.split("  #", 1)[0].rstrip()
        if command.startswith("#") or not command.strip():
            continue
        found.append(command.strip())
    return found


def _environment() -> dict[str, str]:
    """The interpreter running this script is the `python` and `ppy` the commands see."""
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PPY_QUIET"] = "1"
    return env


def run(command: str, folder: Path) -> tuple[str, str]:
    """One command's output and a one-word verdict: `ok`, `failed`, or `skipped`."""
    env = _environment()
    tool = command.split(maxsplit=1)[0]
    if tool in OPTIONAL_TOOLS and shutil.which(tool, path=env["PATH"]) is None:
        return f"not run here: `{tool}` is not installed", "skipped"
    try:
        done = subprocess.run(
            command,
            shell=True,
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {TIMEOUT} s", "failed"
    # A tool that names the folder names it as the reader's `.`, not this machine's path.
    out = done.stdout.rstrip("\n").replace(str(folder.resolve()), ".")
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-4:])
        return f"{out}\n{tail}".strip() + f"\n[exit {done.returncode}]", "failed"
    return out, "ok"


def _clip(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= MAX_LINES:
        return text
    kept = lines[:MAX_LINES]
    return "\n".join(kept) + f"\n… {len(lines) - MAX_LINES} more lines"


def section(folder: Path, listed: list[str]) -> tuple[str, list[str]]:
    """The `## What it prints` section for one README, and what failed."""
    parts = [START, HEADING, ""]
    failed = []
    for command in listed:
        output, verdict = run(command, folder)
        if verdict == "failed":
            failed.append(command)
        parts.append(f"**`{command}`**")
        parts.append("")
        if verdict == "skipped":
            parts.append(f"*{output}*")
        elif output:
            parts.append("```text")
            parts.append(_clip(output))
            parts.append("```")
        else:
            parts.append("*(prints nothing; exits 0)*")
        parts.append("")
    parts.append(END)
    return "\n".join(parts) + "\n", failed


def _place(text: str, rendered: str) -> str:
    """Put the section where it was, or right after the `## Run it` block."""
    if START in text and END in text:
        head = text[: text.index(START)]
        tail = text[text.index(END) + len(END) :].lstrip("\n")
        return head + rendered + ("\n" + tail if tail else "")
    block = RUN_IT.search(text)
    assert block is not None
    end = block.end()
    trailer = text[end:]
    return (
        text[:end].rstrip("\n")
        + "\n\n"
        + rendered
        + ("\n" + trailer.lstrip("\n") if trailer.strip() else "")
    )


def main() -> int:
    check = "--check" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    behind = 0
    for readme in readmes(only):
        text = readme.read_text(encoding="utf-8")
        listed = commands(text)
        if not listed:
            continue
        rendered, failed = section(readme.parent, listed)
        updated = _place(text, rendered)
        where = readme.relative_to(ROOT)
        for command in failed:
            print(f"[FAIL] {where}: {command}")
        if updated != text:
            behind += 1
            if check:
                print(f"[BEHIND] {where}")
            else:
                readme.write_text(updated, encoding="utf-8")
                print(f"[WROTE] {where} ({len(listed)} commands)")
        else:
            print(f"[OK]    {where}")
    return 1 if (check and behind) else 0


if __name__ == "__main__":
    raise SystemExit(main())
