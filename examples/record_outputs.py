"""Run the commands each example README lists and write what they print back into it.

Every README has a `## Run it` block. This runs each of its commands in the
example's folder and keeps the output under `## What it prints`, so the
README shows the command and its answer side by side. Commands that print
the same thing share one block. A short output is shown as it is; a longer
one is folded into a `<details>` block the reader opens; a very long one is
kept as a file under the folder's `outputs/`, linked from the README and
embedded by the documentation site. Each output exists once. Timings differ
from machine to machine and are shown as one run; answers do not, and a run
that changes one is a change to the example.

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
RUN_IT = re.compile(r"^## Run it\n.*?```bash\n(.*?)```", re.DOTALL | re.MULTILINE)
INLINE_LINES = 20
FOLDED_LINES = 300
OUTPUTS = "outputs"
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


def _slug(index: int, command: str) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", command.lower()).strip("-")
    return f"{index:02d}-{words[:40].rstrip('-')}.txt"


def _render(folder: Path, index: int, command: str, text: str) -> list[str]:
    """One output, once: inline, folded, or as a file the README links to."""
    lines = text.splitlines()
    if len(lines) <= INLINE_LINES:
        return ["```text", text, "```"]
    if len(lines) <= FOLDED_LINES:
        return [
            '<details markdown="1">',
            f"<summary>{len(lines)} lines</summary>",
            "",
            "```text",
            text,
            "```",
            "",
            "</details>",
        ]
    name = _slug(index, command)
    (folder / OUTPUTS).mkdir(exist_ok=True)
    (folder / OUTPUTS / name).write_text(text + "\n", encoding="utf-8")
    return [f"*{len(lines)} lines: [{OUTPUTS}/{name}]({OUTPUTS}/{name})*"]


def section(folder: Path, listed: list[str]) -> tuple[str, list[str]]:
    """The `## What it prints` section for one README, and what failed.

    Commands that print the same thing share one block, headed by all of
    them: that the three paths agree is the compiler's contract and
    `run_all.py`'s check, not something a README needs to show three times.
    An output that differs -- a report, a timing -- keeps its own block.
    """
    parts = [START, HEADING, ""]
    failed = []
    shutil.rmtree(folder / OUTPUTS, ignore_errors=True)
    groups: list[tuple[list[str], str, str]] = []
    for command in listed:
        output, verdict = run(command, folder)
        if verdict == "failed":
            failed.append(command)
        for group in groups:
            if verdict == "ok" and group[2] == "ok" and group[1] == output:
                group[0].append(command)
                break
        else:
            groups.append(([command], output, verdict))
    for index, (members, output, verdict) in enumerate(groups, 1):
        parts.append(", ".join(f"**`{command}`**" for command in members))
        parts.append("")
        if verdict == "skipped":
            parts.append(f"*{output}*")
        elif output:
            parts.extend(_render(folder, index, members[0], output))
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
