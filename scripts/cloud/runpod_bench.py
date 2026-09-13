"""Measure the comparisons that want a datacenter GPU, on a Pod this makes and deletes.

    python scripts/cloud/runpod_bench.py                       # 46_gpt2 on an RTX 4090
    python scripts/cloud/runpod_bench.py --gpu-id "NVIDIA L40S"
    python scripts/cloud/runpod_bench.py --sha <commit> 46_gpt2

Every table in the READMEs names the machine it came from, and almost every
one is re-measured by the self-hosted bench runner, which drives
`scripts/compare_docs.py --write` on each push to `dev`. `46_gpt2` is the
exception: GPT-2 XL wants more card than that runner has, so its numbers
are quoted on a rented RTX 4090, and `compare_docs.py` skips the comparison
wherever `PPY_TORCH_CUDA_PYTHON` does not name a PyTorch with a device --
which is everywhere except a Pod this script made. Re-measuring it is
therefore deliberate, and this is the act: rent the card, run the same
`--write` the runner runs, bring the rewritten README and
`compare/measurements.json` back into the checkout, delete the Pod in a
`finally`, and confirm the deletion.

The commit must be pushed, because the Pod clones it. Nothing here is
committed: what comes back is a diff to read. A Pod that existed before
this run is never touched, and the key stays where `runpodctl` keeps it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runpod_matrix import (  # pylint: disable=wrong-import-position
    Failed,
    Pod,
    _authenticated,
    _in_stock,
)

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

#: RunPod's own PyTorch image: CUDA, a C++ toolchain, and an `sshd` that takes
#: `PUBLIC_KEY`. The torch in it is replaced by the accelerator build of the
#: version the lock names, so the Pod measures the project's PyTorch.
IMAGE = "runpod/pytorch:1.3.0-rc.164-cu1290-torch2130-ubuntu2404"

#: The card the tables are quoted on, then what else will do. A comparison
#: measured on a different one is a different table: the README's hardware
#: line has to be rewritten by hand to match, which is why this is a list of
#: fallbacks and not a silent choice.
PREFERRED = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada",
]

#: The comparisons this rents a card for.
FOLDERS = ["46_gpt2"]


def _gpu(wanted: str) -> str:
    """The GPU type to ask for: the one named, else the best of what is in stock."""
    stock = _in_stock("NVIDIA")
    if wanted:
        if wanted not in stock:
            print(f"note: {wanted} is not listed as in stock; asking anyway")
        return wanted
    for candidate in PREFERRED:
        if candidate in stock:
            return candidate
    raise Failed(f"none of {', '.join(PREFERRED)} is in stock; name one with --gpu-id")


def _collect(pod: Pod, results: Path, folders: list[str], apply: bool) -> list[str]:
    """Bring the rewritten tables back, and put them in the checkout when asked."""
    pod.copy_from("/workspace/ppy-results", results.parent)
    written = []
    for folder in folders:
        came_back = results / folder / "README.md"
        if not came_back.is_file():
            print(f"{folder}: no README came back; see {results}")
            continue
        written.append(folder)
        if not apply:
            continue
        shutil.copy2(came_back, ROOT / "examples" / folder / "README.md")
        record = results / folder / "compare" / "measurements.json"
        if record.is_file():
            shutil.copy2(record, ROOT / "examples" / folder / "compare" / "measurements.json")
    return written


def measure(sha: str, folders: list[str], out: Path, wanted: str, keep: bool, apply: bool) -> int:
    """One Pod: build it, measure on it, take the tables home, delete it."""
    out.mkdir(parents=True, exist_ok=True)
    log = out / "bench.log"
    environment = {"image": IMAGE, "start": "", "count": 1, "mode": "cuda", "vendor": "NVIDIA"}
    stamp = dt.datetime.now(dt.UTC).strftime("%H%M%S")
    pod = Pod(f"ppy-bench-{stamp}", environment, _gpu(wanted), log)
    status = 1
    try:
        pod.create()
        pod.wait_ready()
        pod.copy_to(HERE / "remote_bench.sh", "/workspace/remote_bench.sh")
        # Each compiled counterpart warms up for minutes and is run five times;
        # four programs over one comparison is most of an hour.
        status = pod.run(
            f"bash /workspace/remote_bench.sh {sha} {' '.join(folders)}",
            log,
            timeout=4 * 60 * 60,
        )
        written = _collect(pod, out / "ppy-results", folders, apply)
        print(f"measured: {', '.join(written) or 'nothing'}; transcript in {log}")
    finally:
        if keep:
            print(f"left {pod.name} ({pod.id}) running as asked; delete it yourself")
        else:
            pod.delete()
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("folders", nargs="*", default=[], help=f"default: {' '.join(FOLDERS)}")
    parser.add_argument("--sha", default="", help="the commit to measure (pushed); HEAD by default")
    parser.add_argument("--gpu-id", default="", help=f"a GPU type; default: {PREFERRED[0]}")
    parser.add_argument("--results", default="", help="where to keep the logs and what came back")
    parser.add_argument("--keep", action="store_true", help="leave the Pod running (debugging)")
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="keep what came back under --results without touching the checkout",
    )
    options = parser.parse_args()
    sha = (
        options.sha
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    out = Path(options.results) if options.results else ROOT / "cloud-results" / f"bench-{stamp}"
    print(_authenticated())
    print(f"commit {sha}; results under {out}")
    try:
        return measure(
            sha,
            options.folders or FOLDERS,
            out,
            options.gpu_id,
            options.keep,
            not options.no_apply,
        )
    except Failed as error:
        print(f"stopped: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
