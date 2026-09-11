"""Validate the accelerator stack on real hardware, on Pods this script makes and deletes.

    python scripts/cloud/runpod_matrix.py --cuda          # one NVIDIA GPU
    python scripts/cloud/runpod_matrix.py --multigpu      # two or more NVIDIA GPUs in one Pod
    python scripts/cloud/runpod_matrix.py --rocm          # one AMD Instinct GPU
    python scripts/cloud/runpod_matrix.py --all

Not part of hosted CI: every run rents GPUs. For each environment it asks
`runpodctl` what is in stock, makes a Pod named `ppy-test-<env>-<id>` from
a vendor image, waits for SSH, copies `remote_test.sh` over, runs it against
the exact commit given (`--sha`, the checkout's HEAD by default -- which
must be pushed, since the Pod clones it), brings `/workspace/ppy-results`
back under `cloud-results/<stamp>/<env>/`, and deletes the Pod in a
`finally`, then confirms the deletion. Only Pods this run created are ever
touched; a Pod that existed before is never listed as ours. Credentials
stay where `runpodctl` keeps them: nothing here reads or prints the key,
and an unauthenticated CLI is reported and stops the run.

The summary at the end is one line per environment -- the GPU, its count,
`jax.default_backend()`, the devices, and PASS/FAIL per step -- and the
JSON beside it is what the final report is written from. A CPU-only JAX on
a GPU Pod is a FAIL, not a skip; a missing GPU type (AMD out of stock) is
NOT RUN, said in so many words.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

#: Environments: the vendor image, how many GPUs, the GPU types to try in order of
#: preference (cheap and common first), and what the remote script is told.
ENVIRONMENTS = {
    "cuda": {
        "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "count": 1,
        "mode": "cuda",
        "prefer": [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX A6000",
            "NVIDIA A40",
            "NVIDIA L40S",
            "NVIDIA L40",
            "NVIDIA RTX 6000 Ada Generation",
            "NVIDIA L4",
            "NVIDIA A100 80GB PCIe",
        ],
        "vendor": "NVIDIA",
    },
    "multigpu": {
        "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "count": 2,
        "mode": "multigpu",
        "prefer": [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX A6000",
            "NVIDIA A40",
            "NVIDIA L40S",
            "NVIDIA L40",
            "NVIDIA RTX 6000 Ada Generation",
            "NVIDIA A100 80GB PCIe",
        ],
        "vendor": "NVIDIA",
    },
    "rocm": {
        "image": "runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04",
        "count": 1,
        "mode": "rocm",
        "prefer": ["AMD Instinct MI300X OAM", "AMD Instinct MI250X", "AMD Instinct MI250"],
        "vendor": "AMD",
    },
    "rocm-multigpu": {
        "image": "runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04",
        "count": 2,
        "mode": "rocm",
        "prefer": ["AMD Instinct MI300X OAM", "AMD Instinct MI250X"],
        "vendor": "AMD",
    },
}


class Failed(Exception):
    """A step of the harness that cannot go on; the message is the reason."""


def _runpod(*arguments: str, timeout: int = 120) -> object:
    """`runpodctl ... -o json`, parsed; the CLI's own error is the exception's text."""
    done = subprocess.run(
        ["runpodctl", *arguments, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if done.returncode != 0:
        raise Failed(
            f"runpodctl {' '.join(arguments)}: {(done.stderr or done.stdout).strip()[:400]}"
        )
    text = done.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A command that answers prose: keep it, the caller decides.
        return text


def _authenticated() -> str:
    try:
        account = _runpod("user")
    except (Failed, FileNotFoundError) as error:
        raise Failed(f"runpodctl is not usable: {error}; run `runpodctl doctor` first") from error
    if not isinstance(account, dict) or "id" not in account:
        raise Failed("runpodctl is not authenticated; run `runpodctl doctor` (nothing is invented)")
    return f"account {account.get('id', '?')} balance ${account.get('clientBalance', 0):.2f}"


def _in_stock(vendor: str) -> dict[str, dict]:
    listed = _runpod("gpu", "list")
    if not isinstance(listed, list):
        raise Failed("runpodctl gpu list gave no list")
    return {
        entry["gpuId"]: entry
        for entry in listed
        if entry.get("available") and str(entry.get("gpuId", "")).upper().startswith(vendor)
    }


def _choose(environment: dict) -> str | None:
    stock = _in_stock(environment["vendor"])
    for wanted in environment["prefer"]:
        if wanted in stock:
            return wanted
    return next(iter(stock), None)


def _ssh_key() -> str:
    for candidate in ("id_ed25519.pub", "id_rsa.pub"):
        path = Path.home() / ".ssh" / candidate
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise Failed("no ~/.ssh/id_ed25519.pub or id_rsa.pub to give the Pod")


def _find(data: object, *names: str) -> object:
    """The first of `names` found at any depth of `data`."""
    if isinstance(data, dict):
        for name in names:
            if name in data and data[name] not in (None, "", [], {}):
                return data[name]
        for value in data.values():
            found = _find(value, *names)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find(value, *names)
            if found is not None:
                return found
    return None


class Pod:
    """One Pod this run made; nothing else is ever deleted through this class."""

    def __init__(self, name: str, environment: dict, gpu: str, log: Path) -> None:
        self.name = name
        self.environment = environment
        self.gpu = gpu
        self.log = log
        self.id: str | None = None
        self.ssh: tuple[str, int, str] | None = None

    def create(self) -> None:
        ends = (dt.datetime.now(dt.UTC) + dt.timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        created = _runpod(
            "pod",
            "create",
            "--name",
            self.name,
            "--image",
            self.environment["image"],
            "--gpu-id",
            self.gpu,
            "--gpu-count",
            str(self.environment["count"]),
            "--container-disk-in-gb",
            "60",
            "--ports",
            "22/tcp",
            "--env",
            json.dumps({"PUBLIC_KEY": _ssh_key()}),
            "--terminate-after",
            ends,
            timeout=300,
        )
        self.id = str(_find(created, "id", "podId") or "")
        if not self.id:
            raise Failed(f"pod create returned no id: {str(created)[:300]}")
        self._note(
            f"created {self.name} ({self.id}) on {self.gpu} x{self.environment['count']}, "
            f"terminates by itself at {ends}"
        )

    def _note(self, text: str) -> None:
        line = f"[{dt.datetime.now(dt.UTC).strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def wait_ready(self, minutes: int = 25) -> None:
        deadline = time.time() + minutes * 60
        while time.time() < deadline:
            details = _runpod("pod", "get", self.id or "")
            status = str(_find(details, "desiredStatus", "status") or "")
            ip = _find(details, "publicIp", "ip")
            ports = _find(details, "ports", "portMappings")
            self._note(f"status {status}, ip {ip}, ports {str(ports)[:120]}")
            if status.upper() == "RUNNING":
                try:
                    info = _runpod("ssh", "info", self.id or "")
                except Failed as error:
                    self._note(f"ssh info not ready: {error}")
                    info = None
                target = self._ssh_target(info, details)
                if target is not None:
                    self.ssh = target
                    if self._reachable():
                        return
            time.sleep(20)
        raise Failed(f"{self.name} did not become reachable in {minutes} minutes")

    def _ssh_target(self, info: object, details: object) -> tuple[str, int, str] | None:
        """(host, port, user) from `runpodctl ssh info`, else from the Pod's port map."""
        text = json.dumps(info) if info is not None else ""
        for command in (_find(info, "command", "sshCommand"), text):
            if isinstance(command, str) and "ssh " in command:
                words = shlex.split(command)
                host = next((w for w in words if "@" in w), None)
                if host is not None:
                    port = 22
                    if "-p" in words:
                        port = int(words[words.index("-p") + 1])
                    user, _, host = host.partition("@")
                    return host, port, user
        ip = _find(details, "publicIp")
        ports = _find(details, "ports", "portMappings")
        if isinstance(ip, str) and ip:
            port = 22
            if isinstance(ports, list):
                for mapping in ports:
                    if isinstance(mapping, dict) and str(mapping.get("privatePort")) == "22":
                        port = int(mapping.get("publicPort", 22))
            elif isinstance(ports, dict) and "22/tcp" in ports:
                port = int(ports["22/tcp"])
            return ip, port, "root"
        return None

    def _ssh_command(self) -> list[str]:
        assert self.ssh is not None
        host, port, user = self.ssh
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=30",
            "-p",
            str(port),
            f"{user}@{host}",
        ]

    def _reachable(self) -> bool:
        done = subprocess.run(
            [*self._ssh_command(), "echo", "ppy-ready"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if "ppy-ready" in done.stdout:
            self._note(f"ssh to {self.ssh[2]}@{self.ssh[0]}:{self.ssh[1]} answers")
            return True
        self._note(f"ssh not yet: {(done.stderr or done.stdout).strip()[:160]}")
        return False

    def copy_to(self, source: Path, target: str) -> None:
        host, port, user = self.ssh or ("", 22, "")
        done = subprocess.run(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-P",
                str(port),
                "-r",
                str(source),
                f"{user}@{host}:{target}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if done.returncode != 0:
            raise Failed(f"scp to the Pod failed: {done.stderr.strip()[:300]}")

    def copy_from(self, source: str, target: Path) -> None:
        host, port, user = self.ssh or ("", 22, "")
        target.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-P",
                str(port),
                "-r",
                f"{user}@{host}:{source}",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if done.returncode != 0:
            raise Failed(f"scp from the Pod failed: {done.stderr.strip()[:300]}")

    def run(self, command: str, log: Path, timeout: int) -> int:
        """`command` on the Pod, its output streamed to `log`; the exit status."""
        with log.open("a", encoding="utf-8") as handle:
            done = subprocess.run(
                [*self._ssh_command(), command],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        return done.returncode

    def delete(self) -> None:
        if self.id is None:
            return
        try:
            _runpod("pod", "delete", self.id)
            self._note(f"deleted {self.name} ({self.id})")
        except Failed as error:
            self._note(f"delete failed, retrying once: {error}")
            time.sleep(10)
            _runpod("pod", "delete", self.id)
            self._note(f"deleted {self.name} ({self.id}) on the second try")
        remaining = _runpod("pod", "list")
        ids = (
            {str(_find(entry, "id")) for entry in remaining}
            if isinstance(remaining, list)
            else set()
        )
        if self.id in ids:
            raise Failed(f"{self.name} ({self.id}) still listed after deletion; delete it by hand")
        self._note("confirmed: no Pod of this run remains")


def _steps(results: Path) -> dict[str, int]:
    steps = results / "steps.txt"
    if not steps.is_file():
        return {}
    found = {}
    for line in steps.read_text(encoding="utf-8").splitlines():
        name, _, status = line.rpartition(" ")
        if name:
            found[name] = int(status)
    return found


def validate(name: str, sha: str, out: Path, keep: bool) -> dict:
    environment = ENVIRONMENTS[name]
    summary: dict = {"environment": name, "commit": sha, "verdict": "NOT RUN", "steps": {}}
    out.mkdir(parents=True, exist_ok=True)
    log = out / "harness.log"
    gpu = _choose(environment)
    if gpu is None:
        summary["reason"] = (
            f"NOT RUN: no {environment['vendor']} GPU of these kinds in stock: "
            f"{environment['prefer']}"
        )
        print(summary["reason"])
        return summary
    pod = Pod(f"ppy-test-{name}-{uuid.uuid4().hex[:8]}", environment, gpu, log)
    summary["gpu"] = gpu
    summary["gpu_count"] = environment["count"]
    try:
        pod.create()
        summary["pod"] = pod.id
        pod.wait_ready()
        pod.copy_to(HERE / "remote_test.sh", "/workspace/remote_test.sh")
        status = pod.run(
            f"bash /workspace/remote_test.sh {sha} {environment['mode']} {environment['count']}",
            out / "remote.log",
            timeout=3 * 3600,
        )
        pod._note(f"remote script exit {status}")
        pod.copy_from("/workspace/ppy-results", out)
        results = out / "ppy-results"
        summary["steps"] = _steps(results)
        report = results / "accelerator.json"
        if report.is_file():
            summary["accelerator"] = json.loads(report.read_text(encoding="utf-8"))
        required = ["accelerator", "xla_example", "gpu_tests"]
        if environment["mode"] != "rocm":
            required += ["cuda_example", "tile_example"]
        if environment["count"] >= 2:
            required += ["multigpu_train", "multiprocess"]
        failed = [step for step in required if summary["steps"].get(step, 1) != 0]
        summary["failed"] = failed
        summary["verdict"] = "PASS" if not failed else "FAIL"
    except (Failed, subprocess.TimeoutExpired) as error:
        summary["verdict"] = "FAIL"
        summary["reason"] = str(error)
        pod._note(f"FAIL: {error}")
    finally:
        if keep:
            pod._note(f"--keep: {pod.name} ({pod.id}) is left running; delete it yourself")
        else:
            try:
                pod.delete()
            except Failed as error:
                summary["cleanup"] = str(error)
                pod._note(str(error))
    return summary


def _line(summary: dict) -> str:
    accelerator = summary.get("accelerator", {}).get("environment", {})
    devices = accelerator.get("devices", [])
    steps = " ".join(
        f"{name}:{'ok' if status == 0 else 'FAIL'}"
        for name, status in summary.get("steps", {}).items()
    )
    return (
        f"{summary['environment']:14} {summary.get('gpu', '-'):32} x{summary.get('gpu_count', 0)}  "
        f"backend={accelerator.get('default_backend', '-'):5} devices={len(devices)} "
        f"{summary['verdict']}  {summary.get('reason', '')} {steps}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    for name in ENVIRONMENTS:
        parser.add_argument(f"--{name}", action="store_true", help=f"run the {name} environment")
    parser.add_argument("--all", action="store_true", help="every environment")
    parser.add_argument("--sha", default="", help="the commit to test (pushed); HEAD by default")
    parser.add_argument(
        "--results", default="", help="where to keep the logs (cloud-results/<stamp>)"
    )
    parser.add_argument(
        "--keep", action="store_true", help="leave the Pods running (for debugging)"
    )
    options = parser.parse_args()
    chosen = [
        name for name in ENVIRONMENTS if options.all or getattr(options, name.replace("-", "_"))
    ]
    if not chosen:
        parser.error("choose an environment: --cuda, --multigpu, --rocm, --rocm-multigpu, or --all")
    sha = (
        options.sha
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    out = Path(options.results) if options.results else ROOT / "cloud-results" / stamp
    print(_authenticated())
    print(f"commit {sha}; results under {out}")
    summaries = []
    for name in chosen:
        print(f"\n### {name}")
        summaries.append(validate(name, sha, out / name, options.keep))
        (out / "summary.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    print("\nSUMMARY")
    for summary in summaries:
        print(_line(summary))
    return 0 if all(s["verdict"] == "PASS" for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
