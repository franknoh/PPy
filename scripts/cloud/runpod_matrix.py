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
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

#: AMD's own JAX image: ROCm, jaxlib, and the ROCm PJRT plugin built together
#: and known to load, one explicit tag (jax 0.11.0, the lock's minor). PPy is
#: installed beside it without touching the JAX in it; `PPY_ROCM_IMAGE`
#: names another tag of `rocm/jax` (or `rocm/jax-community`) for a try.
ROCM_IMAGE = os.environ.get("PPY_ROCM_IMAGE", "rocm/jax:rocm10.0-jax0.11.0-py3.12")

#: AMD's image is a plain Docker image: its command is a shell and nothing in it
#: listens on port 22, where RunPod's own images start `sshd` for the key in
#: `PUBLIC_KEY`. This start command does what theirs does -- installs the
#: server, admits the key, and keeps the container alive under `sshd` -- and
#: writes the container's environment (the `/opt/venv` on `PATH`, the ROCm
#: paths) where an SSH session reads it back, since a login starts without it.
#: No quotes inside: RunPod hands the string to the container's entrypoint whole.
ROCM_START = (
    "bash -c "
    '"apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server'
    " && mkdir -p /run/sshd /root/.ssh && echo $PUBLIC_KEY > /root/.ssh/authorized_keys"
    " && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys"
    " && echo PermitRootLogin yes >> /etc/ssh/sshd_config"
    " && echo PermitUserEnvironment yes >> /etc/ssh/sshd_config"
    " && env | grep -e ^PATH= -e ^ROCM -e ^HIP -e ^HSA -e ^LD_LIBRARY_PATH= -e ^VIRTUAL_ENV="
    " > /etc/environment && cp /etc/environment /root/.ssh/environment"
    ' && exec /usr/sbin/sshd -D"'
)

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
            "NVIDIA A40",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA L40S",
            "NVIDIA RTX A6000",
            "NVIDIA L40S",
            "NVIDIA L40",
            "NVIDIA RTX 6000 Ada Generation",
            "NVIDIA A100 80GB PCIe",
        ],
        "vendor": "NVIDIA",
    },
    "rocm": {
        "image": ROCM_IMAGE,
        "start": ROCM_START,
        "count": 1,
        "mode": "rocm",
        "prefer": ["AMD Instinct MI300X OAM", "AMD Instinct MI250X", "AMD Instinct MI250"],
        "vendor": "AMD",
    },
    "rocm-multigpu": {
        "image": ROCM_IMAGE,
        "start": ROCM_START,
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
        # The CLI echoes its usage after an API error; the error is the first line.
        said = (done.stderr or done.stdout).strip().splitlines()
        raise Failed(f"runpodctl {arguments[0]} {arguments[1]}: {said[0][:300] if said else '?'}")
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


def _in_datacenters(vendor: str) -> list[tuple[str, str]]:
    """(GPU type, datacenter) pairs the datacenter inventory names for `vendor`.

    `runpodctl gpu list` carries no AMD type at all some days while
    `runpodctl datacenter list` still names an MI300X somewhere, with no
    stock status; a listing is a place to ask, never an answer, and the
    create request decides.
    """
    listed = _runpod("datacenter", "list")
    if not isinstance(listed, list):
        return []
    found = []
    for datacenter in listed:
        for gpu in datacenter.get("gpuAvailability", []) or []:
            kind = str(gpu.get("gpuId", ""))
            if kind.upper().startswith(vendor):
                found.append((kind, str(datacenter.get("id", ""))))
    return found


def _candidates(environment: dict) -> list[tuple[str, str]]:
    """The (GPU type, datacenter) pairs to ask for, in order.

    The preferred types in stock first, then the rest in stock, each in any
    datacenter; where the stock list has none of the vendor, every pair the
    datacenter inventory names, preferred types first.
    """
    stock = _in_stock(environment["vendor"])
    preferred = [wanted for wanted in environment["prefer"] if wanted in stock]
    stocked = [(gpu, "") for gpu in preferred + [gpu for gpu in stock if gpu not in preferred]]
    if stocked:
        return stocked
    named = _in_datacenters(environment["vendor"])
    rank = {wanted: index for index, wanted in enumerate(environment["prefer"])}
    return sorted(named, key=lambda pair: (rank.get(pair[0], len(rank)), pair[1]))


def _ssh_key() -> tuple[str, Path]:
    """The public key the Pod is given through `PUBLIC_KEY`, and its private half."""
    for candidate in ("id_ed25519", "id_rsa"):
        private = Path.home() / ".ssh" / candidate
        public = private.with_suffix(".pub")
        if private.is_file() and public.is_file():
            return public.read_text(encoding="utf-8").strip(), private
    raise Failed("no ~/.ssh/id_ed25519 or id_rsa key pair to give the Pod")


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

    def __init__(
        self, name: str, environment: dict, gpu: str, log: Path, datacenter: str = ""
    ) -> None:
        self.name = name
        self.environment = environment
        self.gpu = gpu
        self.datacenter = datacenter
        self.log = log
        self.id: str | None = None
        self.ssh: tuple[str, int, str] | None = None
        #: The private keys to offer: the one whose public half went in through
        #: `PUBLIC_KEY` (accepted as soon as sshd is up), then the one `runpodctl
        #: doctor` registered (injected by the platform, a minute later).
        self.identities: list[str] = [str(_ssh_key()[1])]

    def create(self) -> None:
        ends = (dt.datetime.now(dt.UTC) + dt.timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        start = self.environment.get("start", "")
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
            json.dumps({"PUBLIC_KEY": _ssh_key()[0]}),
            "--terminate-after",
            ends,
            *(["--data-center-ids", self.datacenter] if self.datacenter else []),
            *(["--docker-args", start] if start else []),
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
        """(host, port, user) from `runpodctl ssh info` -- `ip`, `port`, and the key file it
        registered with the account -- else from the Pod's public address."""
        if isinstance(info, dict) and info.get("ip") and info.get("port"):
            key = info.get("ssh_key") or {}
            registered = (
                str(key.get("path", "")) if isinstance(key, dict) and key.get("exists") else ""
            )
            if registered and registered not in self.identities:
                self.identities.append(registered)
            return str(info["ip"]), int(info["port"]), "root"
        ip = _find(details, "publicIp", "ip")
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
            *self._identity_options(),
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
        said = (done.stderr or done.stdout).strip().splitlines()
        self._note(f"ssh not yet: {said[-1][:160] if said else 'no answer'}")
        return False

    def _identity_options(self) -> list[str]:
        """Only these keys, never a password prompt: a harness has no one to type one."""
        options = ["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "LogLevel=ERROR"]
        for identity in self.identities:
            options += ["-i", identity]
        return options

    def _scp_command(self, port: int) -> list[str]:
        return [
            "scp",
            *self._identity_options(),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            str(port),
            "-r",
        ]

    def copy_to(self, source: Path, target: str) -> None:
        host, port, user = self.ssh or ("", 22, "")
        done = subprocess.run(
            [
                *self._scp_command(port),
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
                *self._scp_command(port),
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


def validate(
    name: str,
    sha: str,
    out: Path,
    keep: bool,
    gpu_id: str = "",
    datacenter: str = "",
    setup_only: bool = False,
) -> dict:
    environment = ENVIRONMENTS[name]
    summary: dict = {"environment": name, "commit": sha, "verdict": "NOT RUN", "steps": {}}
    out.mkdir(parents=True, exist_ok=True)
    log = out / "harness.log"
    # `--gpu-id` names a type the stock list does not carry (an MI300X a datacenter
    # lists without a stock status); otherwise every type in stock is asked for in
    # turn, since stock moves between the listing and the request. The API's own
    # refusal is the reason when none can be had.
    candidates = [(gpu_id, datacenter)] if gpu_id else _candidates(environment)
    if not candidates:
        summary["reason"] = (
            f"NOT RUN: no {environment['vendor']} GPU in stock or in any datacenter's "
            f"inventory (wanted {environment['prefer']})"
        )
        print(summary["reason"])
        return summary
    pod: Pod | None = None
    refusals: list[str] = []
    for gpu, where in candidates:
        attempt = Pod(f"ppy-test-{name}-{uuid.uuid4().hex[:8]}", environment, gpu, log, where)
        try:
            attempt.create()
        except Failed as error:
            if "no longer any instances" in str(error) or "not available" in str(error):
                refusals.append(
                    f"{gpu} x{environment['count']}{f' in {where}' if where else ''}: {error}"
                )
                attempt._note(refusals[-1])
                continue
            summary["verdict"] = "FAIL"
            summary["reason"] = str(error)
            return summary
        pod = attempt
        break
    if pod is None:
        summary["reason"] = "NOT RUN: every request refused -- " + "; ".join(refusals)
        summary["refused"] = refusals
        print(summary["reason"])
        return summary
    summary["gpu"] = pod.gpu
    summary["gpu_count"] = environment["count"]
    try:
        summary["pod"] = pod.id
        pod.wait_ready()
        pod.copy_to(HERE / "remote_test.sh", "/workspace/remote_test.sh")
        mode = "setup" if setup_only else environment["mode"]
        status = pod.run(
            f"bash /workspace/remote_test.sh {sha} {mode} {environment['count']}",
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
        if environment["mode"] == "rocm":
            required += ["hip_emit"]
        else:
            required += ["cuda_example", "tile_example"]
        if environment["count"] >= 2:
            required += ["multigpu_train", "multiprocess"]
        steps = summary["steps"]
        # A step that hung on NCCL's peer-to-peer transport and passed with it
        # disabled counts as passed, and the report names the transport.
        failed = [
            step
            for step in required
            if steps.get(step, 1) != 0 and steps.get(f"{step}_p2p_off", 1) != 0
        ]
        summary["p2p_disabled"] = [s for s in steps if s.endswith("_p2p_off")]
        summary["failed"] = failed
        summary["verdict"] = "PASS" if not failed else "FAIL"
    except (Failed, subprocess.TimeoutExpired) as error:
        summary["verdict"] = "FAIL"
        summary["reason"] = str(error)
        pod._note(f"FAIL: {error}")
    finally:
        _cleanup(summary, pod, keep)
    return summary


def _cleanup(summary: dict, pod: Pod, keep: bool) -> None:
    """Delete the Pod and record it; a Pod that is still there fails the run.

    A test that passed on a GPU that keeps billing is not a pass. `--keep`
    (and `--setup-only`) leave the Pod on purpose and say so in the record.
    """
    if keep:
        pod._note(f"--keep: {pod.name} ({pod.id}) is left running; delete it yourself")
        summary["cleanup_ok"] = None
        summary["cleanup"] = f"kept on request: {pod.id}"
        return
    try:
        pod.delete()
    except Failed as error:
        summary["cleanup_ok"] = False
        summary["cleanup"] = str(error)
        summary["verdict"] = "FAIL"
        pod._note(f"FAIL: {error}")
        return
    summary["cleanup_ok"] = True


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
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="build the environment, then leave the Pod running (implies --keep) for a look",
    )
    parser.add_argument("--gpu-id", default="", help="a GPU type to ask for instead of the stock")
    parser.add_argument("--data-center", default="", help="a datacenter id to ask in (EU-RO-1)")
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
        summaries.append(
            validate(
                name,
                sha,
                out / name,
                options.keep or options.setup_only,
                options.gpu_id,
                options.data_center,
                setup_only=options.setup_only,
            )
        )
        (out / "summary.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    print("\nSUMMARY")
    for summary in summaries:
        print(_line(summary))
    return 0 if all(s["verdict"] == "PASS" for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
