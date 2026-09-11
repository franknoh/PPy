"""The multi-accelerator trainer: two virtual CPU devices exercise its sharding here;
real GPUs are `scripts/cloud/runpod_matrix.py`'s business, and it says so when it has one."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "45_multi_gpu_jax"

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="jax is not installed"
)


def _run(arguments: list[str], devices: int | None) -> subprocess.CompletedProcess:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    if devices is not None:
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=EXAMPLE,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@requires_jax
def test_fewer_than_two_accelerators_is_reported_not_worked_around():
    done = _run(["train.ppy"], None)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "requires >= 2 physical accelerator devices; found 0"
    virtual = _run(["train.ppy"], 2)
    assert virtual.stdout.strip() == "requires >= 2 physical accelerator devices; found 0", (
        "a virtual CPU device is not an accelerator"
    )


@requires_jax
@pytest.mark.parametrize("runner", [[], ["-m", "ppy_compiler", "run"]])
def test_the_sharded_trainer_matches_a_single_device_on_virtual_devices(runner):
    done = _run([*runner, "train.ppy", "--allow-cpu-devices"], 2)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    assert lines[0] == "# devices: 2 x cpu"
    assert lines[-1] == "PASS", done.stdout
    first, last = (float(part) for part in lines[3].split("loss")[1].split("->"))
    assert first > last
