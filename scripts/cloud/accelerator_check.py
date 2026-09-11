"""What this machine's JAX actually runs on, and proof that it computes there.

    python scripts/cloud/accelerator_check.py --require gpu --min-devices 2 --out report.json

Prints the environment -- interpreter, JAX, jaxlib, the accelerator plugin
packages, the driver and toolkit versions the vendor tools report,
`jax.default_backend()`, every device -- runs a matrix multiplication and
a reduction under `jax.jit` on the default device against a NumPy
reference, and with `--min-devices 2` shards a computation across the
devices and reduces across them. A `--require gpu` run fails, with the
report written, when every visible device is a CPU: a CPU is not a pass
for a GPU test, and an import that succeeded proves nothing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time

PLUGIN_PREFIXES = (
    "jax-cuda",
    "jax_cuda",
    "jax-rocm",
    "jax_rocm",
    "jax-plugin",
    "jax_plugin",
    "nvidia-",
    "rocm",
)


def _tool(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return f"{command[0]}: not installed"
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"{command[0]}: {error}"
    return (done.stdout or done.stderr).strip()


def _packages() -> dict[str, str]:
    found = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] or ""
        lowered = name.lower()
        if lowered in {"jax", "jaxlib"} or lowered.startswith(PLUGIN_PREFIXES):
            found[name] = dist.version
    return dict(sorted(found.items()))


def environment() -> dict:
    import jax

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _packages(),
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "local_devices": [str(device) for device in jax.local_devices()],
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "device_kinds": sorted({device.device_kind for device in jax.devices()}),
        "platforms": sorted({device.platform for device in jax.devices()}),
        "nvidia-smi": _tool(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "nvidia-smi -L": _tool(["nvidia-smi", "-L"]),
        "nvcc": _tool(["nvcc", "--version"]).splitlines()[-1]
        if shutil.which("nvcc")
        else "nvcc: not installed",
        "rocm-smi": _tool(["rocm-smi", "--showproductname"]),
        "rocminfo": "\n".join(
            line
            for line in _tool(["rocminfo"]).splitlines()
            if "Marketing Name" in line or "gfx" in line
        )[:2000],
    }
    listed = _tool(["nvidia-smi", "--query", "--display=COMPUTE"]).splitlines()
    report["cuda_runtime"] = listed[0] if listed else ""
    return report


def compute(min_devices: int) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    results: dict = {}
    # Full float32 precision: on Ampere and later a float32 matmul is TF32 by default,
    # which is a different answer from NumPy's, not a wrong device.
    jax.config.update("jax_default_matmul_precision", "highest")
    rng = np.random.default_rng(7)
    a = rng.standard_normal((1024, 512)).astype(np.float32)
    b = rng.standard_normal((512, 256)).astype(np.float32)

    @jax.jit
    def kernel(x, y):
        product = jnp.dot(x, y)
        return product, jnp.sum(product * product)

    product, total = kernel(a, b)
    product.block_until_ready()
    started = time.perf_counter()
    for _ in range(10):
        product, total = kernel(a, b)
    product.block_until_ready()
    results["jit_ms_per_call"] = (time.perf_counter() - started) * 100.0
    reference = a @ b
    results["matmul_max_abs_error"] = float(np.max(np.abs(np.asarray(product) - reference)))
    results["matmul_rel_error"] = results["matmul_max_abs_error"] / float(np.max(np.abs(reference)))
    results["reduction_rel_error"] = float(
        abs(float(total) - float(np.sum(reference * reference)))
        / float(np.sum(reference * reference))
    )
    results["ran_on"] = sorted({str(d) for d in product.devices()})
    results["jit_ok"] = results["matmul_rel_error"] < 1e-4 and results["reduction_rel_error"] < 1e-4
    devices = jax.devices()
    if len(devices) >= min_devices >= 2:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        mesh = Mesh(devices, ("rows",))
        rows = NamedSharding(mesh, PartitionSpec("rows"))
        big = rng.standard_normal((len(devices) * 512, 512)).astype(np.float32)
        sharded = jax.device_put(big, rows)
        results["shards_on"] = sorted({str(d) for d in sharded.devices()})

        @jax.jit
        def across(x):
            return jnp.sum(jnp.dot(x, x.T[:, :128]) ** 2)

        value = across(sharded)
        value.block_until_ready()
        expected = float(np.sum((big @ big.T[:, :128]) ** 2))
        results["collective_rel_error"] = abs(float(value) - expected) / expected
        results["collective_ok"] = (
            results["collective_rel_error"] < 1e-3 and len(results["shards_on"]) >= min_devices
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--require", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--min-devices", type=int, default=1)
    parser.add_argument("--out", default="")
    options = parser.parse_args()
    report: dict = {"require": options.require, "min_devices": options.min_devices}
    verdict = "PASS"
    reasons: list[str] = []
    try:
        report["environment"] = environment()
        env = report["environment"]
        accelerators = [d for d in env["devices"] if not d.lower().startswith("cpu")]
        if options.require == "gpu":
            if not accelerators:
                reasons.append(f"every visible device is a CPU: {env['devices']}")
            if env["default_backend"] == "cpu":
                reasons.append("jax.default_backend() is cpu")
        if env["local_device_count"] < options.min_devices or (
            options.require == "gpu" and len(accelerators) < options.min_devices
        ):
            reasons.append(
                f"requires >= {options.min_devices} physical accelerator devices, "
                f"found {len(accelerators)}"
            )
        if not reasons:
            report["compute"] = compute(options.min_devices)
            if not report["compute"]["jit_ok"]:
                reasons.append("the jitted matmul or reduction disagrees with NumPy")
            if options.min_devices >= 2 and not report["compute"].get("collective_ok"):
                reasons.append(
                    "the sharded computation across devices failed or was not on every device"
                )
    except Exception as error:  # noqa: BLE001 - the report must be written whatever failed
        reasons.append(f"{type(error).__name__}: {error}")
    if reasons:
        verdict = "FAIL"
    report["verdict"] = verdict
    report["reasons"] = reasons
    text = json.dumps(report, indent=1)
    if options.out:
        with open(options.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    print(f"accelerator check: {verdict}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
