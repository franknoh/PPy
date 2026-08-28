"""Build-time JAX export in an isolated process (spec 21.3, 21.8, 31.3).

Exporting a staged function runs user code, so it happens only with explicit
permission, in a separate process with a controlled environment, an explicit
import path, a time limit, and captured output. The serialized StableHLO is
cached under a fingerprint of every input that could change it.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "EXPORT_TIMEOUT",
    "ExportRequest",
    "ExportResult",
    "accelerator_fingerprint",
    "export_function",
    "runtime_call",
]

#: Build-time execution must not run unbounded (spec 31.3).
EXPORT_TIMEOUT = 120


@dataclass(frozen=True, slots=True)
class ExportRequest:
    module_path: Path
    module_name: str
    function: str
    shapes: tuple[tuple[int | str, ...], ...]
    dtypes: tuple[str, ...]
    search_paths: tuple[Path, ...] = ()

    def fingerprint_input(self) -> tuple:
        return (
            self.module_name,
            self.function,
            self.shapes,
            self.dtypes,
        )


@dataclass(slots=True)
class ExportResult:
    ok: bool
    payload: bytes = b""
    signature: str = ""
    platforms: tuple[str, ...] = ()
    versions: dict[str, str] = field(default_factory=dict)
    reason: str = ""


_WORKER = r"""
import base64, importlib.util, json, sys

def fail(reason):
    print(json.dumps({"ok": False, "reason": reason}))
    raise SystemExit(0)

try:
    import jax
    from jax import export as jax_export
    import jaxlib
except Exception as exc:  # noqa: BLE001
    fail(f"jax is not importable: {exc}")

request = json.loads(sys.argv[1])
for entry in request["search_paths"]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

from importlib.machinery import SourceFileLoader

# A `.ppy` file is ordinary Python source, but its extension is not one the
# import machinery infers a loader from, so the loader is named explicitly.
loader = SourceFileLoader(request["module_name"], request["module_path"])
spec = importlib.util.spec_from_file_location(
    request["module_name"], request["module_path"], loader=loader
)
if spec is None or spec.loader is None:
    fail("the module could not be loaded")
module = importlib.util.module_from_spec(spec)
sys.modules[request["module_name"]] = module
try:
    spec.loader.exec_module(module)
except Exception as exc:  # noqa: BLE001
    fail(f"importing the module raised {type(exc).__name__}: {exc}")

target = getattr(module, request["function"], None)
if target is None:
    fail(f"`{request['function']}` is not defined in the module")

import numpy

# Every symbolic dimension in one export must share a scope, so it is
# created once here rather than per parameter.
scope = jax_export.SymbolicScope() if hasattr(jax_export, "SymbolicScope") else None
specs = []
try:
    for shape, dtype in zip(request["shapes"], request["dtypes"]):
        if any(isinstance(dim, str) for dim in shape):
            text = ", ".join(str(dim) for dim in shape)
            resolved = (
                jax_export.symbolic_shape(text, scope=scope)
                if scope is not None
                else jax_export.symbolic_shape(text)
            )
        else:
            resolved = tuple(shape)
        specs.append(jax.ShapeDtypeStruct(tuple(resolved), numpy.dtype(dtype)))
except Exception as exc:  # noqa: BLE001
    fail(f"the declared shape or dtype is unusable: {exc}")

try:
    staged = target if isinstance(target, jax.stages.Wrapped) else jax.jit(target)
    if not hasattr(jax_export, "export"):
        fail("the installed jax has no `jax.export.export`")
    # The default backend is what the program will run on, and it is also the
    # only name that matches the check `Exported.call` makes: `jax.devices()`
    # reports the legacy alias "gpu" where the artifact records "cuda".
    exported = jax_export.export(staged)(*specs)
    payload = exported.serialize()
except Exception as exc:  # noqa: BLE001
    fail(f"export failed: {type(exc).__name__}: {exc}")

print(json.dumps({
    "ok": True,
    "payload": base64.b64encode(payload).decode("ascii"),
    "signature": str(exported.in_avals) + " -> " + str(exported.out_avals),
    "platforms": list(exported.platforms),
    "versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
}))
"""


def accelerator_fingerprint() -> str:
    """A cheap description of the accelerator an export would target.

    An exported artifact records the platform it was built for and refuses to
    run elsewhere, so the cache key has to change when the accelerator does --
    without importing JAX here, which would seize device memory.
    """
    import shutil

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    has_nvidia = bool(shutil.which("nvidia-smi")) or Path("/dev/nvidiactl").exists()
    return f"nvidia={int(has_nvidia)}:visible={visible}:platforms={platforms}"


def export_function(request: ExportRequest, *, timeout: int = EXPORT_TIMEOUT) -> ExportResult:
    """Stage and export one function, in a process of its own."""
    payload = json.dumps(
        {
            "module_path": str(request.module_path),
            "module_name": request.module_name,
            "function": request.function,
            "shapes": [list(shape) for shape in request.shapes],
            "dtypes": list(request.dtypes),
            "search_paths": [str(p) for p in request.search_paths],
        }
    )
    # A controlled environment (spec 31.3): only the variables the export
    # genuinely needs are forwarded, and only when they are actually set --
    # an empty `CUDA_VISIBLE_DEVICES` would hide the accelerator entirely.
    environment = {
        name: os.environ[name]
        for name in (
            "PATH",
            "HOME",
            "LD_LIBRARY_PATH",
            "JAX_PLATFORMS",
            "XLA_FLAGS",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
        )
        if os.environ.get(name)
    }
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            # A build-time export must not seize the accelerator's memory pool.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER, payload],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return ExportResult(False, reason=f"export exceeded the {timeout}s build-time limit")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return ExportResult(False, reason=detail[-1] if detail else "the export process failed")

    line = next((x for x in reversed(completed.stdout.splitlines()) if x.startswith("{")), "")
    try:
        answer = json.loads(line)
    except json.JSONDecodeError:
        return ExportResult(False, reason="the export process produced no result")
    if not answer.get("ok"):
        return ExportResult(False, reason=answer.get("reason", "export failed"))
    return ExportResult(
        ok=True,
        payload=base64.b64decode(answer["payload"]),
        signature=answer.get("signature", ""),
        platforms=tuple(answer.get("platforms", ())),
        versions=answer.get("versions", {}),
    )


def runtime_call(payload: bytes):  # type: ignore[no-untyped-def]
    """Rehydrate an exported computation for execution.

    The exported artifact carries its own StableHLO and calling metadata, so
    the staged graph is never rebuilt from Python source at run time.
    """
    from jax import export as jax_export

    exported = jax_export.deserialize(bytearray(payload))
    return exported.call
