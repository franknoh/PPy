"""Native object emission, linking, and launcher generation (spec 4.2, 16.3).

`ppy build` produces object code for every lowered function, links it into a
loadable shared library, and writes the binding manifest that describes each
exported symbol's ABI (spec 26.2). For a script target it additionally builds a
native launcher that embeds the target CPython interpreter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

from .lowering import NativeParam, NativeSignature

__all__ = [
    "MANIFEST_ABI_VERSION",
    "BuildArtifacts",
    "ToolchainError",
    "build_launcher",
    "emit_object",
    "link_shared_library",
    "toolchain_status",
    "write_manifest",
]

#: Version of the PPY Native Binding Manifest schema (spec 26.2).
MANIFEST_ABI_VERSION = 1

_C_COMPILERS = ("cc", "gcc", "clang")


class ToolchainError(RuntimeError):
    """No usable C toolchain or CPython development files were found."""


@dataclass(slots=True)
class BuildArtifacts:
    objects: list[Path] = field(default_factory=list)
    library: Path | None = None
    manifest: Path | None = None
    launcher: Path | None = None
    notes: list[str] = field(default_factory=list)
    #: Modules whose object came from the cache instead of the code generator.
    reused: list[str] = field(default_factory=list)


def _compiler() -> str | None:
    for name in _C_COMPILERS:
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def _python_library() -> tuple[Path, str] | None:
    """The directory and link name of the target interpreter's shared library."""
    directory = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDLIBRARY") or ""
    if not directory or not library.startswith("lib") or not library.endswith(".so"):
        return None
    path = Path(directory)
    if not (path / library).is_file():
        return None
    return path, library.removeprefix("lib").removesuffix(".so")


def toolchain_status() -> tuple[bool, str]:
    compiler = _compiler()
    if compiler is None:
        return False, "no C compiler (cc, gcc, or clang) is on PATH"
    include = Path(sysconfig.get_paths()["include"])
    if not (include / "Python.h").is_file():
        return False, f"CPython headers are missing from {include}"
    if _python_library() is None:
        return False, "no shared libpython to embed"
    return True, f"{compiler}, headers in {include}"


def emit_object(engine, ir: str, destination: Path) -> Path:  # type: ignore[no-untyped-def]
    """Compile one LLVM module to a relocatable object file."""
    from llvmlite import binding

    module = binding.parse_assembly(ir)
    module.verify()
    engine._optimize(module)
    machine = engine.target_machine or engine._machine()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(machine.emit_object(module))
    return destination


def link_shared_library(objects: list[Path], destination: Path) -> Path:
    """Link the emitted objects into one loadable shared library."""
    compiler = _compiler()
    if compiler is None:
        raise ToolchainError("no C compiler (cc, gcc, or clang) is on PATH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-shared", "-fPIC", "-o", str(destination), *[str(o) for o in objects]]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ToolchainError(f"link failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return destination


def write_manifest(
    destination: Path,
    entries: dict[str, NativeSignature],
    *,
    library: Path | None,
    fused: dict[str, tuple[int, int]] | None = None,
) -> Path:
    """Write the PPY Native Binding Manifest for the built symbols (spec 26.2)."""
    payload = {
        "abi_version": MANIFEST_ABI_VERSION,
        # By name: the library sits next to the manifest, and the pair must
        # survive being moved or shipped together.
        "native_library": library.name if library else None,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "calling_convention": "c",
        "entries": [
            {
                "python_qualname": signature.qualname,
                "native_symbol": signature.symbol,
                "calling_convention": "c",
                "arguments": [_argument(parameter) for parameter in signature.parameters],
                "returns": {
                    "semantic_type": _semantic(signature.ret),
                    "native_type": signature.ret,
                    "ownership": "value",
                    "passed_as": "out_parameter",
                },
                "status": {
                    "native_type": "i32",
                    "meaning": "0 = ok, non-zero = re-run the Python implementation",
                },
                "effects": ["may_raise"],
                "gil": "not_required",
                "thread_safe": True,
            }
            for signature in sorted(entries.values(), key=lambda s: s.qualname)
        ],
        "fused_kernels": [
            {"native_symbol": symbol, "arrays": arrays, "scalars": scalars}
            for symbol, (arrays, scalars) in sorted((fused or {}).items())
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _argument(parameter: NativeParam) -> dict[str, object]:
    if parameter.is_buffer:
        return {
            "name": parameter.name,
            "semantic_type": f"list[{parameter.element}]",
            "native_type": f"{parameter.abi[0]}, i64",
            "ownership": "borrowed",
            "note": "data pointer followed by an element count",
        }
    return {
        "name": parameter.name,
        "semantic_type": parameter.kind,
        "native_type": parameter.abi[0],
        "ownership": "value",
    }


def _semantic(abi: str) -> str:
    return {"i64": "int", "double": "float", "i8": "bool"}.get(abi, abi)


_LAUNCHER = """/* generated by ppy; do not edit. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

/* The launcher embeds the target interpreter so that imports, Python objects,
 * callbacks, and unsupported library calls keep working (spec 16.3). */
int main(int argc, char **argv)
{{
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.isolated = 0;
    /* argv belongs to the PPY program, not to the interpreter, so it must not
     * be parsed as a Python command line. */
    config.parse_argv = 0;

    status = PyConfig_SetBytesArgv(&config, argc, argv);
    if (PyStatus_Exception(status)) {{
        PyConfig_Clear(&config);
        Py_ExitStatusException(status);
    }}
    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {{
        Py_ExitStatusException(status);
    }}

    int code = PyRun_SimpleString({bootstrap});
    if (Py_FinalizeEx() < 0) {{
        return 120;
    }}
    return code == 0 ? 0 : 1;
}}
"""

#: The launcher is `ppy run` in a compiled coat: it enters the same CLI, the
#: same pipeline, and the same guarded bindings -- only the machine code is
#: taken from the library built next to it instead of being JIT-compiled.
_BOOTSTRAP = """import sys
for extra in {paths!r}:
    if extra not in sys.path:
        sys.path.append(extra)
sys.path.insert(0, {search!r})
from ppy_compiler.driver.cli import main
sys.exit(main([
    "run",
    "--prebuilt", {manifest!r},
    "--safeguards", {safeguards!r},
    {entry!r},
    "--", *sys.argv[1:],
]))
"""


def _c_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def build_launcher(
    entry: Path, destination: Path, search_paths: list[Path], manifest: Path, safeguards: str
) -> Path:
    """Compile a native launcher that embeds CPython and runs the entry point."""
    compiler = _compiler()
    if compiler is None:
        raise ToolchainError("no C compiler (cc, gcc, or clang) is on PATH")
    library = _python_library()
    if library is None:
        raise ToolchainError("no shared libpython to embed")
    library_dir, library_name = library

    include = sysconfig.get_paths()["include"]
    if not Path(include, "Python.h").is_file():
        raise ToolchainError(f"CPython headers are missing from {include}")

    search = str(search_paths[0]) if search_paths else str(entry.parent)
    # The launcher is pinned to the interpreter it was built against, so the
    # build-time import path is the right one to record (spec 16.5).
    bootstrap = _BOOTSTRAP.format(
        search=search,
        entry=str(entry),
        manifest=str(manifest),
        safeguards=safeguards,
        paths=[p for p in sys.path if p],
    )
    source = _LAUNCHER.format(bootstrap=_c_string(bootstrap))

    destination.parent.mkdir(parents=True, exist_ok=True)
    csource = destination.with_suffix(".c")
    csource.write_text(source, encoding="utf-8")

    command = [
        compiler,
        str(csource),
        "-I",
        include,
        "-o",
        str(destination),
        f"-L{library_dir}",
        f"-l{library_name}",
        f"-Wl,-rpath,{library_dir}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ToolchainError(
            f"launcher build failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    os.chmod(destination, 0o755)
    return destination
