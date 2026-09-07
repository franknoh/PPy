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

from ppy_runtime.manifest import SUPPORTED_ABI as MANIFEST_ABI_VERSION

from .lowering import NativeParam, NativeSignature

__all__ = [
    "MANIFEST_ABI_VERSION",
    "BuildArtifacts",
    "ToolchainError",
    "build_launcher",
    "c_compiler",
    "c_prototype",
    "emit_object",
    "link_shared_library",
    "standalone_toolchain_status",
    "toolchain_status",
    "write_header",
    "write_manifest",
]

#: Version of the PPY Native Binding Manifest schema (spec 26.2).

_C_COMPILERS = ("cc", "gcc", "clang")


class ToolchainError(RuntimeError):
    """No usable C toolchain or CPython development files were found."""


@dataclass(slots=True)
class BuildArtifacts:
    objects: list[Path] = field(default_factory=list)
    header: Path | None = None
    library: Path | None = None
    manifest: Path | None = None
    launcher: Path | None = None
    notes: list[str] = field(default_factory=list)
    #: Modules whose object came from the cache instead of the code generator.
    reused: list[str] = field(default_factory=list)
    #: The generated Python per module, for a build that packages it further.
    generated: dict[str, object] = field(default_factory=dict)
    #: What the manifest lists: every exposed signature by qualname, the
    #: public C symbols by name, and the libraries the bindings need.
    signatures: dict[str, NativeSignature] = field(default_factory=dict)
    exports: dict[str, NativeSignature] = field(default_factory=dict)
    libraries: tuple[str, ...] = ()
    #: `--python-extension`: the importable module; `--library`: the package.
    extension: Path | None = None
    package: Path | None = None


def _compiler() -> str | None:
    for name in _C_COMPILERS:
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def c_compiler() -> str | None:
    """The C compiler PPY itself will use, or None when there is none."""
    return _compiler()


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
    """What a hybrid build needs: a C compiler, the headers, and a libpython."""
    usable, detail = standalone_toolchain_status()
    if not usable:
        return usable, detail
    include = Path(sysconfig.get_paths()["include"])
    if not (include / "Python.h").is_file():
        return False, f"CPython headers are missing from {include}"
    if _python_library() is None:
        return False, "no shared libpython to embed"
    return True, f"{detail}, headers in {include}"


def standalone_toolchain_status() -> tuple[bool, str]:
    """What a standalone build needs, which is only a C compiler.

    `--standalone` links no interpreter, so an installation whose CPython is
    static or header-less still builds one. Holding it to the hybrid path's
    requirements would refuse the builds it exists to make.
    """
    compiler = _compiler()
    if compiler is None:
        return False, "no C compiler (cc, gcc, or clang) is on PATH"
    return True, str(compiler)


def emit_object(  # type: ignore[no-untyped-def]
    engine, ir: str, destination: Path, *, host_cpu: bool = False, target=None
) -> Path:
    """Compile one LLVM module to a relocatable object file.

    An object goes into an artifact that may run on another machine, so both
    the pipeline and the code emitter stay on the portable baseline even when
    the engine itself is tuned for this host. `host_cpu` is the opt-in that
    trades that portability for this machine's instruction set. A `target`
    other than the host retargets the module -- its triple and data layout
    -- and emits through that target's machine.
    """
    from llvmlite import binding

    module = binding.parse_assembly(ir)
    module.verify()
    if target is not None and not target.is_host:
        module.triple = target.triple
        if target.data_layout:
            module.data_layout = target.data_layout
    machine = engine.object_machine(host_cpu, target)
    engine._optimize(module, machine)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(machine.emit_object(module))
    return destination


def link_shared_library(
    objects: list[Path], destination: Path, libraries: tuple[str, ...] = (), target=None
) -> Path:
    """Link the emitted objects into one loadable shared library.

    `libraries` are the names the program's C bindings gave (`"m"`); each
    is linked by name so the dynamic loader finds it where the system has it.
    A `target` other than the host needs a toolchain for it: the
    `<triple>-gcc` of a cross toolchain, or clang with `--target`.
    """
    command = [*_linker(target), "-shared", "-fPIC", "-o", str(destination)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    command.extend(str(o) for o in objects)
    command.extend(f"-l{library}" for library in libraries)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ToolchainError(f"link failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return destination


def _linker(target) -> list[str]:  # type: ignore[no-untyped-def]
    """The command that links for `target`: the host's compiler, or a cross one."""
    if target is None or target.is_host:
        compiler = _compiler()
        if compiler is None:
            raise ToolchainError("no C compiler (cc, gcc, or clang) is on PATH")
        return [compiler]
    for name in (f"{target.triple}-gcc", f"{target.triple}-cc", f"{target.triple}-clang"):
        found = shutil.which(name)
        if found is not None:
            return [found]
    clang = shutil.which("clang")
    if clang is not None:
        return [clang, *target.link_flags]
    raise ToolchainError(
        f"no toolchain links for {target.triple}: install `{target.triple}-gcc` or clang"
    )


def write_pkg_config(
    destination: Path, name: str, version: str, prefix: str = "${pcfiledir}/../.."
) -> Path:
    """A pkg-config file for a packaged library, relative to where it sits."""
    lines = [
        f"prefix={prefix}",
        "libdir=${prefix}/lib",
        "includedir=${prefix}/include",
        "",
        f"Name: {name}",
        f"Description: {name}, built by ppy",
        f"Version: {version}",
        f"Libs: -L${{libdir}} -lppy_{name}",
        "Cflags: -I${includedir}",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


_C_TYPES = {"i64": "int64_t", "double": "double", "i8": "int8_t"}


def c_prototype(name: str, signature: NativeSignature) -> str:
    """The C declaration of an exported function."""
    parameters: list[str] = []
    for parameter in signature.parameters:
        if parameter.is_buffer:
            element = _C_TYPES[_abi_of(parameter.element)]
            parameters.append(f"{element} *{parameter.name}")
            parameters.append(f"int64_t {parameter.name}_len")
        elif parameter.is_pointer:
            const = "const " if parameter.kind == "const_ptr" else ""
            parameters.append(f"{const}{_C_TYPES[_abi_of(parameter.element)]} *{parameter.name}")
        elif parameter.is_tuple or parameter.is_object:
            for index, atom in enumerate(parameter.abi):
                parameters.append(f"{_C_TYPES[atom]} {parameter.name}_{index}")
        else:
            parameters.append(f"{_C_TYPES[parameter.abi[0]]} {parameter.name}")
    result = _C_TYPES[signature.returns[0]] if signature.returns else "void"
    return f"{result} {name}({', '.join(parameters) or 'void'});"


def _abi_of(scalar: str) -> str:
    return {"int": "i64", "float": "double", "bool": "i8", "i8": "i8", "u8": "i8"}[scalar]


def write_header(destination: Path, exports: dict[str, NativeSignature]) -> Path:
    """A C header declaring every exported symbol of the library."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(header_text(destination.stem, exports), encoding="utf-8")
    return destination


def header_text(stem: str, exports: dict[str, NativeSignature]) -> str:
    """The C header declaring `exports`, guarded by a name made from `stem`."""
    guard = "PPY_" + "".join(c if c.isalnum() else "_" for c in stem).upper() + "_H"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]
    lines.extend(c_prototype(name, signature) for name, signature in sorted(exports.items()))
    lines.extend(["", "#ifdef __cplusplus", "}", "#endif", "", f"#endif /* {guard} */", ""])
    return "\n".join(lines)


def write_manifest(
    destination: Path,
    entries: dict[str, NativeSignature],
    *,
    library: Path | None,
    fused: dict[str, tuple[int, int]] | None = None,
    program: dict | None = None,
    wrappers: dict | None = None,
    regions: dict | None = None,
    exports: dict[str, str] | None = None,
    libraries: tuple[str, ...] = (),
    target: str = "",
) -> Path:
    """Write the PPY Native Binding Manifest for the built symbols (spec 26.2)."""
    payload = {
        "abi_version": MANIFEST_ABI_VERSION,
        # The triple the objects were compiled for; a runtime on another
        # machine refuses the artifact instead of loading it.
        "target": target,
        "program": program,
        "wrappers": wrappers,
        # Public C symbols and their C signatures, and the shared libraries
        # the artifact's own bindings need at load time.
        "exports": exports or {},
        "libraries": list(libraries),
        # Compiled ATen regions, per generated module: the extension library
        # beside the manifest and the C++ symbol of each region.
        "regions": regions,
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
                "gil": "not_required" if signature.releases_gil else "required",
                "thread_safe": True,
                "module": _owning_module(signature.qualname, program),
                "binding": _binding_of(signature.qualname, program),
                "abi": {
                    "parameters": [
                        {
                            "name": parameter.name,
                            "kind": parameter.kind,
                            "element": parameter.element,
                            "elements": list(parameter.elements),
                            "fields": [list(pair) for pair in parameter.fields],
                            "class_name": parameter.class_name,
                        }
                        for parameter in signature.parameters
                    ],
                    "returns": list(signature.returns),
                    "releases_gil": signature.releases_gil,
                },
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


def _owning_module(qualname: str, program: dict | None) -> str:
    """The generated module a binding belongs to, by longest name prefix."""
    modules = (program or {}).get("modules", ())
    best = ""
    for module in modules:
        if qualname.startswith(module + ".") and len(module) > len(best):
            best = module
    return best or qualname.rpartition(".")[0]


def _binding_of(qualname: str, program: dict | None) -> str:
    module = _owning_module(qualname, program)
    return qualname[len(module) + 1 :] if qualname.startswith(module + ".") else qualname


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

#: The launcher is compiled software: it loads the runtime, the manifest,
#: and the native library, and executes. The compiler is not imported -- all
#: analysis happened at build time, and uninstalling `ppy_compiler` must not
#: break a built application.
_BOOTSTRAP = """import sys
from pathlib import Path
for extra in {paths!r}:
    if extra not in sys.path:
        sys.path.append(extra)
sys.path.insert(0, {search!r})
from ppy_runtime.launch import main
sys.exit(main(Path({manifest!r}), sys.argv[1:]))
"""


def _c_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def build_launcher(
    entry: Path, destination: Path, search_paths: list[Path], manifest: Path
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
        manifest=str(manifest),
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
