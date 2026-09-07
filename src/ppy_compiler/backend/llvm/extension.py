"""`ppy build foo.ppy --python-extension`: one importable CPython module.

The extension is the generated boundary wrapper, the module's native
objects, and the module's own optimized Python, in one shared library:
`import foo` runs that Python -- so every class, constant, and function
the module defines exists -- and, as each native-eligible function is
defined, binds it to its compiled code through the same hook the launcher
uses, holding the Python definition as the fallback a refused guard runs.
Nothing is bound by name at runtime and no manifest is read: the symbols
are linked in.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sysconfig
from pathlib import Path

from ppy_runtime.generated import BINDER_NAME, _insert_definition_bindings, _restore_lines

from ...target import TargetInfo, host_target
from .link import ToolchainError, _compiler
from .lowering import NativeSignature
from .wrapper import _HEADER, C_TYPES, _function

__all__ = ["build_python_extension", "extension_source"]

_INIT = """
typedef struct {{
    const char *key;
    int index;
}} ppy_entry;

static const ppy_entry ppy_entries[] = {{
{entries}
    {{NULL, -1}}
}};

static PyMethodDef ppy_calls[] = {{
{calls}
    {{NULL, NULL, 0, NULL}}
}};

{adopters}

/* `__ppy_bind_native__(key, function)`: the generated module calls this
 * right after defining a native-eligible function. The definition becomes
 * the fallback, and the caller gets the compiled entry in its place. */
static PyObject *ppy_bind_native(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {{
    if (nargs != 2 || !PyUnicode_Check(args[0])) {{
        PyErr_SetString(PyExc_TypeError, "__ppy_bind_native__(key, function)");
        return NULL;
    }}
    const char *key = PyUnicode_AsUTF8(args[0]);
    if (key == NULL) {{
        return NULL;
    }}
    for (const ppy_entry *entry = ppy_entries; entry->key != NULL; entry++) {{
        if (strcmp(entry->key, key) == 0) {{
            PyObject *bound = ppy_adopters[entry->index](args[1]);
            if (bound == NULL) {{
                return NULL;
            }}
            return bound;
        }}
    }}
    Py_INCREF(args[1]);
    return args[1];
}}

static PyMethodDef ppy_binder_def = {{
    "__ppy_bind_native__", (PyCFunction)(void *)ppy_bind_native, METH_FASTCALL, NULL
}};

static const char ppy_source[] = {source};

static struct PyModuleDef ppy_module = {{
    PyModuleDef_HEAD_INIT, "{name}", NULL, -1, NULL, NULL, NULL, NULL, NULL
}};

PyMODINIT_FUNC PyInit_{name}(void) {{
    PyObject *module = PyModule_Create(&ppy_module);
    if (module == NULL) {{
        return NULL;
    }}
    PyObject *dict = PyModule_GetDict(module);
    PyObject *binder = PyCFunction_NewEx(&ppy_binder_def, NULL, NULL);
    if (binder == NULL || PyDict_SetItemString(dict, "__ppy_bind_native__", binder) != 0) {{
        Py_XDECREF(binder);
        Py_DECREF(module);
        return NULL;
    }}
    Py_DECREF(binder);
    if (PyDict_SetItemString(dict, "__file__", PyUnicode_FromString({file})) != 0) {{
        Py_DECREF(module);
        return NULL;
    }}
    PyObject *ran = PyRun_String(ppy_source, Py_file_input, dict, dict);
    if (ran == NULL) {{
        Py_DECREF(module);
        return NULL;
    }}
    Py_DECREF(ran);
    PyDict_DelItemString(dict, "__ppy_bind_native__");
    return module;
}}
"""

_ADOPTER = """
static PyObject *ppy_adopt_{index}(PyObject *function) {{
    PyObject *types = NULL;
{lookup}
    Py_XINCREF(function);
    Py_XDECREF(ppy_fallback_{index});
    ppy_fallback_{index} = function;
    Py_XDECREF(ppy_types_{index});
    ppy_types_{index} = types;
    ppy_target_{index} = (ppy_fn_{index})&{symbol};
{assignments}    ppy_spec_count_{index} = 0;
    return PyCFunction_NewEx(&ppy_calls[{position}], NULL, NULL);
}}
"""


def _lookup(index: int, signature: NativeSignature) -> str:
    """Resolve each value class the function's parameters name, from the
    function's own globals: the class the definition sees is the class the
    wrapper guards on."""
    object_params = [p for p in signature.parameters if p.is_object]
    if not object_params:
        return ""
    globals_line = "PyFunction_Check(function) ? PyFunction_GetGlobals(function) : NULL;"
    lines = [
        f"    PyObject *globals = {globals_line}",
        f"    types = PyTuple_New({len(object_params)});",
        "    if (types == NULL) return NULL;",
    ]
    for position, parameter in enumerate(object_params):
        class_name = parameter.class_name.rpartition(".")[2]
        lines.append(
            f"    PyObject *cls{position} = "
            f'globals ? PyDict_GetItemString(globals, "{class_name}") : NULL;'
        )
        lines.append(f"    if (cls{position} == NULL || !PyType_Check(cls{position})) {{")
        lines.append("        Py_DECREF(types);")
        lines.append(
            f'        PyErr_SetString(PyExc_ImportError, "{signature.qualname}: value class '
            f'{class_name} is not defined by the module");'
        )
        lines.append("        return NULL;")
        lines.append("    }")
        lines.append(f"    Py_INCREF(cls{position});")
        lines.append(f"    PyTuple_SET_ITEM(types, {position}, cls{position});")
    return "\n".join(lines)


def extension_source(
    name: str,
    signatures: dict[str, NativeSignature],
    keys: dict[str, str],
    python_source: str,
    source_path: str,
) -> str:
    """The C of the extension: wrappers, adopters, and the module init.

    `keys` maps each qualname to the key the generated module binds it
    under (`f`, or `Class.method`); `python_source` is the module's
    optimized Python with the binding calls already in place.
    """
    from .wrapper import _type_assignments

    ordered = sorted(signatures.items())
    parts = [_HEADER, "#include <string.h>"]
    entries: list[str] = []
    calls: list[str] = []
    adopters: list[str] = []
    for index, (qualname, signature) in enumerate(ordered):
        parts.append(_function(index, signature))
        atoms = [
            C_TYPES[atom.removesuffix("*")] + ("*" if atom.endswith("*") else "")
            for atom in signature.params
        ]
        outs = [f"{C_TYPES[atom]} *" for atom in signature.returns]
        parts.append(f"int32_t {signature.symbol}({', '.join([*atoms, *outs])});")
        entries.append(f'    {{"{keys[qualname]}", {index}}},')
        method_name = keys[qualname].rpartition(".")[2]
        calls.append(
            f'    {{"{method_name}", (PyCFunction)(void *)ppy_call_{index}, METH_FASTCALL, NULL}},'
        )
        object_params = [p for p in signature.parameters if p.is_object]
        adopters.append(
            _ADOPTER.format(
                index=index,
                position=index,
                symbol=signature.symbol,
                lookup=_lookup(index, signature),
                assignments=_type_assignments(index, object_params),
            )
        )
    table = (
        "static PyObject *(*ppy_adopters[])(PyObject *) = {"
        + ", ".join(f"ppy_adopt_{index}" for index in range(len(ordered)))
        + "};"
    )
    parts.append(
        _INIT.format(
            entries="\n".join(entries),
            calls="\n".join(calls),
            adopters="\n".join(adopters) + "\n" + table,
            source=_c_bytes(python_source),
            name=name,
            file=_c_string(source_path),
        )
    )
    return "\n".join(parts)


def _c_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _c_bytes(text: str) -> str:
    """A string literal of any length, as a byte array initializer."""
    data = text.encode("utf-8")
    body = ",".join(str(b) for b in data)
    return f"{{{body},0}}"


def bound_python(code: str, line_map: dict[int, int], native_keys: frozenset[str]) -> str:
    """The generated module with its binding calls in place, as text."""
    tree = ast.parse(code)
    _restore_lines(tree, line_map)
    _insert_definition_bindings(tree, [(BINDER_NAME, native_keys)])
    return ast.unparse(tree) + "\n"


def build_python_extension(
    name: str,
    signatures: dict[str, NativeSignature],
    keys: dict[str, str],
    python_source: str,
    source_path: Path,
    objects: list[Path],
    destination_dir: Path,
    *,
    libraries: tuple[str, ...] = (),
    target: TargetInfo | None = None,
) -> Path:
    """Compile the extension `name` into `destination_dir`; the path."""
    target = target or host_target()
    if not target.is_host:
        raise ToolchainError(
            f"a Python extension is built against this interpreter, not for {target.triple}"
        )
    compiler = os.environ.get("CC") or _compiler()
    if compiler is None:
        raise ToolchainError("no C compiler (cc, gcc, or clang) is on PATH")
    include = Path(sysconfig.get_paths()["include"])
    if not (include / "Python.h").is_file():
        raise ToolchainError(f"CPython headers are missing from {include}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    source = destination_dir / f"{name}_extension.c"
    source.write_text(
        extension_source(name, signatures, keys, python_source, str(source_path)),
        encoding="utf-8",
    )
    library = destination_dir / f"{name}{target.extension_suffix}"
    command = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        "-I",
        str(include),
        str(source),
        *[str(o) for o in objects],
        "-o",
        str(library),
        *[f"-l{lib}" for lib in libraries],
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ToolchainError(f"the extension did not compile: {detail}")
    return library
