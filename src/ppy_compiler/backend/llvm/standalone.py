"""`ppy build --standalone`: a native executable with no CPython inside.

The reachable graph from `main` must be entirely native; anything else is
rejected with the path that reaches it (`E1803`), not worked around. What
runs is what the hybrid native path would have run -- same lowering, same
guard modes -- plus a few C shims for printing. There is no Python to fall
back to, so a failed guard is a runtime abort, never a silent wrap.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ...diagnostics import Diagnostic, Severity
from .jit import JitEngine, LlvmUnavailable, available
from .link import ToolchainError, _compiler, emit_object
from .lowering import LoweringResult, eligible, lower_module

__all__ = ["build_standalone"]

_SUPPORT = """#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void ppy_rt_print_i64(int64_t value) { printf("%" PRId64, value); }
void ppy_rt_print_bool(int8_t value) { fputs(value ? "True" : "False", stdout); }
void ppy_rt_print_str(const char *text, int64_t length) {
    fwrite(text, 1, (size_t)length, stdout);
}
void ppy_rt_print_sep(void) { fputc(' ', stdout); }
void ppy_rt_print_nl(void) { fputc('\\n', stdout); }

/* `ppy.input[int]()` with no interpreter under it: the same buffered scan
 * the runtime reader does, reading standard input directly. At end of input
 * it answers 0, because a standalone binary has no exception to raise. */
static int ppy_rt_next(void) {
    static char room[1 << 16];
    static long filled = 0;
    static long position = 0;
    if (position == filled) {
        filled = (long)fread(room, 1, sizeof room, stdin);
        if (filled <= 0) {
            return -1;
        }
        position = 0;
    }
    return (unsigned char)room[position++];
}

/* A buffer a standalone program makes for itself. There is no interpreter
 * to own it, and a program that exits is the only lifetime that matters. */
int64_t *ppy_rt_alloc(int64_t count) {
    int64_t *room = calloc(count > 0 ? (size_t)count : 1, sizeof(int64_t));
    if (room == NULL) {
        fputs("ppy: out of memory\\n", stderr);
        exit(1);
    }
    return room;
}

int64_t ppy_rt_read_ints(int64_t *data, int64_t capacity) {
    int64_t count = 0;
    while (count < capacity) {
        int c = ppy_rt_next();
        while (c != -1 && (c < '0' || c > '9') && c != '-') {
            c = ppy_rt_next();
        }
        if (c == -1) {
            break;
        }
        int negative = 0;
        if (c == '-') {
            negative = 1;
            c = ppy_rt_next();
        }
        int64_t value = 0;
        while (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
            c = ppy_rt_next();
        }
        data[count++] = negative ? -value : value;
    }
    return count;
}

int64_t ppy_rt_read_int(void) {
    int c = ppy_rt_next();
    while (c != -1 && (c < '0' || c > '9') && c != '-') {
        c = ppy_rt_next();
    }
    if (c == -1) {
        return 0;
    }
    int negative = 0;
    if (c == '-') {
        negative = 1;
        c = ppy_rt_next();
    }
    int64_t value = 0;
    while (c >= '0' && c <= '9') {
        value = value * 10 + (c - '0');
        c = ppy_rt_next();
    }
    return negative ? -value : value;
}
"""

_MAIN = """#include <stdint.h>
#include <stdio.h>

int32_t {symbol}(int64_t *out);

int main(void) {{
    int64_t out = 0;
    int32_t status = {symbol}(&out);
    if (status != 0) {{
        fputs("ppy: a native guard failed and there is no Python to fall back to\\n", stderr);
        return 70;
    }}
    return 0;
}}
"""


def _fail(reporter, message: str, help_text: str | None = None) -> int:  # type: ignore[no-untyped-def]
    reporter.emit(Diagnostic("E1803", Severity.ERROR, message, help=help_text))
    return 1


def build_standalone(  # type: ignore[no-untyped-def]
    bundle, reporter, entry: Path, output: Path | None, opt_level: int | None = None
) -> int:
    if not available():
        raise LlvmUnavailable("llvmlite is not installed, so the LLVM backend is unavailable")
    compiler = _compiler()
    if compiler is None:
        raise ToolchainError("no C compiler (cc, gcc, or clang) is on PATH")

    module_name = None
    for name, symbols in bundle.symbols.modules.items():
        if symbols.path == entry.resolve():
            module_name = name
            break
    if module_name is None:
        return _fail(reporter, f"{entry} is not a module of this project")
    symbols = bundle.symbols.modules[module_name]
    analysis = bundle.analysis.modules[module_name]

    entry_qualname = f"{module_name}.main"
    if "main" not in symbols.functions:
        return _fail(
            reporter,
            f"a standalone build starts at `{module_name}.main`, which does not exist",
            help_text="define `def main() -> None:` and call it at module level",
        )
    problem = _module_shape(symbols)
    if problem is not None:
        return _fail(
            reporter, problem, "a standalone module holds defs, `import ppy`, and one `main()` call"
        )

    # Reachability: every project function `main` can reach must lower.
    reachable: list[str] = []
    frontier = [entry_qualname]
    reached_from: dict[str, str] = {entry_qualname: "entrypoint"}
    while frontier:
        qualname = frontier.pop()
        if qualname in reachable:
            continue
        reachable.append(qualname)
        function = analysis.functions.get(qualname)
        if function is None:
            continue
        for callee in sorted(function.calls):
            if callee in analysis.functions and callee not in reached_from:
                reached_from[callee] = qualname
                frontier.append(callee)

    functions = {}
    for qualname in reachable:
        info = symbols.functions.get(qualname.rpartition(".")[2])
        function = analysis.functions.get(qualname)
        if info is None or function is None:
            return _fail(
                reporter, _chain(reached_from, qualname, "is not a function of this module")
            )
        ok, reason = eligible(info, function, allow_io=True)
        if not ok:
            return _fail(reporter, _chain(reached_from, qualname, reason))
        functions[qualname] = (info, function, info.node)

    result: LoweringResult = lower_module(
        analysis,
        functions,
        safeguards=bundle.project.config.llvm.safeguards or "off",
        standalone=True,
    )
    for qualname, reason in sorted(result.rejected.items()):
        return _fail(reporter, _chain(reached_from, qualname, reason))
    if entry_qualname not in result.functions:
        return _fail(reporter, f"`{entry_qualname}` did not lower")

    level = opt_level if opt_level is not None else bundle.project.config.opt_level
    build_directory = output or (bundle.project.config.cache_path / "standalone")
    build_directory.mkdir(parents=True, exist_ok=True)
    engine = JitEngine(opt_level=level).open()
    object_path = build_directory / f"{module_name}.o"
    emit_object(engine, result.ir, object_path, host_cpu=bundle.project.config.llvm.host_cpu)

    support = build_directory / "ppy_support.c"
    support.write_text(_SUPPORT, encoding="utf-8")
    main_c = build_directory / f"{module_name}_main.c"
    main_c.write_text(
        _MAIN.format(symbol=result.functions[entry_qualname].signature.symbol),
        encoding="utf-8",
    )
    destination = build_directory / entry.stem
    command = [
        compiler,
        "-O2",
        str(main_c),
        str(support),
        str(object_path),
        "-o",
        str(destination),
        "-lm",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ToolchainError(
            f"standalone link failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    os.chmod(destination, 0o755)
    reporter.note(f"standalone executable: {destination}")
    return 0


def _binds_a_constant(statement, constants: dict) -> bool:  # type: ignore[no-untyped-def]
    """`MOD = 10**9 + 7` at module level: a value, not a step to run."""
    import ast

    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id in constants
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        return isinstance(target, ast.Name) and target.id in constants
    return False


def _module_shape(symbols) -> str | None:  # type: ignore[no-untyped-def]
    import ast

    for index, statement in enumerate(symbols.module.tree.body):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # A docstring is text in the binary, not a statement that runs.
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        # A proven constant global is folded into the code that reads it, so
        # nothing has to run to bind the name.
        if _binds_a_constant(statement, symbols.constant_globals):
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            names = getattr(statement, "module", None) or ""
            listed = [alias.name for alias in statement.names]
            if names == "ppy" or listed == ["ppy"]:
                continue
            return f"`{ast.unparse(statement)}` reaches the Python runtime"
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "main"
            and not statement.value.args
        ):
            continue
        return f"`{ast.unparse(statement).splitlines()[0]}` cannot run without CPython"
    return None


def _chain(reached_from: dict[str, str], qualname: str, reason: str) -> str:
    steps = [qualname]
    while reached_from.get(steps[-1], "entrypoint") != "entrypoint":
        steps.append(reached_from[steps[-1]])
    path = " -> ".join(reversed(steps))
    return f"a standalone build requires a fully native reachable graph: {path}: {reason}"
