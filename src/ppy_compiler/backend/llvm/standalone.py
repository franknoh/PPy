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
from ..c.runtime import program_main, support_source
from . import prover_for
from .jit import JitEngine, LlvmUnavailable, available
from .link import ToolchainError, _compiler, emit_object
from .lowering import LoweringResult, eligible

__all__ = ["build_standalone", "standalone_ir"]


def _fail(reporter, message: str, help_text: str | None = None) -> int:  # type: ignore[no-untyped-def]
    reporter.emit(Diagnostic("E1803", Severity.ERROR, message, help=help_text))
    return 1


def _program(bundle, reporter, entry: Path):  # type: ignore[no-untyped-def]
    """The functions a standalone program is made of, or the exit status.

    `main` in the entry module, and every project function it reaches,
    each of which must lower; the first that cannot is reported with the
    path that reaches it.
    """
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
    return module_name, entry_qualname, functions, reached_from


def standalone_ir(bundle, reporter, entry: Path, opt_level: int | None = None):  # type: ignore[no-untyped-def]
    """The canonical IR of a whole standalone program, or the exit status.

    What `ppy emit c --standalone` hands the C backend: the same reachable
    graph a standalone build compiles, lowered with the standalone shims
    and run through the shared passes. The result carries the entry's
    symbol as `ppy.entry`.
    """
    from ...lowering import lower_module_to_ir
    from .ir_pipeline import optimize

    program = _program(bundle, reporter, entry)
    if isinstance(program, int):
        return program
    module_name, entry_qualname, functions, reached_from = program
    analysis = bundle.analysis.modules[module_name]
    config = bundle.project.config
    lowered = lower_module_to_ir(
        analysis,
        functions,
        safeguards=config.llvm.safeguards or "hoisted",
        standalone=True,
        prover=prover_for(config),
        root=bundle.project.root,
    )
    for qualname, reason in sorted(lowered.rejected.items()):
        return _fail(reporter, _chain(reached_from, qualname, reason))
    if entry_qualname not in lowered.functions:
        return _fail(reporter, f"`{entry_qualname}` did not lower")
    level = opt_level if opt_level is not None else config.opt_level
    optimize(
        lowered.module,
        level,
        bundle.project.plugins,
        config.parallel,
        sanitize=config.llvm.sanitize,
    )
    lowered.module.attributes["ppy.entry"] = lowered.functions[entry_qualname].signature.symbol
    return lowered.module


def _runtime_sources(result) -> list[str]:  # type: ignore[no-untyped-def]
    """Runtime sources a standalone program compiles in: the async runtime, when it awaits."""
    if "ppy_aio" not in tuple(getattr(result, "libraries", ())):
        return []
    from ppy_runtime.aio import source_path

    return [str(source_path())]


def build_standalone(  # type: ignore[no-untyped-def]
    bundle, reporter, entry: Path, output: Path | None, opt_level: int | None = None
) -> int:
    if not available():
        raise LlvmUnavailable("llvmlite is not installed, so the LLVM backend is unavailable")
    compiler = _compiler()
    if compiler is None:
        raise ToolchainError("no C compiler (cc, gcc, or clang) is on PATH")

    program = _program(bundle, reporter, entry)
    if isinstance(program, int):
        return program
    module_name, entry_qualname, functions, reached_from = program
    analysis = bundle.analysis.modules[module_name]

    config = bundle.project.config
    from .ir_pipeline import lower_module_via_ir

    result: LoweringResult = lower_module_via_ir(
        analysis,
        functions,
        safeguards=config.llvm.safeguards or "hoisted",
        standalone=True,
        opt_level=opt_level if opt_level is not None else config.opt_level,
        prover=prover_for(config),
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
    support.write_text(support_source(), encoding="utf-8")
    main_c = build_directory / f"{module_name}_main.c"
    symbol = result.functions[entry_qualname].signature.symbol
    main_c.write_text(
        f"#include <stdint.h>\n#include <stdio.h>\n\nint32_t {symbol}(int64_t *out);\n\n"
        + program_main(symbol),
        encoding="utf-8",
    )
    destination = build_directory / entry.stem
    command = [
        compiler,
        "-O2",
        str(main_c),
        str(support),
        *_runtime_sources(result),
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
