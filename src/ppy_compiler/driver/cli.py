"""The `ppy` command-line driver (spec 4.3, 4.4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import commands
from .reporting import Reporter

__all__ = ["main", "build_parser"]

_EXECUTION_SUFFIXES = (".ppy", ".py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppy",
        description="Pretty Python compiler and runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "running a file:\n"
            "  ppy FILE.ppy [-- ARGS...]     optimize and run on the Python backend\n"
            "  ppy run FILE.ppy [-- ARGS...] compile and run through the LLVM backend\n"
        ),
    )
    parser.add_argument("--version", action="store_true", help="print the compiler version")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress non-error output")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("-O", "--opt-level", type=int, choices=(0, 1, 2, 3), help="override the optimization level")
    parser.add_argument("--no-strict", action="store_true", help="downgrade strict-mode errors where possible")

    subparsers = parser.add_subparsers(dest="command")

    convert = subparsers.add_parser("convert", help="convert .py sources to typed .ppy sources")
    convert.add_argument("path", type=Path)
    convert.add_argument(
        "--in-place",
        action="store_true",
        help="replace the source: write the .ppy and remove the .py it came from",
    )
    convert.add_argument("--force", action="store_true", help="overwrite an existing .ppy file")
    convert.add_argument("--dry-run", action="store_true", help="print the result without writing")
    convert.add_argument(
        "--format",
        dest="apply_format",
        action="store_true",
        help="run the project's formatter over the result, after the deterministic pass",
    )
    convert.add_argument(
        "--promote-buffers",
        action="store_true",
        help="declare read-only numeric list parameters as borrowed buffers, "
        "rewriting the values that feed them into `array.array`",
    )

    run = subparsers.add_parser("run", help="compile through LLVM and execute")
    run.add_argument("file", type=Path)
    run.add_argument("args", nargs=argparse.REMAINDER)

    build = subparsers.add_parser("build", help="compile without running")
    build.add_argument("target", type=Path)
    build.add_argument("--backend", choices=("llvm", "python"), default="llvm")
    build.add_argument("-o", "--output", type=Path, help="output directory")

    check = subparsers.add_parser("check", help="run all static validation")
    check.add_argument("path", type=Path, nargs="?", default=Path("."))
    check.add_argument("--remarks", action="store_true", help="show optimization remarks")

    fmt = subparsers.add_parser("fmt", help="format PPY sources")
    fmt.add_argument("path", type=Path, nargs="?", default=Path("."))
    fmt.add_argument("--check", action="store_true", help="exit non-zero if a file would change")

    explain = subparsers.add_parser("explain", help="explain a location, function, or diagnostic code")
    explain.add_argument("location", help="FILE:LINE, a function qualname, or a diagnostic code")

    inspect = subparsers.add_parser("inspect", help="show generated artifacts for a target")
    inspect.add_argument("target", type=Path)
    inspect.add_argument("--backend", choices=("python", "llvm"), default="python")
    inspect.add_argument("--ir", action="store_true", help="print backend IR instead of generated Python")

    lint = subparsers.add_parser("lint", help="run an installed type checker or linter")
    lint.add_argument("path", type=Path, nargs="?", default=Path("."))
    lint.add_argument(
        "--backend",
        choices=("auto", "pyright", "pylint", "ruff", "mypy"),
        default="auto",
        help="which tool to run; `auto` picks the first one installed",
    )
    lint.add_argument(
        "--all-rules",
        action="store_true",
        help="ruff only: enable every rule instead of the project's selection",
    )
    lint.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="run the tool in its default mode instead of its strictest one",
    )

    cache = subparsers.add_parser("cache", help="inspect and maintain the compilation cache")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("status", help="show cache contents and size")
    cache_sub.add_parser("clean", help="remove every cached artifact")
    gc = cache_sub.add_parser("gc", help="remove unreachable and expired artifacts")
    gc.add_argument("--max-age-days", type=float, default=30.0)
    gc.add_argument("--max-bytes", type=int, default=None)

    subparsers.add_parser("clean", help="remove the project cache and build outputs")

    doctor = subparsers.add_parser("doctor", help="report the state of the toolchain")
    doctor.add_argument("--verbose", action="store_true")

    test = subparsers.add_parser("test", help="run differential conformance tests for a target")
    test.add_argument("path", type=Path, nargs="?", default=Path("."))
    test.add_argument(
        "--backend",
        choices=("differential", "pytest"),
        default="differential",
        help="`differential` compares the three paths; `pytest` runs a test suite",
    )
    test.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the backend")

    lsp = subparsers.add_parser("lsp", help="run the PPY language server over stdio")
    lsp.add_argument("--root", type=Path, default=Path("."), help="project root")

    return parser


def _split_execution_argv(argv: list[str]) -> tuple[Path, list[str]] | None:
    """Recognize the `ppy FILE.ppy [-- ARGS...]` form (spec 4.4)."""
    for index, token in enumerate(argv):
        if token.startswith("-"):
            continue
        candidate = Path(token)
        if candidate.suffix in _EXECUTION_SUFFIXES:
            rest = argv[index + 1 :]
            if rest and rest[0] == "--":
                rest = rest[1:]
            return candidate, rest
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    known = {
        "convert", "run", "build", "check", "fmt", "explain",
        "inspect", "cache", "clean", "doctor", "test", "lint", "lsp",
    }
    execution = None
    if argv and argv[0] not in known:
        execution = _split_execution_argv(argv)

    if execution is not None:
        file, program_args = execution
        options = parser.parse_args([a for a in argv if a.startswith("-") and a != "--"][:0])
        flags = [a for a in argv if a.startswith("-") and a not in {"--"}]
        if flags:
            options = parser.parse_args(flags)
        reporter = _reporter(options)
        return commands.run_python_backend(file, program_args, options, reporter)

    options = parser.parse_args(argv)
    if options.version:
        from .pipeline import COMPILER_VERSION

        print(f"ppy {COMPILER_VERSION}")
        return 0
    if options.command is None:
        parser.print_help()
        return 0

    reporter = _reporter(options)
    match options.command:
        case "convert":
            return commands.convert(options, reporter)
        case "run":
            return commands.run_llvm_backend(options.file, _program_args(options.args), options, reporter)
        case "build":
            return commands.build(options, reporter)
        case "check":
            return commands.check(options, reporter)
        case "fmt":
            return commands.fmt(options, reporter)
        case "explain":
            return commands.explain(options, reporter)
        case "lint":
            from . import linting

            return linting.run_lint(options, reporter)
        case "inspect":
            return commands.inspect(options, reporter)
        case "cache":
            return commands.cache(options, reporter)
        case "clean":
            return commands.clean(options, reporter)
        case "doctor":
            return commands.doctor(options, reporter)
        case "test":
            if getattr(options, "backend", "differential") == "pytest":
                from . import linting

                return linting.run_pytest(options, reporter)
            return commands.differential_test(options, reporter)
        case "lsp":
            return commands.language_server(options, reporter)
    parser.print_help()
    return 2


def _program_args(rest: list[str]) -> list[str]:
    return rest[1:] if rest and rest[0] == "--" else rest


def _reporter(options: argparse.Namespace) -> Reporter:
    color = None if options.color == "auto" else options.color == "always"
    return Reporter(color=color, quiet=options.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
