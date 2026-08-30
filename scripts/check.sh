#!/usr/bin/env bash
# The one gate: CI runs exactly this, so passing here means passing there.
set -euo pipefail
cd "$(dirname "$0")/.."

FILES=(examples/run_all.py examples/lint_all.py examples/verify_conversions.py
       examples/bench_startup.py examples/bench_boundary.py)

uv run ruff check src tests "${FILES[@]}"
uv run ruff format --check src tests
uv run pylint src/ppy src/ppy_compiler src/ppy_runtime tests "${FILES[@]}"
uv run pytest -q
uv run python examples/verify_conversions.py
uv run python examples/run_all.py
uv run python examples/lint_all.py
