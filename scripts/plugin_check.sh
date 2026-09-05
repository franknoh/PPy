#!/usr/bin/env bash
# One plugin's claim, proven rather than skipped.
#
# The base gate installs no accelerator runtime, so every plugin test skips
# there and a regression can merge with CI green. This runs the tests for one
# plugin and fails if none of them actually ran -- a skip is not a pass.
set -euo pipefail
cd "$(dirname "$0")/.."

plugin="${1:?usage: plugin_check.sh <torch|jax|uvicorn>}"
case "$plugin" in
  torch)   library=torch;   examples=(09_torch 21_training_torch 31_torchrun) ;;
  jax)     library=jax;     examples=(22_training_jax 25_jax_export 29_flax) ;;
  uvicorn) library=uvicorn; examples=(27_uvicorn) ;;
  *) echo "unknown plugin: $plugin" >&2; exit 2 ;;
esac

uv run python -c "import ${library}" || {
  echo "the ${plugin} group is not installed" >&2
  exit 1
}

report=$(mktemp)
trap 'rm -f "$report"' EXIT
uv run pytest -q -k "$plugin" tests/test_plugins.py tests/test_library_integration.py \
  | tee "$report"

if ! grep -qE "[0-9]+ passed" "$report"; then
  echo "no ${plugin} test ran: a skipped plugin proves nothing" >&2
  exit 1
fi

uv run python examples/run_all.py "${examples[@]}"
