#!/usr/bin/env bash
# The half of a comparison run that happens on the Pod: build the environment
# the tables want, run `compare_docs.py --write` over the folders named, and
# leave the rewritten READMEs and records under /workspace/ppy-results.
# `runpod_bench.py` copies this over, runs it, and brings that back.
#
#   bash remote_bench.sh <commit-sha> <folder> [<folder> ...]
set -uo pipefail
SHA=$1
shift
FOLDERS=("$@")
RESULTS=/workspace/ppy-results
REPO=/workspace/PPy
mkdir -p "$RESULTS"
: > "$RESULTS/steps.txt"
echo "$SHA" > "$RESULTS/commit.txt"
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[remote $(date -u +%H:%M:%S)] $*"; }
step() {
    local name=$1
    shift
    log "== $name"
    ( "$@" ) > "$RESULTS/$name.log" 2>&1
    local status=$?
    echo "$name $status" >> "$RESULTS/steps.txt"
    log "$name exit $status"
}

step apt bash -c 'for tool in git curl cc c++ make ninja; do command -v $tool >/dev/null || missing=1; done; [ -z "${missing:-}" ] && echo "toolchain present" || (timeout 600 apt-get update -qq && timeout 600 apt-get install -y -qq git build-essential curl ninja-build)'
step uv bash -c 'command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh'
step clone bash -c "rm -rf $REPO && git clone -q https://github.com/franknoh/PPy $REPO && cd $REPO && git checkout -q $SHA && git rev-parse HEAD"
cd "$REPO" || exit 1

# The lock resolves torch from the CPU index, which is right for a checkout
# and wrong for a card; the accelerator build is the local override the
# project documents, named here rather than pinned in `pyproject.toml`.
step sync bash -c "uv python install -q 3.13 && uv sync -q -p 3.13 --group all && uv pip install --reinstall --index-strategy unsafe-best-match --extra-index-url ${PPY_TORCH_INDEX:-https://download.pytorch.org/whl/cu128} torch"
PY=$REPO/.venv/bin/python
step vendor bash -c "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; $PY -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, torch.cuda.get_device_name(0))'; $PY -VV; c++ --version | head -1"

# A CPU PyTorch here would measure the wrong machine quietly; it is a failure.
step accelerator bash -c "$PY -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'"
if ! grep -q '^accelerator 0$' "$RESULTS/steps.txt"; then
    log "the installed PyTorch has no device; nothing is measured and no table is written"
    exit 1
fi

for folder in "${FOLDERS[@]}"; do
    step "compare_$folder" env PPY_TORCH_CUDA_PYTHON="$PY" "$PY" scripts/compare_docs.py "$folder" --write
    if [ -f "examples/$folder/README.md" ]; then
        mkdir -p "$RESULTS/$folder/compare"
        cp "examples/$folder/README.md" "$RESULTS/$folder/README.md"
        [ -f "examples/$folder/compare/measurements.json" ] &&
            cp "examples/$folder/compare/measurements.json" "$RESULTS/$folder/compare/measurements.json"
    fi
done

log "done; steps:"
cat "$RESULTS/steps.txt"
