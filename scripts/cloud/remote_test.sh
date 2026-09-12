#!/usr/bin/env bash
# The half of a hardware validation that runs on the Pod: environment, then
# tests, examples, and a few benchmarks, every step logged and its exit status
# recorded, nothing skipped silently. `runpod_matrix.py` copies this over,
# runs it, and brings /workspace/ppy-results back.
#
#   bash remote_test.sh <commit-sha> <cuda|multigpu|rocm> <min-devices>
set -uo pipefail
SHA=$1
MODE=$2
MIN=$3
RESULTS=/workspace/ppy-results
REPO=/workspace/PPy
mkdir -p "$RESULTS"
: > "$RESULTS/steps.txt"
echo "$SHA" > "$RESULTS/commit.txt"
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[remote $(date -u +%H:%M:%S)] $*"; }
# step NAME COMMAND...: run, log to its own file, record the status, never abort the matrix.
step() {
    local name=$1
    shift
    log "== $name"
    ( "$@" ) > "$RESULTS/$name.log" 2>&1
    local status=$?
    echo "$name $status" >> "$RESULTS/steps.txt"
    log "$name exit $status"
}
# across NAME COMMAND...: a step whose collectives cross devices, given fifteen minutes.
# A collective that never returns is the transport, not the program: on a host
# whose container cannot do PCIe peer-to-peer, NCCL spins forever. The step is
# then run again with NCCL_P2P_DISABLE=1, as its own record, so the report says
# which transport the result came from and the hang is not mistaken for a pass.
across() {
    local name=$1
    shift
    log "== $name"
    ( NCCL_DEBUG=WARN timeout 900 "$@" ) > "$RESULTS/$name.log" 2>&1
    local status=$?
    echo "$name $status" >> "$RESULTS/steps.txt"
    log "$name exit $status"
    if [ "$status" -ne 0 ]; then
        log "== $name failed ($status); again with NCCL_P2P_DISABLE=1"
        ( NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 timeout 900 "$@" ) > "$RESULTS/${name}_p2p_off.log" 2>&1
        status=$?
        echo "${name}_p2p_off $status" >> "$RESULTS/steps.txt"
        log "${name}_p2p_off exit $status"
    fi
}

# The vendor image already carries git, curl, and a C compiler; apt only fills a
# gap, and a mirror that hangs (it happens) is given ten minutes, not the run.
step apt bash -c 'for tool in git curl cc c++ make; do command -v $tool >/dev/null || missing=1; done; [ -z "${missing:-}" ] && echo "toolchain present" || (timeout 600 apt-get update -qq && timeout 600 apt-get install -y -qq git build-essential curl pkg-config)'
step uv bash -c 'command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh'
step clone bash -c "rm -rf $REPO && git clone -q https://github.com/franknoh/PPy $REPO && cd $REPO && git checkout -q $SHA && git rev-parse HEAD"
cd "$REPO" || exit 1
if [ "$MODE" = rocm ]; then
    # AMD's image carries a JAX built for ROCm with its PJRT plugin; PPy goes in
    # beside that interpreter with the image's `jax` and `jaxlib` as constraints,
    # so nothing in PPy's groups pulls the CPU wheel over them. The stack is then checked to be the stack that was there: any
    # change to jax, jaxlib, or the ROCm plugin after the sync is a failure.
    # The image's JAX lives in its `/opt/venv`; an SSH login may start without
    # that on `PATH`, so it is named outright when it is there.
    [ -d /opt/venv/bin ] && export PATH="/opt/venv/bin:$PATH"
    PY=$(command -v python3.12 || command -v python3)
    cat > "$RESULTS/jax_stack.py" <<'PYEOF'
import importlib.metadata as m

print(" ".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata["Name"].lower().startswith("jax"))))
PYEOF
    BEFORE=$("$PY" "$RESULTS/jax_stack.py")
    log "the image's JAX stack: $BEFORE at $PY"
    "$PY" -c 'import jax, jaxlib; print(f"jax=={jax.__version__}\njaxlib=={jaxlib.__version__}")' > "$RESULTS/constraints.txt"
    # The pip ROCm SDK keeps the device library away from `ROCM_PATH`; clang
    # reads `HIP_DEVICE_LIB_PATH`, so `hipcc` can build for the card.
    LIBS=$(find /opt/venv/lib /opt/rocm -type d -path '*amdgcn/bitcode' 2>/dev/null | head -1)
    if [ -n "$LIBS" ]; then export HIP_DEVICE_LIB_PATH="$LIBS"; log "HIP device library: $LIBS"; fi
    # Every group, under the constraints; when that cannot resolve (the `jax`
    # group's flax wants a newer jax than the image has), every group but
    # `jax`, with flatbuffers from it by name and flax not at all: nothing in
    # the ROCm scope needs it. The transcript says which one was installed.
    step sync bash -c "uv pip install -p $PY -c $RESULTS/constraints.txt -e '.' --group all pytest && echo 'installed: every group' || (echo 'the jax group did not resolve against the image; installing without it' && uv pip install -p $PY -c $RESULTS/constraints.txt -e '.' --group dev --group torch --group uvicorn --group scipy --group pandas --group pyarrow 'flatbuffers>=24.0' pytest && echo 'installed: every group but jax')"
    step jax_plugin bash -c "AFTER=\$($PY $RESULTS/jax_stack.py); echo \"before: $BEFORE\"; echo \"after: \$AFTER\"; [ \"\$AFTER\" = \"$BEFORE\" ]"
    PPY=$($PY -c 'import shutil; print(shutil.which("ppy") or "")')
    [ -n "$PPY" ] || PPY="$PY -m ppy_compiler"
    # ROCm 10 in AMD's image is the pip SDK under the venv, not `/opt/rocm`:
    # whichever of the tools is there reports; the accelerator step is the test.
    step vendor bash -c 'ls /dev/kfd /dev/dri 2>&1; (rocminfo | grep -E "Marketing Name|gfx") 2>&1; rocm-smi --showproductname --showdriverversion 2>&1; cat /opt/rocm/.info/version 2>/dev/null; rocm-sdk version 2>&1; echo "ROCM_PATH=$ROCM_PATH"; true'
else
    step sync bash -c 'uv python install -q 3.13 && uv sync -q -p 3.13 --group all'
    PY=$REPO/.venv/bin/python
    PPY=$REPO/.venv/bin/ppy
    JAX_VERSION=$("$PY" -c 'import jax; print(jax.__version__)')
    log "jax $JAX_VERSION from the lock; installing the accelerator plugin for it"
    step jax_plugin bash -c "uv pip install -p $PY 'jax[cuda12]==$JAX_VERSION'"
    step vendor bash -c 'nvidia-smi; nvidia-smi -L; nvcc --version || echo "nvcc: not installed"'
fi
# The environment as installed: what is in the venv after the plugin, in the lock's terms.
step versions bash -c "$PY -c 'import sys, jax, jaxlib; print(sys.version); print(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)'; uv pip list -p $PY | grep -Ei 'jax|nvidia|rocm|torch|numpy|ppy'"
# `setup` stops here: a Pod left running for a hand-driven investigation.
if [ "$MODE" = setup ]; then
    log "environment ready; stopping as asked"
    exit 0
fi
# The acceptance check: a CPU-only JAX fails here, and the matrix stops being green.
across accelerator "$PY" scripts/cloud/accelerator_check.py --require gpu --min-devices "$MIN" --out "$RESULTS/accelerator.json"
# PPy's own device paths, the reason each took, and the tests around them.
step xla_example bash -c "cd examples/39_xla && $PPY run device_math.ppy && $PPY explain device_math.ppy:6"
if [ "$MODE" = rocm ]; then
    # `ppy.hip` is source only: the HIP C++ is written and looked at, never launched.
    step hip_emit bash -c "cd examples/38_cuda && $PPY emit hip saxpy.ppy > $RESULTS/saxpy.hip.cpp && grep -q '__global__' $RESULTS/saxpy.hip.cpp && grep -q 'hipLaunchKernelGGL\|<<<' $RESULTS/saxpy.hip.cpp && wc -l $RESULTS/saxpy.hip.cpp && if command -v hipcc >/dev/null; then hipcc -c -x hip $RESULTS/saxpy.hip.cpp -o $RESULTS/saxpy.hip.o && echo 'hipcc: compiled for the device' || echo 'hipcc: present but failed to compile the source'; else echo 'hipcc: not installed, source not compiled'; fi"
    step cuda_example bash -c "cd examples/38_cuda && $PPY run saxpy.ppy"
    step tile_example bash -c "cd examples/44_tile && $PPY run tiles.ppy"
else
    step cuda_example bash -c "cd examples/38_cuda && $PPY run saxpy.ppy && $PPY explain saxpy.ppy:9 && $PPY explain saxpy.ppy:36"
    step tile_example bash -c "cd examples/44_tile && $PPY run tiles.ppy"
fi
step gpu_tests "$PY" -m pytest tests/test_gpu_frontend.py tests/test_ir_gpu.py tests/test_tile.py tests/test_xla.py -q -p no:cacheprovider
step limits_tests "$PY" -m pytest tests/test_native_limits.py tests/test_multi_device_jax.py -q
if [ "$MIN" -ge 2 ]; then
    across multigpu_train bash -c "cd examples/45_multi_gpu_jax && $PPY run train.ppy && $PY train.ppy"
    # One process per device, all started at once; an empty loop is a failure,
    # not a pass, so the count of transcripts is checked against the devices.
    across multiprocess bash -c "
        status=0
        for i in \$(seq 0 $((MIN - 1))); do
            $PY scripts/cloud/multiprocess_smoke.py \$i $MIN > $RESULTS/multiprocess_\$i.log 2>&1 &
        done
        for job in \$(jobs -p); do wait \$job || status=1; done
        cat $RESULTS/multiprocess_*.log
        [ \$(grep -l ': PASS' $RESULTS/multiprocess_*.log | wc -l) -eq $MIN ] || status=1
        exit \$status"
fi
# Benchmarks: validation on this machine, never the canonical tables.
step bench_jax bash -c "cd examples/05_numpy && $PY ../compare.py 3 'ppy=$PPY run compare/fusion_bench.ppy' 'jax=$PY compare/fusion_jax.py'"
step bench_torch bash -c "cd examples/09_torch && $PY ../compare.py 3 'ppy=$PPY run compare/layer_bench.ppy' 'eager=$PY compare/layer_eager.py' 'compile=$PY compare/layer_compile.py'"
if [ "$MODE" != rocm ]; then
    step cupy bash -c "uv pip install -p $PY cupy-cuda12x triton"
    step bench_cuda bash -c "cd examples/38_cuda && $PY ../compare.py 3 'ppy=$PPY run compare/saxpy_bench.ppy' 'cupy=$PY compare/saxpy_cupy.py'"
    step bench_tile bash -c "cd examples/44_tile && $PY ../compare.py 3 'ppy=$PPY run compare/tiles_bench.ppy'"
    # Triton's wheel needs a driver its backend can see; where it cannot, the row is its own.
    step bench_tile_triton bash -c "cd examples/44_tile && $PY ../compare.py 3 'ppy=$PPY run compare/tiles_bench.ppy' 'triton=$PY compare/tiles_triton.py'"
fi
log "done; steps:"
cat "$RESULTS/steps.txt"
