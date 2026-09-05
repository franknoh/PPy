"""A trainer that runs the same under `python`, `torchrun`, and `accelerate launch`.

`import ppy` comes first and is the whole integration: the kernels in
`features.ppy` and `model.ppy` are served from their native build -- made
into `.ppy-cache` by the first process to import them, or ahead of time by
`ppy build --warm .` -- and every later process, every rank of a launch,
finds that build. Nothing here knows which launcher started it: the
distributed setup reads the environment the launchers agree on.
"""

import os
import sys
import time
from array import array

import ppy  # the import hook: .ppy modules, natively where a build exists
import torch
import torch.distributed as dist

import features
import model

ROWS = 20000
COLS = 16
STEPS = 100
RATE = 0.02
BINS = 8


def distributed() -> tuple[int, int]:
    """(rank, world size) under a launcher; (0, 1) under plain `python`."""
    world = int(os.getenv("WORLD_SIZE") or "1")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return int(os.getenv("RANK") or "0"), world


def device_for(rank: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK") or rank))
    return "cuda"


def preprocess(rank: int):
    """This rank's shard of rows, standardized and bucketed by the native kernels."""
    torch.manual_seed(1000 + rank)
    raw = array("d", torch.randn(ROWS * COLS).tolist())
    out = array("d", [0.0] * (ROWS * COLS * 2))
    counts = array("q", [0] * BINS)
    started = time.perf_counter()
    checksum = features.standardize(raw, out, ROWS, COLS)
    outside = features.bucketize(out, ROWS * COLS * 2, counts, BINS, -3.0, 3.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return out, checksum, outside, elapsed_ms


def train(
    x: torch.Tensor, y: torch.Tensor, params: list[torch.Tensor], world: int
) -> tuple[float, float]:
    first = last = 0.0
    for step in range(STEPS):
        loss = model.forward_loss(x, y, params[0], params[1], params[2], params[3])
        loss.backward()
        with torch.no_grad():
            for parameter in params:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if world > 1:
                    # Data parallel by hand: average the gradients across ranks.
                    dist.all_reduce(gradient)
                    gradient /= world
                parameter -= RATE * gradient
                parameter.grad = None
        last = loss.item()
        if step == 0:
            first = last
    return first, last


def main() -> None:
    rank, world = distributed()
    device = device_for(rank)
    out, checksum, outside, prep_ms = preprocess(rank)

    x = torch.tensor(out, dtype=torch.float32).reshape(ROWS, COLS * 2).to(device)
    torch.manual_seed(2000 + rank)
    y = torch.randn(ROWS, 1).to(device)
    torch.manual_seed(0)  # the same initial parameters on every rank
    params = [
        torch.randn(COLS * 2, 32, device=device, requires_grad=True),
        torch.zeros(32, device=device, requires_grad=True),
        torch.randn(32, 1, device=device, requires_grad=True),
        torch.zeros(1, device=device, requires_grad=True),
    ]
    with torch.no_grad():
        params[0] *= 0.1
        params[2] *= 0.1

    started = time.perf_counter()
    first, last = train(x, y, params, world)
    train_ms = (time.perf_counter() - started) * 1000.0

    native = "features" in ppy.native_imports()
    region = getattr(model.forward_loss, "__ppy_region__", False)
    # One write per rank, newline included: `print` writes the text and the
    # newline separately, and on the pipe the ranks share that interleaves.
    sys.stdout.write(
        f"# rank {rank}/{world} device={device} native={native} region={region} "
        f"loader={type(features.__loader__).__name__}\n"
        f"rank {rank}: prep  {prep_ms:8.1f} ms   checksum={checksum:.6f} outside={outside}\n"
        f"rank {rank}: train {train_ms:8.1f} ms   loss {first:.4f} -> {last:.4f}\n"
    )
    sys.stdout.flush()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
