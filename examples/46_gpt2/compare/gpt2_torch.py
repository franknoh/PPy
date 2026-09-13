"""The same GPT-2 XL in plain PyTorch: eager, or through `torch.compile`.

The weights are drawn in the same order as `gpt2_bench.ppy` draws them, so
all four columns are the same model on the same numbers. The mode is an
argument because eager and the two compiled modes differ by one line.

    python gpt2_torch.py eager
    python gpt2_torch.py compile
    python gpt2_torch.py reduce-overhead
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

EPS = 1e-5
LAYERS = 48
HEADS = 25
HEAD_DIM = 64
VOCAB = 50257
ON_CUDA = torch.cuda.is_available()


def block(
    x,
    ln1_w,
    ln1_b,
    q_w,
    q_b,
    k_w,
    k_b,
    v_w,
    v_b,
    o_w,
    o_b,
    ln2_w,
    ln2_b,
    fc_w,
    fc_b,
    proj_w,
    proj_b,
    batch,
    length,
    heads,
    head_dim,
    width,
    eps,
):
    h = torch.layer_norm(x, (width,), ln1_w, ln1_b, eps)
    q = F.linear(h, q_w, q_b).reshape((batch, length, heads, head_dim)).transpose(1, 2)
    k = F.linear(h, k_w, k_b).reshape((batch, length, heads, head_dim)).transpose(1, 2)
    v = F.linear(h, v_w, v_b).reshape((batch, length, heads, head_dim)).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    merged = a.transpose(1, 2).reshape((batch, length, width))
    attended = torch.add(x, F.linear(merged, o_w, o_b))
    n = torch.layer_norm(attended, (width,), ln2_w, ln2_b, eps)
    f = F.gelu(F.linear(n, fc_w, fc_b), approximate="tanh")
    return torch.add(attended, F.linear(f, proj_w, proj_b))


def scaled(rows, columns, spread, device):
    return torch.mul(torch.randn(rows, columns), spread).to(device).bfloat16()


def flat(width, value, device):
    return torch.mul(torch.ones(width), value).to(device).bfloat16()


class Block:
    def __init__(self, width, device):
        inner = 4 * width
        spread = 0.02
        self.ln1_w = flat(width, 1.0, device)
        self.ln1_b = flat(width, 0.0, device)
        self.q_w = scaled(width, width, spread, device)
        self.q_b = flat(width, 0.0, device)
        self.k_w = scaled(width, width, spread, device)
        self.k_b = flat(width, 0.0, device)
        self.v_w = scaled(width, width, spread, device)
        self.v_b = flat(width, 0.0, device)
        self.o_w = scaled(width, width, spread, device)
        self.o_b = flat(width, 0.0, device)
        self.ln2_w = flat(width, 1.0, device)
        self.ln2_b = flat(width, 0.0, device)
        self.fc_w = scaled(inner, width, spread, device)
        self.fc_b = flat(inner, 0.0, device)
        self.proj_w = scaled(width, inner, spread, device)
        self.proj_b = flat(width, 0.0, device)

    def trainable(self):
        self.fc_w = self.fc_w.requires_grad_(True)
        self.proj_w = self.proj_w.requires_grad_(True)

    def zero_grads(self):
        self.fc_w.grad = None
        self.proj_w.grad = None


class Model:
    def __init__(self, device):
        width = HEADS * HEAD_DIM
        self.width = width
        self.wte = scaled(VOCAB, width, 0.02, device)
        self.wpe = scaled(1024, width, 0.01, device)
        self.lnf_w = flat(width, 1.0, device)
        self.lnf_b = flat(width, 0.0, device)
        self.blocks = [Block(width, device) for _i in range(LAYERS)]

    def trainable(self):
        for layer in self.blocks:
            layer.trainable()

    def zero_grads(self):
        for layer in self.blocks:
            layer.zero_grads()

    def forward(self, ids, batch, length):
        h = torch.add(self.wte[ids], self.wpe[0:length])
        for layer in self.blocks:
            h = block(
                h,
                layer.ln1_w,
                layer.ln1_b,
                layer.q_w,
                layer.q_b,
                layer.k_w,
                layer.k_b,
                layer.v_w,
                layer.v_b,
                layer.o_w,
                layer.o_b,
                layer.ln2_w,
                layer.ln2_b,
                layer.fc_w,
                layer.fc_b,
                layer.proj_w,
                layer.proj_b,
                batch,
                length,
                HEADS,
                HEAD_DIM,
                self.width,
                EPS,
            )
        normed = torch.layer_norm(h, (self.width,), self.lnf_w, self.lnf_b, EPS)
        return torch.matmul(normed, self.wte.t())


def ids_for(batch, length, device):
    return torch.remainder(torch.arange(batch * length).reshape((batch, length)), VOCAB).to(device)


def best_ms(work, rounds):
    best = 1e9
    for _round in range(rounds):
        if ON_CUDA:
            torch.cuda.synchronize()
        start = time.perf_counter()
        work()
        if ON_CUDA:
            torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def answer(logits):
    wide = logits.float()
    return (
        f"mean={float(wide.mean()):.3f} "
        f"absmean={float(wide.abs().mean()):.3f} "
        f"meansq={float(torch.mul(wide, wide).mean()):.3f}"
    )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "eager"
    device = "cuda" if ON_CUDA else "cpu"
    torch.manual_seed(0)
    model = Model(device)

    forward = model.forward
    if mode == "compile":
        forward = torch.compile(model.forward)
    elif mode == "reduce-overhead":
        forward = torch.compile(model.forward, mode="reduce-overhead")
    elif mode != "eager":
        raise SystemExit(f"unknown mode {mode!r}")

    for batch, length in [(1, 32), (1, 128), (1, 512), (8, 512)]:
        ids = ids_for(batch, length, device)
        for _warm in range(8):
            forward(ids, batch, length)
        print(answer(forward(ids, batch, length)))
        elapsed = best_ms(lambda: forward(ids, batch, length), 10)  # noqa: B023
        print(f"# forward b{batch} s{length}: {elapsed:.3f} ms")

    model.trainable()
    ids = ids_for(4, 512, device)
    for _warm in range(8):
        model.zero_grads()
        forward(ids, 4, 512).sum().backward()

    def training_step():
        # Accumulating instead would leave `reduce-overhead` with a gradient
        # buffer its CUDA graph had already overwritten.
        model.zero_grads()
        forward(ids, 4, 512).sum().backward()

    step = best_ms(training_step, 10)
    print(f"# training step b4 s512: {step:.3f} ms")


if __name__ == "__main__":
    main()
