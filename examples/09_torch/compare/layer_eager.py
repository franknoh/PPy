"""The same three operators as PyTorch runs them eagerly: one Python round trip per operator."""

import time

import torch


def layer(x, weight, bias):
    return torch.relu(torch.add(torch.matmul(x, weight), bias))


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    x = torch.randn(8, 32)
    w = torch.randn(32, 32)
    b = torch.randn(32)
    for _ in range(300):
        layer(x, w, b)
    rounds = 20000
    best = 1e9
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(rounds):
            out = layer(x, w, b)
        best = min(best, (time.perf_counter() - start) / rounds * 1000.0)
    print(f"# layer, per call: {best:.4f} ms")
    print(f"{float(out.sum()):.4f}")


main()
