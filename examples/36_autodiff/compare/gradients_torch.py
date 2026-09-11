"""The same derivatives under PyTorch: `torch.func.grad` and `vmap` over the starting points."""

import time

import torch
from torch.func import grad, vmap

torch.set_default_dtype(torch.float64)
# PyTorch's default is one thread per logical core; on a hybrid CPU that
# count spins on the efficiency cores and the batch takes hundreds of
# milliseconds some runs and seven others. The performance cores' count is
# the number to give it, as a program would.
torch.set_num_threads(min(8, torch.get_num_threads()))


def f(x, y):
    return torch.sin(x) * y + x * x


def g(x):
    return torch.exp(-x * x) * torch.cos(3.0 * x)


df = grad(f)
dg = grad(g)


def newton(x):
    for _ in range(6):
        x = x - g(x) / dg(x)
    return x


newton_all = vmap(newton)


def timed(label, run):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        answer = run()
        best = min(best, time.perf_counter() - started)
    print(f"# {label}: {best * 1000:.2f} ms")
    return answer


def main():
    x = torch.tensor(0.7)
    y = torch.tensor(2.0, requires_grad=True)
    v = f(x, y)
    grad_y = torch.autograd.grad(v, y)[0]
    print(f"{float(df(x, torch.tensor(2.0))):.9f} {float(v.detach() * 2.0 + grad_y):.9f}")
    starts = 0.4 + torch.arange(100_000, dtype=torch.float64) * 1e-6
    newton_all(starts)
    roots = timed("newton, 100k starts", lambda: newton_all(starts))
    print(f"{float(roots.mean()):.9f} {float(vmap(g)(roots).abs().max()):.0e}")


main()
