import math
import time

import torch


def standardize(raw, out, rows, cols):
    total = 0.0
    for row in range(rows):
        base = row * cols
        target = row * cols * 2

        sum_ = 0.0
        for i in range(cols):
            sum_ += raw[base + i]
        mean = sum_ / cols

        spread = 0.0
        for i in range(cols):
            spread += (raw[base + i] - mean) * (raw[base + i] - mean)
        deviation = math.sqrt(spread / cols) + 1e-8

        for i in range(cols):
            value = (raw[base + i] - mean) / deviation
            out[target + i] = value
            total += value
        for i in range(cols):
            interaction = out[target + i] * out[target + (i + 1) % cols]
            out[target + cols + i] = interaction
            total += interaction
    return total


def forward_loss(x, y, w1, b1, w2, b2):
    hidden = torch.relu(torch.add(torch.matmul(x, w1), b1))
    predicted = torch.add(torch.matmul(hidden, w2), b2)
    residual = torch.sub(predicted, y)
    return torch.mean(torch.mul(residual, residual))


def descend(params, rate):
    with torch.no_grad():
        for parameter in params:
            gradient = parameter.grad
            if gradient is None:
                continue
            parameter -= rate * gradient
            parameter.grad = None


def train_step(x, y, params, rate):
    loss = forward_loss(x, y, params[0], params[1], params[2], params[3])
    loss.backward()
    descend(params, rate)
    return loss.item()


def preferred_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    rows = 20000
    cols = 16
    device = preferred_device()

    torch.manual_seed(0)
    raw = torch.randn(rows * cols).tolist()
    features = [0.0] * (rows * cols * 2)

    started = time.perf_counter()
    checksum = standardize(raw, features, rows, cols)
    prep_ms = (time.perf_counter() - started) * 1000.0

    x = torch.tensor(features, dtype=torch.float32).reshape(rows, cols * 2).to(device)
    y = torch.randn(rows, 1).to(device)
    params = [
        torch.randn(cols * 2, 32, device=device, requires_grad=True),
        torch.zeros(32, device=device, requires_grad=True),
        torch.randn(32, 1, device=device, requires_grad=True),
        torch.zeros(1, device=device, requires_grad=True),
    ]
    with torch.no_grad():
        params[0] *= 0.1
        params[2] *= 0.1

    started = time.perf_counter()
    first = 0.0
    last = 0.0
    for index in range(100):
        last = train_step(x, y, params, 0.02)
        if not index:
            first = last
    train_ms = (time.perf_counter() - started) * 1000.0

    print(f"# device: {device}")
    print(f"# native prep: {getattr(standardize, '__ppy_native__', None) is not None}")
    print(f"# aten region: {getattr(forward_loss, '__ppy_region__', False)}")
    print(f"prep  {prep_ms:8.1f} ms   checksum={checksum:.6f}")
    print(f"train {train_ms:8.1f} ms   loss {first:.4f} -> {last:.4f}")


main()
