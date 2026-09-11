"""The same trainer with `forward_loss` under `torch.compile`: Inductor fuses the operators."""

import time

import torch


def standardize(raw: torch.Tensor) -> tuple[torch.Tensor, float]:
    mean = raw.mean(dim=1, keepdim=True)
    deviation = torch.sqrt(((raw - mean) ** 2).mean(dim=1, keepdim=True)) + 1e-8
    z = (raw - mean) / deviation
    interaction = z * z.roll(-1, dims=1)
    out = torch.cat([z, interaction], dim=1)
    return out, float(out.sum())


@torch.compile
def forward_loss(x, y, w1, b1, w2, b2):
    hidden = torch.relu(torch.add(torch.matmul(x, w1), b1))
    predicted = torch.add(torch.matmul(hidden, w2), b2)
    residual = torch.sub(predicted, y)
    return torch.mean(torch.mul(residual, residual))


def train_step(x, y, params, rate):
    loss = forward_loss(x, y, params[0], params[1], params[2], params[3])
    loss.backward()
    with torch.no_grad():
        for weight in params:
            if weight.grad is not None:
                weight -= rate * weight.grad
                weight.grad = None
    return loss.item()


def main():
    rows, cols = 20000, 16
    torch.manual_seed(0)
    torch.set_num_threads(8)
    raw = torch.randn(rows * cols).to(torch.float64).reshape(rows, cols)
    standardize(raw[:10])
    best_prep = 1e9
    for _ in range(5):
        started = time.perf_counter()
        features, checksum = standardize(raw)
        best_prep = min(best_prep, (time.perf_counter() - started) * 1000.0)
    x = features.to(torch.float32)
    y = torch.randn(rows, 1)
    torch.manual_seed(1)
    params = [
        torch.randn(cols * 2, 32, requires_grad=True),
        torch.zeros(32, requires_grad=True),
        torch.randn(32, 1, requires_grad=True),
        torch.zeros(1, requires_grad=True),
    ]
    with torch.no_grad():
        params[0] *= 0.1
        params[2] *= 0.1
    last = train_step(x, y, params, 0.02)
    best_forward = 1e9
    with torch.no_grad():
        for _ in range(200):
            started = time.perf_counter()
            loss = forward_loss(x, y, params[0], params[1], params[2], params[3]).item()
            best_forward = min(best_forward, (time.perf_counter() - started) * 1000.0)
    print(f"# standardize, 20000 rows: {best_prep:.2f} ms")
    print(f"# forward pass, per call: {best_forward:.4f} ms")
    print(f"{checksum:.4f} {last:.4f} {loss:.4f}")


main()
