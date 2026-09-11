"""Where does a data-parallel loss go wrong on real devices? Each stage against NumPy.

    python scripts/cloud/sharding_probe.py

Two or more devices: a batch sharded over a mesh, the same small MLP the
multi-GPU example trains, and every intermediate -- the local matmul, the
squared residual, its global mean, the gradient of the mean -- compared
with a single-device run and with NumPy, so a collective that answers
wrong is caught at the operation that used it.
"""

from __future__ import annotations

import sys


def main() -> int:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = jax.devices()
    print("devices", devices, "backend", jax.default_backend())
    if len(devices) < 2:
        print("requires >= 2 devices")
        return 2
    rng = np.random.default_rng(0)
    rows, cols, hidden = 4096, 32, 32
    x_np = rng.standard_normal((rows, cols)).astype(np.float32)
    y_np = rng.standard_normal((rows, 1)).astype(np.float32)
    w1_np = (rng.standard_normal((cols, hidden)) * 0.1).astype(np.float32)
    w2_np = (rng.standard_normal((hidden, 1)) * 0.1).astype(np.float32)
    b1_np = np.zeros((hidden,), np.float32)
    b2_np = np.zeros((1,), np.float32)

    def loss_np():
        h = np.maximum(x_np @ w1_np + b1_np, 0.0)
        p = h @ w2_np + b2_np
        return float(np.mean((p - y_np) ** 2))

    @jax.jit
    def forward_loss(x, y, w1, b1, w2, b2):
        hidden_ = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)
        predicted = jnp.dot(hidden_, w2) + b2
        residual = predicted - y
        return jnp.mean(residual * residual)

    grad = jax.jit(jax.value_and_grad(forward_loss, argnums=(2, 3, 4, 5)))
    expected = loss_np()
    print(f"numpy loss {expected:.6f}")
    for label, device_list in (("single", devices[:1]), ("sharded", devices)):
        m = Mesh(device_list, ("batch",))
        sh, rep = NamedSharding(m, PartitionSpec("batch")), NamedSharding(m, PartitionSpec())
        x, y = jax.device_put(x_np, sh), jax.device_put(y_np, sh)
        w1, b1, w2, b2 = (jax.device_put(a, rep) for a in (w1_np, b1_np, w2_np, b2_np))
        print(f"[{label}] x sharding {x.sharding} shape {x.shape}")
        print(f"[{label}] sum(x) {float(jnp.sum(x)):.4f} numpy {float(x_np.sum()):.4f}")
        squares = float(jnp.mean(x * x))
        print(f"[{label}] mean(x*x) {squares:.6f} numpy {float((x_np * x_np).mean()):.6f}")
        loss = forward_loss(x, y, w1, b1, w2, b2)
        print(f"[{label}] loss {float(loss):.6f} (sharding {loss.sharding}) numpy {expected:.6f}")
        value, grads = grad(x, y, w1, b1, w2, b2)
        norm = float(jnp.linalg.norm(grads[2]))
        print(f"[{label}] value_and_grad loss {float(value):.6f}; |grad w2| {norm:.6f}")
        hidden_ = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)
        print(f"[{label}] hidden sharding {hidden_.sharding}; sum {float(jnp.sum(hidden_)):.4f}")
        residual = jnp.dot(hidden_, w2) + b2 - y
        total = float(jnp.sum(residual * residual))
        print(f"[{label}] residual sum sq {total:.4f} / rows {rows} = {total / rows:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
