"""The same two expressions under `jax.jit` on the CPU: XLA fuses them."""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def normalize(x):
    scale = jnp.sqrt(jnp.sum(x * x))
    return x / scale


@jax.jit
def blend(a, b):
    return jnp.sin(a) * 2.0 + jnp.cos(b)


def timed(label, run):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        answer = run()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.2f} ms")
    return answer


def main():
    values = jnp.linspace(0.0, 10.0, 8_000_000)
    other = jnp.linspace(1.0, 5.0, 8_000_000)
    normalize(values).block_until_ready()
    blend(values, other).block_until_ready()
    normalized = timed("normalize", lambda: normalize(values).block_until_ready())
    mixed = timed("blend", lambda: blend(values, other).block_until_ready())
    print(f"{float(normalized[1]):.9f} {float(normalized[-1]):.9f}")
    print(f"{float(mixed[1]):.9f} {float(mixed[-1]):.9f}")


main()
