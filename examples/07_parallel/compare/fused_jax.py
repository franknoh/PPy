"""The same expression under JAX on the CPU: `jax.jit` fuses it through XLA."""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def fused(a, b):
    return (a * b + a) * (b - a) + a * 0.5 - b * 0.25


@jax.jit
def sum_of_squares(a):
    return jnp.sum(a * a)


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
    x = jnp.linspace(0.0, 10.0, 8_000_000)
    y = jnp.linspace(1.0, 5.0, 8_000_000)
    fused(x, y).block_until_ready()
    sum_of_squares(x).block_until_ready()
    out = timed("fused", lambda: fused(x, y).block_until_ready())
    print(f"{float(out[7]):.12f} {float(out[-1]):.12f}")
    print(f"{timed('sum of squares', lambda: float(sum_of_squares(x).block_until_ready())):.3f}")


main()
