"""The same derivatives under JAX: `jax.grad`, and `vmap` over the starting points, jitted."""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def f(x, y):
    return jnp.sin(x) * y + x * x


def g(x):
    return jnp.exp(-x * x) * jnp.cos(3.0 * x)


df = jax.grad(f)
both = jax.value_and_grad(f, argnums=1)
dg = jax.grad(g)


def newton(x):
    def step(_, x):
        return x - g(x) / dg(x)

    return jax.lax.fori_loop(0, 6, step, x)


newton_all = jax.jit(jax.vmap(newton))


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
    v, grad_y = both(0.7, 2.0)
    print(f"{float(df(0.7, 2.0)):.9f} {float(v * 2.0 + grad_y):.9f}")
    starts = 0.4 + jnp.arange(100_000) * 1e-6
    newton_all(starts).block_until_ready()
    roots = timed("newton, 100k starts", lambda: newton_all(starts).block_until_ready())
    print(f"{float(jnp.mean(roots)):.9f} {float(jnp.max(jnp.abs(jax.vmap(g)(roots)))):.0e}")


main()
