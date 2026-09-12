"""One JAX process per accelerator on one host: distributed initialization, then a collective.

    python scripts/cloud/multiprocess_smoke.py <process_id> <process_count>

The harness starts `process_count` of these at once, each pinned to one
local device through `local_device_ids`, with a coordinator on the loopback
address. Each asserts what `jax.process_count()`, `jax.process_index()`,
`jax.devices()`, and `jax.local_devices()` say, builds a global array from
its own shard, and reduces it across every process; the sum is checked
against arithmetic. A smoke test for the multi-node plumbing, on one node.
"""

from __future__ import annotations

import sys


def main() -> int:
    process_id, count = int(sys.argv[1]), int(sys.argv[2])
    port = sys.argv[3] if len(sys.argv) > 3 else "12355"
    import jax

    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{port}",
        num_processes=count,
        process_id=process_id,
        local_device_ids=[process_id],
    )
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    local = jax.local_devices()
    print(
        f"process {jax.process_index()}/{jax.process_count()}: local {local}, "
        f"global {len(jax.devices())} device(s) on {jax.default_backend()}",
        flush=True,
    )
    assert jax.process_count() == count, jax.process_count()
    assert jax.process_index() == process_id
    assert len(local) == 1 and all(d.platform != "cpu" for d in local), local
    assert len(jax.devices()) == count, jax.devices()
    mesh = Mesh(jax.devices(), ("x",))
    shard = np.full((4,), float(process_id + 1), dtype=np.float32)
    whole = jax.make_array_from_process_local_data(
        NamedSharding(mesh, PartitionSpec("x")), shard, global_shape=(4 * count,)
    )
    total = jax.jit(jnp.sum, out_shardings=NamedSharding(mesh, PartitionSpec()))(whole)
    total.block_until_ready()
    expected = float(sum(4 * (i + 1) for i in range(count)))
    assert abs(float(total) - expected) < 1e-5, (float(total), expected)
    print(f"process {process_id}: collective sum {float(total)} == {expected}: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
