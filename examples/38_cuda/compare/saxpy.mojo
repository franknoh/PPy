"""saxpy over sixteen million doubles and a per-block max: Mojo 1.0 kernels through MAX's DeviceContext."""

from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.memory import AddressSpace, UnsafePointer, bitcast, stack_allocation
from std.time import perf_counter_ns

comptime N = 1 << 24


def fma(a: Float64, x: Float64, y: Float64) -> Float64:
    return a * x + y


def saxpy(n: Int64, a: Float64, x: UnsafePointer[Float64, MutAnyOrigin], y: UnsafePointer[Float64, MutAnyOrigin]):
    var i = Int(global_idx.x)
    if Int64(i) < n:
        y[i] = fma(a, x[i], y[i])


def block_max(x: UnsafePointer[Float64, MutAnyOrigin], result: UnsafePointer[Float64, MutAnyOrigin]):
    var parked = stack_allocation[64, Float64, address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    parked[tid] = x[Int(global_idx.x)]
    barrier()
    var mine = parked[tid]
    # The shuffle has no Float64 form here: the bits go across as an Int64.
    var other = bitcast[DType.float64, 1](shuffle_xor(mine.to_bits[DType.uint64](), 1))
    mine = other if other > mine else mine
    parked[tid] = mine
    barrier()
    if tid == 0:
        var best = parked[0]
        for k in range(1, Int(block_dim.x)):
            best = parked[k] if parked[k] > best else best
        result[Int(block_idx.x)] = best


def main() raises:
    var ctx = DeviceContext()
    var hx = List[Float64](length=N, fill=0.0)
    var hy = List[Float64](length=N, fill=1.0)
    var hv = List[Float64](length=N, fill=0.0)
    var hout = List[Float64](length=N // 64, fill=0.0)
    for i in range(N):
        hx[i] = Float64(i)
        hv[i] = Float64((i * 37) % 101)
    var x = ctx.enqueue_create_buffer[DType.float64](N)
    var y = ctx.enqueue_create_buffer[DType.float64](N)
    var v = ctx.enqueue_create_buffer[DType.float64](N)
    var result = ctx.enqueue_create_buffer[DType.float64](N // 64)
    ctx.enqueue_copy(x, hx.unsafe_ptr())
    ctx.enqueue_copy(y, hy.unsafe_ptr())
    ctx.enqueue_copy(v, hv.unsafe_ptr())
    ctx.enqueue_function[saxpy](Int64(N), 2.0, x.unsafe_ptr(), y.unsafe_ptr(), grid_dim=(N + 255) // 256, block_dim=256)
    ctx.enqueue_function[block_max](v.unsafe_ptr(), result.unsafe_ptr(), grid_dim=N // 64, block_dim=64)
    ctx.enqueue_copy(hy.unsafe_ptr(), y)
    ctx.enqueue_copy(hout.unsafe_ptr(), result)
    ctx.synchronize()
    var total = 0.0
    for i in range(N):
        total += hy[i]
    var best = 0.0
    for i in range(N // 64):
        best = hout[i] if hout[i] > best else best
    print(total, best)
    var best_saxpy = 1e18
    var best_block = 1e18
    for _ in range(10):
        var started = perf_counter_ns()
        ctx.enqueue_function[saxpy](Int64(N), 2.0, x.unsafe_ptr(), y.unsafe_ptr(), grid_dim=(N + 255) // 256, block_dim=256)
        ctx.synchronize()
        var took = Float64(perf_counter_ns() - started) / 1e6
        best_saxpy = took if took < best_saxpy else best_saxpy
        started = perf_counter_ns()
        ctx.enqueue_function[block_max](v.unsafe_ptr(), result.unsafe_ptr(), grid_dim=N // 64, block_dim=64)
        ctx.synchronize()
        took = Float64(perf_counter_ns() - started) / 1e6
        best_block = took if took < best_block else best_block
    print("# saxpy:", best_saxpy, "ms")
    print("# block_max:", best_block, "ms")
    best_saxpy = 1e18
    best_block = 1e18
    for _ in range(10):
        var started = perf_counter_ns()
        ctx.enqueue_copy(x, hx.unsafe_ptr())
        ctx.enqueue_copy(y, hy.unsafe_ptr())
        ctx.enqueue_function[saxpy](Int64(N), 2.0, x.unsafe_ptr(), y.unsafe_ptr(), grid_dim=(N + 255) // 256, block_dim=256)
        ctx.enqueue_copy(hy.unsafe_ptr(), y)
        ctx.synchronize()
        var took = Float64(perf_counter_ns() - started) / 1e6
        best_saxpy = took if took < best_saxpy else best_saxpy
        started = perf_counter_ns()
        ctx.enqueue_copy(v, hv.unsafe_ptr())
        ctx.enqueue_function[block_max](v.unsafe_ptr(), result.unsafe_ptr(), grid_dim=N // 64, block_dim=64)
        ctx.enqueue_copy(hout.unsafe_ptr(), result)
        ctx.synchronize()
        took = Float64(perf_counter_ns() - started) / 1e6
        best_block = took if took < best_block else best_block
    print("# saxpy with copies:", best_saxpy, "ms")
    print("# block_max with copies:", best_block, "ms")
