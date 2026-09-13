# GPT-2 XL: one region per block

A transformer block is seventeen weights and one straight line of tensor
operations, which is exactly what an ATen region is. The block is written
once and `@ppy.opt(3)` compiles it into a single C++ function; forty-eight
calls to it are GPT-2 XL, and the loop between them is all the Python that
is left.

```python
@ppy.opt(3)
def block(x, ln1_w, ln1_b, q_w, q_b, ..., batch, length, heads, head_dim, width, eps):
    h = torch.layer_norm(x, (width,), ln1_w, ln1_b, eps)
    q = F.linear(h, q_w, q_b).reshape((batch, length, heads, head_dim)).transpose(1, 2)
    k = F.linear(h, k_w, k_b).reshape((batch, length, heads, head_dim)).transpose(1, 2)
    v = F.linear(h, v_w, v_b).reshape((batch, length, heads, head_dim)).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    merged = a.transpose(1, 2).reshape((batch, length, width))
    attended = torch.add(x, F.linear(merged, o_w, o_b))
    n = torch.layer_norm(attended, (width,), ln2_w, ln2_b, eps)
    f = F.gelu(F.linear(n, fc_w, fc_b), approximate="tanh")
    return torch.add(attended, F.linear(f, proj_w, proj_b))
```

## What it becomes

The dimensions are parameters of the region, not constants folded into it,
so one compiled function serves every batch and every length:

```cpp
at::Tensor ppy_region_gpt2_block(const at::Tensor& x, const at::Tensor& ln1_w, ...,
                                 int64_t batch, int64_t length, int64_t heads,
                                 int64_t head_dim, int64_t width, double eps) {
    auto h = at::layer_norm(x, {width}, ln1_w, ln1_b, eps);
    auto q = ((at::linear(h, q_w, q_b)).reshape({batch, length, heads, head_dim})).transpose(1, 2);
    auto k = ((at::linear(h, k_w, k_b)).reshape({batch, length, heads, head_dim})).transpose(1, 2);
    auto v = ((at::linear(h, v_w, v_b)).reshape({batch, length, heads, head_dim})).transpose(1, 2);
    auto a = at::scaled_dot_product_attention(q, k, v, {}, 0.0, true);
    auto merged = ((a).transpose(1, 2)).reshape({batch, length, width});
    auto attended = at::add(x, at::linear(merged, o_w, o_b));
    auto n = at::layer_norm(attended, {width}, ln2_w, ln2_b, eps);
    auto f = at::gelu(at::linear(n, fc_w, fc_b), "tanh");
    return at::add(attended, at::linear(f, proj_w, proj_b));
}
```

The region does not reimplement any of it. `at::scaled_dot_product_attention`
is the same call `F.scaled_dot_product_attention` makes, so it reaches the
same flash-attention kernel; `at::linear` reaches the same cuBLAS GEMM.
Every `at::` call goes through the dispatcher, which is why autograd is
unchanged and the training step below differentiates through the region
exactly as it does through the Python. What the region removes is the ten
Python round trips per block -- four hundred and eighty per forward pass.

## Compared with PyTorch eager and `torch.compile` at model scale

<!-- compare:start -->
| | PPY ATen regions | PyTorch eager | `torch.compile` | `torch.compile` reduce-overhead |
|---|---:|---:|---:|---:|
| forward b1 s32 | 8.78 ± 0.12 | 10.77 ± 0.08 | **7.46 ± 0.28** | 13.36 ± 0.02 |
| forward b1 s128 | 9.24 ± 0.18 | 11.06 ± 0.08 | **7.88 ± 0.28** | 14.11 ± 0.01 |
| forward b1 s512 | 14.87 ± 0.01 | 14.90 ± 0.01 | **14.74 ± 0.07** | 22.86 ± 0.02 |
| forward b8 s512 | **95.33 ± 0.30** | 95.50 ± 0.18 | 95.55 ± 0.26 | 102.96 ± 0.66 |
| decode 128 tokens | 1204.24 ± 28.34 | 1484.26 ± 17.63 | **1057.01 ± 41.66** | 1658.16 ± 1.84 |
| training step b4 s512 | 136.62 ± 0.23 | 136.73 ± 0.21 | **135.58 ± 0.25** | 143.53 ± 0.67 |
<!-- compare:end -->

Milliseconds per pass of the whole model, bfloat16, best of ten in each of
five processes; the decode row is a 128-token prompt and then 128 tokens one
at a time out of the cache, timed without the prefill. Every column is the
same forty-eight blocks on the same weights, and all four agree on the
answer to three decimals -- which is what the agreement is checked on,
because the greedy token is not stable under any of this: on weights drawn
from a generator the top two logits of a position are usually a tie, and
fusing a reduction differently flips it. The decode loop feeds the next
token of a fixed sequence rather than the model's own argmax, for the same
reason: sampling would send the four columns down four different sequences.

The rows are one axis: how much of the wall clock is the Python between
operators. At `b1 s32` the GEMMs are small and reading the weights is about
three milliseconds, so the rest is dispatch -- and removing it is worth 18%
to the region and 31% to `torch.compile`. At `b8 s512` the same forty-eight
blocks are 95 ms of cuBLAS and the Python is a rounding error; the first
three columns are the same number, and they should be. The training step is
the same story with a backward pass on it.

Decode is that axis at its far end. One token through forty-eight blocks is
6144 region calls for 128 tokens, and almost no arithmetic per call: the
region takes 19% off eager, `torch.compile` 29%, and the two are within 14%
of each other. It is the row where removing the interpreter is worth the
most, and it is still the row where a fusing compiler wins.

That is the honest shape of it. A region is not a fusing compiler: it
removes the interpreter between the operators and leaves the kernels where
they were, so where Inductor's own win is also interpreter overhead the two
land close, and where Inductor could fuse -- the two `layer_norm`s, the
`gelu`, the residual adds -- there is not enough of it in a model that is
98% GEMM for the fusion to show at this size. What the columns do not show
is what each cost to get: the region is one C++ translation unit, 19 s to
build once and cached on disk for every process after; `torch.compile`
spends about 18 s on the first shape's forward graph in every new process,
and several minutes to reach all five shapes and the backward with a cold
Inductor cache.

`reduce-overhead` -- the CUDA-graphs mode -- is slower here in every row
than plain `torch.compile`, and slower than eager too. Reproducibly so, the
same numbers in its own process with a single shape rather than five, and
with no warning from Inductor that it declined to capture.

It also does not run a KV cache as written. A cache is a tensor the caller
keeps across invocations, which is exactly the memory CUDA graphs reuse, and
the decode loop fails outright -- *accessing tensor output of CUDAGraphs
that has been overwritten by a subsequent run* -- until the counterpart
announces each invocation with `cudagraph_mark_step_begin()` **and** sets
`cudagraph_trees_generation_cloning = "user_visible"`, which are the two
things its own error message asks for. The cloning that second switch turns
on is part of what the last column costs. The training step needed the same
care for the same reason: gradients are set to `None` before each backward
rather than accumulated, which is what an optimizer does anyway.

The PPY column is `ppy <file>` -- the staged Python that loads the compiled
regions -- not `ppy run`. Compiling the driver natively has nothing to do
here: outside the regions the program is forty-eight calls and a loop.

NVIDIA GeForce RTX 4090 24 GB, driver 580.126.20, CUDA 13.0; PyTorch
2.14.0+cu130 on CPython 3.13.15, on a rented Pod. Not the machine the other
tables come from: `scripts/compare_docs.py` skips this comparison unless
`PPY_TORCH_CUDA_PYTHON` names a PyTorch with a device, so the bench runner
leaves it alone, and `scripts/cloud/runpod_bench.py` is what re-measures it.

## Run it

`gpt2.ppy` is the same model with the same region. It is GPT-2 XL where
there is a device for it and a six-layer stack where there is not, so the
program below is the CPU shape; the table above is what it does on the
4090. `ppy run` compiles the driver through LLVM as well, which has nothing
to gain on a program that is a loop and forty-eight calls; the three paths
print the same number and the same answer.

```bash
python  gpt2.ppy
ppy     gpt2.ppy
ppy run gpt2.ppy
```

<!-- outputs:start -->
## What it prints

**`python  gpt2.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: False
# 6 layers, 12 heads, width 768, batch 1, length 128
mean=0.004 absmean=0.439 meansq=0.301
# forward: 22.246 ms
cached decode matches the whole forward: True
```

**`ppy     gpt2.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
# 6 layers, 12 heads, width 768, batch 1, length 128
mean=0.004 absmean=0.439 meansq=0.301
# forward: 22.646 ms
cached decode matches the whole forward: True
```

**`ppy run gpt2.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
# 6 layers, 12 heads, width 768, batch 1, length 128
mean=0.004 absmean=0.439 meansq=0.301
# forward: 24.435 ms
cached decode matches the whole forward: True
```

<!-- outputs:end -->

## The cache the region keeps

A region may hand back several tensors, which is what a KV cache needs: the
block returns its output beside the key and value it just grew, so the cache
never leaves the region to be reassembled in Python.

```python
@ppy.opt(3)
def step(x, k_cache, v_cache, ..., batch, length, ..., causal: bool
         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
    k_all = torch.cat((k_cache, k), 2)
    v_all = torch.cat((v_cache, v), 2)
    a = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=causal)
    ...
    return torch.add(attended, F.linear(f, proj_w, proj_b)), k_all, v_all
```

It becomes a `std::tuple<at::Tensor, at::Tensor, at::Tensor>`, which pybind11
hands Python as an ordinary tuple. `causal` is a `bool` parameter rather than
a constant folded in, and `length` an `int64_t`, so **one** compiled function
prefills a prompt under a causal mask and then decodes a token at a time out
of the cache it built. `gpt2.ppy` prints whether the two agree, because that
is the check that matters: the last position of a whole forward pass, and the
same position reached one token at a time, are the same logits.

## What a region will not do yet

Everything a region reaches is straight-line: no loop, no branch, and no
tensor it did not receive as a parameter. The cache above grows by `cat`
rather than being written into a preallocated buffer, because a region has
no in-place write; that costs a copy per token, the same copy in every
column here.

Read on: [PyTorch ATen regions](../09_torch/README.md) ·
[Plugins: PyTorch](../../docs/internals/plugins.md) ·
[Training with torch](../21_training_torch/README.md)

`gpt2.ppy` and the programs under `compare/` are hand-written; there is no
`.py` source and no conversion step.
