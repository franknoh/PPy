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
| forward b1 s32 | 9.22 ± 0.16 | 11.29 ± 0.21 | **8.14 ± 0.31** | 13.61 ± 0.06 |
| forward b1 s128 | 9.65 ± 0.24 | 11.60 ± 0.29 | **8.53 ± 0.12** | 14.33 ± 0.03 |
| forward b1 s512 | 15.25 ± 0.02 | 15.30 ± 0.01 | **15.15 ± 0.02** | 23.43 ± 0.03 |
| forward b8 s512 | **96.74 ± 0.21** | 96.99 ± 0.10 | 96.94 ± 0.12 | 104.44 ± 0.10 |
| training step b4 s512 | 139.05 ± 0.08 | 139.35 ± 0.12 | **137.65 ± 0.25** | 145.18 ± 0.10 |
<!-- compare:end -->

Milliseconds per pass of the whole model, bfloat16, best of ten in each of
five processes. Every column is the same forty-eight blocks on the same
weights, and all four agree on the answer to three decimals -- which
is what the agreement is checked on, because the greedy token is not stable
under any of this: on weights drawn from a generator the top two logits of
a position are usually a tie, and fusing a reduction differently flips it.

The rows are one axis: how much of the wall clock is the Python between
operators. At `b1 s32` the GEMMs are small and reading the weights is about
three milliseconds, so the other six to eight are dispatch -- and removing
it is worth 18% to the region and 28% to `torch.compile`. At `b8 s512` the
same forty-eight blocks are 97 ms of cuBLAS and the Python is a rounding
error; the first three columns are the same number, and they should be.
The training step is the same story with a backward pass on it.

That is the honest shape of it. A region is not a fusing compiler: it
removes the interpreter between the operators and leaves the kernels where
they were, so where Inductor's own win is also interpreter overhead the two
land within 10% of each other, and where Inductor could fuse -- the two
`layer_norm`s, the `gelu`, the residual adds -- there is not enough of it
in a model that is 98% GEMM for the fusion to show at this size. What the
columns do not show is what each cost to get: the region is one C++
translation unit, 19 s to build once and cached on disk for every process
after; `torch.compile` spends about 18 s on the first shape's forward
graph in every new process, and roughly three minutes to reach all four
shapes and the backward with a cold Inductor cache.

`reduce-overhead` -- the CUDA-graphs mode -- is slower here in every row
than plain `torch.compile`, and slower than eager too: 13.61 ms against
8.14 and 11.29 at `b1 s32`. Reproducibly so, the same numbers in its own
process with a single shape rather than four, and with no warning from
Inductor that it declined to capture. It is the mode whose gains a
launch-bound model was most likely to show, and on this one it did not. It
also refuses a training step that accumulates into `.grad`, so the
counterparts set the gradients to `None` before each backward -- which is
what an optimizer does anyway, and what the error message asks for.

The PPY column is `ppy <file>` -- the staged Python that loads the compiled
regions -- not `ppy run`. Compiling the driver natively has nothing to do
here: outside the regions the program is forty-eight calls and a loop.

NVIDIA GeForce RTX 4090 24 GB, driver 580.126.20, CUDA 13.0; PyTorch
2.14.0+cu130 on CPython 3.13.15, on a rented Pod. Not the machine the other
tables come from: `scripts/compare_docs.py` skips this comparison unless
`PPY_TORCH_CUDA_PYTHON` names a PyTorch with a device, so the bench runner
leaves it alone, and `scripts/cloud/runpod_bench.py` is what re-measures
it.

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
# forward: 24.859 ms
```

**`ppy     gpt2.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
# 6 layers, 12 heads, width 768, batch 1, length 128
mean=0.004 absmean=0.439 meansq=0.301
# forward: 23.898 ms
```

**`ppy run gpt2.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
# 6 layers, 12 heads, width 768, batch 1, length 128
mean=0.004 absmean=0.439 meansq=0.301
# forward: 24.954 ms
```

<!-- outputs:end -->

## What a region will not do yet

A region returns one tensor. A block cannot hand back its own key and value
projections beside its output, so incremental decoding with a KV cache --
where the Python between operators costs the most -- stays in Python for
now. Everything a region does reach is straight-line: no loop, no branch,
and no tensor it did not receive as a parameter.

Read on: [PyTorch ATen regions](../09_torch/README.md) ·
[Plugins: PyTorch](../../docs/internals/plugins.md) ·
[Training with torch](../21_training_torch/README.md)

`gpt2.ppy` and the programs under `compare/` are hand-written; there is no
`.py` source and no conversion step.
