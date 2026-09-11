# Examples

Each folder holds one example and a `README.md` explaining it. Code carries no
comments; the explanation lives in the markdown, and every README ends with
the commands to run and what each one prints (`record_outputs.py` keeps the
second in step with the first).

Where a folder holds both `<name>.py` and `<name>.ppy`, the `.ppy` is **exactly**
what `ppy convert <name>.py` writes — `ppy migrate` for the folder whose README
says so. `verify_conversions.py` regenerates every one of them and fails on any
difference, so a hand edit to a file that claims to be generated is caught
rather than believed.

Seven folders carry a `compare/` directory: the same work written for other
tools -- Numba, Cython, NumPy, numexpr, JAX, PyTorch, CuPy, Taichi, Mojo,
Codon, Rust, C, CUDA C -- each the way its tool wants it, and a section of the README that
puts the timings side by side and says what each port asked for.
`compare.py` is the harness: it runs every program several times, refuses to
print a table until every one of them prints the same answers, and reports
the mean and spread. The site collects those sections on one page.

| | |
|---|---|
| `01_basics` | fixed-width markers, purity, per-function opt levels |
| `02_arbitrary_precision` | machine words, and the fallback when they overflow |
| `03_effects_and_contracts` | what `@ppy.pure` refuses |
| `04_classes` | classes with statically known fields |
| `05_numpy` | elementwise NumPy fused into one kernel |
| `06_pydantic` | models keep their runtime validation |
| `07_parallel` | splitting a fused kernel, bit-identically; against NumPy, numexpr, Numba, JAX |
| `08_native_data` | which values have a native representation |
| `09_torch` | a function of tensor ops as one C++ ATen region |
| `10_narrowing` | every narrowing form the checker understands |
| `11_numerics` | floor division, remainder sign, overflow; what Numba, Codon, Mojo, C, Rust print instead |
| `12_buffers_and_jit` | borrowed buffers, `@ppy.fastmath`, specialization; against Numba, Cython, NumPy, C |
| `13_value_classes` | an all-scalar dataclass with no boxed form |
| `14_tuples` | fixed tuples as scalar ABI atoms |
| `15_algorithms` | eight compute kernels against C, and six judge problems in `15a`–`15f` read from stdin |
| `16_dynamic` | explicit dynamic boundaries |
| `17_containers` | element inference, aliasing, local mutation |
| `18_errors` | exception behavior as part of the contract |
| `19_strings` | string work stays on CPython, and says so |
| `20_inventory` | untyped Python that converts cleanly |
| `21_training_torch` | a torch MLP: native preprocessing plus an ATen region |
| `22_training_jax` | the same with JAX |
| `23_inference` | how far inference reaches on untyped input |
| `24_interop` | a plain `.py` importing a `.ppy` module |
| `25_jax_export` | build-time export to StableHLO |
| `26_project` | a multi-module project as one call graph |
| `27_uvicorn` | serving over Uvicorn: a raw ASGI app, and a converted FastAPI service |
| `28_threads` | a native region releasing the GIL |
| `29_flax` | a Flax/optax training loop, converted and strict-checked |
| `30_migrate` | a deliberately dynamic legacy script rewritten by `ppy migrate` into strict PPY |
| `31_torchrun` | a trainer under `torchrun` and `accelerate launch`: `import ppy` serves its kernels natively on every rank |
| `32_native_memory` | typed pointers, stack memory, a `libm` binding, and a function exported as a C symbol |
| `33_simd_and_cpu` | vector lanes with `ppy.simd`, and the machine as a facade with `ppy.cpu` |
| `34_atomics_and_threads` | `ppy.atomic` and `ppy.concurrent`: counters, a mutex, a condition, four workers |
| `35_parallel_range` | `parallel.range` loops, reductions that keep their order, and the parallel backends |
| `36_autodiff` | `ppy.grad` and `value_and_grad`, the same rule table on every path; against JAX, PyTorch |
| `37_aio` | an echo server and its client as `ppy.aio` coroutines, native runtime or asyncio |
| `38_cuda` | a saxpy and a block reduction in `ppy.cuda`: reference launch, PTX, CUDA and HIP source; against CuPy, Numba, Mojo, CUDA C |
| `39_xla` | `@xla.jit` functions as StableHLO, run through PJRT where a device is present |
| `40_generics` | type parameters, bounds, and Protocols, monomorphized in native code |
| `41_columnar` | pandas Series expressions fused as one kernel over the columnar dialect |
| `42_toolbox` | `ppy emit`, `ppy inspect --stage`, `--report-opt`, `--sanitize`, `--profile`, and `--pgo` on one program |
| `43_regex` | regular expressions over byte buffers, compiled to native matchers; against CPython's `re`, Rust's `regex` |

A folder of related problems keeps them in numbered subfolders, and every
runner reaches them: `15_algorithms/15a_nqueens` and its five siblings are
separate programs with their own README, C reference, and measurements.

## Checking all of it

```bash
python run_all.py             # every example x 3 paths, outputs compared
python verify_conversions.py  # every .ppy regenerated from its .py, and linted
python lint_all.py            # pylint over every file, source and converted
```

`verify_conversions.py` also checks that each folder's README says whether its
`.ppy` is generated or hand-written, and that the claim matches what is on disk.
It fails if a conversion introduces a pylint finding the source did not have.

`run_all.py` normalizes what is *measured* rather than computed — wall clock,
speedup ratios, float deltas, column padding — and skips lines an example marks
with `# ` as path-specific, so it compares answers and not benchmarks.
