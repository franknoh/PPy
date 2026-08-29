# Examples

Each folder holds one example and a `README.md` explaining it. Code carries no
comments; the explanation lives in the markdown.

Where a folder holds both `<name>.py` and `<name>.ppy`, the `.ppy` is **exactly**
what `ppy convert <name>.py` writes. `verify_conversions.py` regenerates every
one of them and fails on any difference, so a hand edit to a file that claims to
be generated is caught rather than believed.

| | |
|---|---|
| `01_basics` | fixed-width markers, purity, per-function opt levels |
| `02_arbitrary_precision` | machine words, and the fallback when they overflow |
| `03_effects_and_contracts` | what `@ppy.pure` refuses |
| `04_classes` | classes with statically known fields |
| `05_numpy` | elementwise NumPy fused into one kernel |
| `06_pydantic` | models keep their runtime validation |
| `07_parallel` | splitting a fused kernel, bit-identically |
| `08_native_data` | which values have a native representation |
| `09_torch` | a function of tensor ops as one C++ ATen region |
| `10_narrowing` | every narrowing form the checker understands |
| `11_numerics` | floor division, remainder sign, overflow |
| `12_buffers_and_jit` | borrowed buffers, `@ppy.fastmath`, specialization |
| `13_value_classes` | an all-scalar dataclass with no boxed form |
| `14_tuples` | fixed tuples as scalar ABI atoms |
| `15_algorithms` | competitive-programming kernels, 10x to 150x |
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
