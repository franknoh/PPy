# Migrating a real project

`ppy migrate` converts Python to PPY. This page is about *what* to hand it,
because on a real codebase the answer is not "the project".

## Where the speed is

PPY makes three things fast: loops over numbers, loops over buffers, and the
boundary between them and the rest of the program. It leaves everything
else exactly as fast as CPython, because everything else *is* CPython —
the optimized backend runs it, the launcher embeds it. A 6,000-line training
loop that calls into PyTorch, drives an emulator over a socket, and wraps
every step in `try` is not going to get faster by being `.ppy`; its time is
in PyTorch and the emulator. The 700-line file next to it that computes
rewards from arrays of floats is going to get 10–100× faster, and it is
usually the file with the fewest dynamic features too.

So the unit of migration is the **kernel**, not the repository:

1. **Profile.** `python -m cProfile -s cumulative train.py` (or whatever the
   entry is) for a representative run. The functions that matter are the
   ones with real self-time doing arithmetic — not the ones waiting on a
   library.
2. **Find their files.** Usually two or three modules hold the numeric
   work: reward functions, observation builders, geometry, physics, scoring.
   They import `math` and `numpy`, take arrays or floats in, and give
   numbers back.
3. **Migrate those files, and only those.**

   ```bash
   ppy migrate train/reward.py --dry-run     # see what it would write
   ppy migrate train/reward.py --in-place    # write reward.ppy, remove reward.py
   ppy check train/reward.ppy                # what is left to type
   ```

4. **Leave the orchestration as `.py`.** It imports the kernels through the
   loader `import ppy` installs, and nothing about it changes:

   ```python
   import ppy  # installs the .ppy loader

   from train import reward  # train/reward.ppy, as ordinary Python

   score = reward.total(obs, prev, action)
   ```

   Under plain CPython that call runs the Python body. Under `ppy run` or a
   `ppy build` launcher, `reward.total` is native and the orchestration
   never knows.

[`examples/26_project`](../examples/26_project/README.md) is this shape at
toy scale — a `.ppy` kernel behind a `.py` application — and
[`examples/24_interop`](../examples/24_interop/README.md) is the import hook
on its own.

## Reading the report

Migration reports in three registers, and the order to read them is the
reverse of the order they print.

**`E1304` — cannot infer a stable type for parameter `x`.** These are the
to-do list. Each names a parameter no call site typed and no arithmetic
pinned down; annotate it (`x: float`) and re-run. On a kernel file there are
usually a handful, and they are usually the entry points, because nothing
inside the file calls them. Annotating the entry points is most of the work.

**`W2006` — *n* further errors only restated a type that could not be
resolved and are not shown.** Before this line existed, every place an
untyped value flowed to reported an error of its own: ten call sites of one
untyped function were ten errors, none of which said which parameter to
fix. Now they are one count, and the line names the calls whose signatures
the analysis does not know — a library without a plugin or stub, or a
method on an unknown object. Fix the `E1304`s first; most of the count goes
with them.

**Everything else** is a finding about the code itself. `E1206: self.buffer
may be None` means the field is `None` in `__init__` and something else
later, and the read has no `is not None` in front of it — which is true, and
which the native path also needs to know. `E1301` with two concrete types
on either side is a real mismatch.

If the count on the `W2006` line is large and the `E1304`s are few, the
unknowns are coming from a library: `math.tanh` that the stdlib model does
not know yet, `np.something` outside the NumPy plugin's surface, a
`ctypes` call. Those go behind `ppy.dynamic` — or into the model, which is a
pull request.

## What does not convert, and should not

- `eval`/`exec`, `globals()` in a function, monkey-patching: `E15xx`.
  `ppy migrate` converts the module faithfully and marks the site; `ppy
  check` will insist on a `ppy.dynamic` boundary around it. If the site is
  in a kernel, the kernel has a Python island in it and the loop containing
  it stays Python.
- A class whose fields are assigned in six different methods with six
  different types is a class, not a struct. It can be `.ppy`; it will not
  lower. That is fine — the loops that read its numbers can.
- `try`/`except` around the hot loop. The loop inside lowers; the handler
  is Python. Move the `try` outside the loop if the loop is the point.

## What to expect from the numbers

On a kernel file that types cleanly, `ppy run` gives the JIT and `ppy build`
gives an artifact; the [algorithms folder](../examples/15_algorithms/README.md)
has the measured spread between plain CPython, the native path, and C. On
the orchestration, expect nothing, and do not migrate it to find out.
