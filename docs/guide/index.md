# Guide

A `.ppy` file is valid Python 3.12+. Everything PPY adds is carried by
annotations and decorators from the `ppy` package, all of which are inert at
runtime: under plain CPython the decorators return the function unchanged and
the markers are ordinary `typing.Annotated` aliases. What the compiler adds is
enforcement and speed, never behavior.

The pages below are the language, one topic each. Every namespace has a
Python reference implementation and a lowering that agrees with it, so a
program using any of them runs the same on every path.

<div class="grid cards" markdown>

-   **[The subset](subset.md)**

    ---

    What the compiler accepts, the compatibility policy, and the three
    absences of a type: unknown, `Any`, and `Dynamic`.

-   **[Directives and markers](directives.md)**

    ---

    `@ppy.pure`, `@ppy.native`, `@ppy.jit`, `@ppy.dynamic` and the rest;
    fixed-width integers, `Buffer[T]`, `Range`, ownership.

-   **[Native memory and FFI](native.md)**

    ---

    `ppy.native` pointers, stack allocation, `extern` and `export`, and
    `ppy.ffi` over them.

-   **[Lanes and the machine](simd-cpu.md)**

    ---

    `ppy.simd` vectors and `ppy.cpu` features, hints, and targets.

-   **[Atomics and threads](concurrency.md)**

    ---

    `ppy.atomic` with C11 orders; `ppy.concurrent` spawn, join, mutex,
    condition, barrier.

-   **[Parallel loops](parallel.md)**

    ---

    `parallel.range`, reductions, and what the body may do.

-   **[Derivatives](autodiff.md)**

    ---

    `ppy.grad` and `ppy.value_and_grad`, one rule table on every path.

-   **[Coroutines](aio.md)**

    ---

    `ppy.aio`: sockets and sleeps under asyncio or the native epoll loop.

-   **[GPU kernels](gpu.md)**

    ---

    `ppy.cuda` and `ppy.hip`: kernels in Python, PTX under `ppy run`.

-   **[XLA](xla.md)**

    ---

    `@xla.jit`: StableHLO from the compiler, run through PJRT.

-   **[Generics](generics.md)**

    ---

    Type parameters, bounds, monomorphization, static dispatch.

-   **[Effects and the three paths](effects.md)**

    ---

    The effect vocabulary, purity, and why three ways of running agree.

-   **[Reading input](input.md)**

    ---

    `ppy.input` reads lines, `ppy.scan` reads tokens, `ppy.read_*` fill a buffer.

-   **[Native lowering](native-lowering.md)**

    ---

    When a function gets a boundary, and what its parameters may be.

-   **[Regular expressions](regex.md)**

    ---

    A pattern compiled from a bytes literal, matched natively over a byte buffer.

</div>
