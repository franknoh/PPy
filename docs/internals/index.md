# Internals

How the compiler is put together. None of it is needed to write a program;
read it to change the compiler, to write a plugin, or to follow a decision
about how something compiled all the way down.

<div class="grid cards" markdown>

-   **[Architecture](architecture.md)**

    ---

    The pipeline, the package map, the three-path invariant, the cache and
    incremental builds, the boundary, threads.

-   **[The IR](ir.md)**

    ---

    The typed SSA IR between analysis and every backend: dialects, passes,
    `.ppyir`, the linker, sanitizers, profiles.

-   **[Conversion and inference](conversion.md)**

    ---

    What `ppy convert` and `ppy migrate` infer and how: where types come
    from, protocol widening, `Final`, the strict gate.

-   **[Migrating a real project](migrating.md)**

    ---

    What to hand `ppy migrate` on a real codebase: profile, carve the
    kernels, leave the rest.

-   **[Plugins](plugins.md)**

    ---

    How each library integration works, and how to write one against the
    interface.

-   **[Where a solver fits](solver.md)**

    ---

    The two places an SMT solver earns its keep: proving overflow guards
    away and validating the optimizer.

</div>
