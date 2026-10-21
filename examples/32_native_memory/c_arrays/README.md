# Borrowed C arrays

`array_api.ppy` is hand-written. Its exported functions use the native type
system directly. The C caller owns `int ice[8][8]`; PPy changes that same
allocation through `native.ptr[i32, 8]`, whose C ABI is `int32_t (*)[8]`.
`game_init` has a real void interface and uses the same export pipeline.

`int` and `int32_t` must denote the same C type for this example, which the
caller checks at compile time. Other C APIs should use the matching explicit
PPy element type. Memory must stay alive and aligned for the call; PPy neither
copies nor frees it and does not track its lifetime at runtime.

## Run it

From this directory, with PPy and a C11 compiler available:

```sh
ppy emit c array_api.ppy -o array_api.c
ppy emit header array_api.ppy -o array_api.h
cc -std=c11 -Wall -Werror array_api.c caller.c -lm -o caller
./caller
```

For a sanitizer check, add `-fsanitize=address,undefined -fno-omit-frame-pointer`
to the C command. `ppy emit cpp` can also generate the implementation; compile
it with a C++17 compiler and link it to the C caller compiled as C11.

## What it prints

```text
original C array updated: ice[7][7] = 63
```
