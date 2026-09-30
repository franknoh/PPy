# Effects in native code

Under `ppy run`, a function that prints, reads input, reads or writes a
text file, or calls a Python function can still compile. This page says
how that stays correct, what goes native, and what stays in Python and why.

## The problem

A native function checks what it cannot handle: an integer past 64 bits, a
list changed under it, a value of the wrong type. When a check fails, the
call falls back: Python runs the same call again from the start.

That is only safe while nothing the native call did can be seen. If it had
already printed a line, the second run would print it again.

PPy solves this in two ways, depending on whether the effect can be taken
back.

## Output is held

`print` writes into a buffer the thread keeps, not to the terminal. When
the call returns, the boundary writes the buffer through `sys.stdout` (or
`sys.stderr`) in one piece. When the call falls back, the buffer is dropped,
and Python prints the same lines as it runs the call again.

```python
def grow(n: int) -> int:
    x = 1
    for i in range(n):
        print("i", i)
        x = x * 1000
    return x


print(grow(3))
print(grow(10))
```

`grow(3)` runs natively and prints `i 0` to `i 2`, then `1000000000`. In
`grow(10)`, `x` passes 64 bits at `i = 6`: the call falls back, the lines it
held are dropped, and Python prints `i 0` to `i 9` once.

So a function that only prints keeps every check it had. The output goes
through Python's own `sys.stdout`, so it interleaves with what Python
prints, and redirection, `contextlib.redirect_stdout`, and pytest's capture
all see it.

What `print` takes natively:

- positional arguments of any type an f-string field takes: numbers, bools,
  strings, lists, dicts, sets, tuples, `None`, and project classes (their
  `__str__` or `__repr__`);
- `sep=` and `end=` as string literals or `None`;
- `file=sys.stdout` or `file=sys.stderr`;
- `flush=True` or `flush=False`.

Python evaluates every argument before it writes any. PPy writes each
argument as it evaluates it, which gives the same output as long as the
arguments after the first have no effect of their own. A later argument
that calls something that could print or change what an earlier one shows
keeps the function in Python.

The output of a native call appears when the call returns, or at the next
barrier (below). A long native loop that prints progress shows it at the
end, where CPython would show it as it goes. Use `flush=True` where the
timing matters.

## Barriers

Some effects cannot be taken back:

- `input()`, which consumes a line of stdin;
- `print(..., flush=True)`;
- a call into Python;
- opening, reading, or writing a file.

Each of these is a barrier. It first writes out what the call has held, so
output keeps its order, and then does its work with the GIL held. From the
first barrier on, the call must never fall back. The compiler proves that
over the whole function before it lets the function go native:

- a check CPython itself raises for (an index out of range, a missing key, a
  division by zero) raises the same exception natively, with CPython's text.
  A module with a barrier is compiled in exception mode for this. A check
  whose text native code only approximates (`int()` of a bad string, which
  CPython follows with the string) cannot follow a barrier;
- a check that stands for a limit of native code (an integer past 64 bits)
  must be proven away by the prover, or the function stays in Python;
- a call to a function that may fall back, or may raise an exception PPy
  cannot give CPython's exact text for, keeps the caller in Python when it
  can follow a barrier;
- a loop counts: a check in a loop body after a barrier, or before it, in
  the next iteration, follows it;
- a function called from Python that takes a list, a dict, a set, or an
  object by copy stays in Python if it has a barrier. Python code that runs
  at the barrier would read or change the caller's object, not the copy.

The same holds across calls. A caller that calls a function with a barrier
has crossed one when the call returns, and the rule applies to what follows
the call.

```python
def ask(n: int) -> str:
    print("asking", n)
    name = input("name? ")
    print("hello", name)
    return name


def after(n: int) -> int:
    s = input()
    return n * n + len(s)
```

`ask` is native: "asking 1" is written before the prompt. In `after`, `n * n` may pass 64 bits after the barrier, so
`after` stays in Python.

A native exception after a barrier is raised by the boundary as it is,
rather than by running the call again: a builtin exception is rebuilt from
its class and its text, and an exception Python raised inside the call (an
`EOFError` from `input()`, a `FileNotFoundError` from `open`) is raised as
that very object, with its traceback. Native code can catch either with
`try`/`except`.

## Calls into Python

A native function can call a function that stays in Python: another
function of the module that did not compile, a function of an imported
module, or a builtin. The call goes through Python with the GIL held:

- arguments are numbers, bools, strings, and `None`, boxed;
- the result is unboxed by the type the checker gave the call: a `float`, a
  `bool`, a `str`, or nothing. A value of another type raises `TypeError`;
- what the callee raises is raised natively.

An `int` result is not taken back: a Python function may return an integer
no word holds, and after the call there is no falling back. A call whose
result is an `int` keeps its caller in Python.

A function that calls into Python holds the GIL where it does so. It is
native, but not free-threaded.

## Text files

`with open(path, ...) as f:` compiles when the mode is a text mode:

```python
def count(path: str) -> list[str]:
    found: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("line"):
                found.append(line.strip())
    return found
```

The file is Python's own: `open`, `read`, `readline`, `readlines`, `write`,
iteration, and `close` go through `io`, so encodings, newline translation,
and errors are CPython's exactly. The file is closed on every way out of
the block, as its `__exit__` does.

## What `ppy explain` says

`ppy explain` names the rule each native function with effects runs under:

```text
llvm backend: native; effects: output is held until the call returns, and dropped if it falls back
llvm backend: native; effects: output is written at each barrier and when the call returns; nothing falls back after the first barrier (`input()`)
llvm backend: boxed: integer arithmetic that may not fit 64 bits can follow `input()`
```

`ppy explain --summary` counts functions kept in Python this way under
"something that may fall back follows an effect native code cannot take
back".

## Standalone binaries

A standalone binary has no Python to fall back to, so none of this applies
there: `print` and `input` go to native shims, and a failed check prints
what CPython would and exits.

See also: [Effects and the three paths](effects.md),
[Native lowering](native-lowering.md).
