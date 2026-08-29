"""Deterministic small-program generator for differential testing.

Programs are built from the subset where the three paths make hard promises:
integer and float arithmetic, branches and bounded loops, lists and tuples,
calls, aliases, in-place mutation, and exceptions. Every program is a pure
function of its seed, so a failing seed is a permanent regression test, and
every generated loop is bounded, so every program terminates.
"""

from __future__ import annotations

import random

__all__ = ["generate_program"]

_INT_OPS = ("+", "-", "*", "//", "%")
_FLOAT_OPS = ("+", "-", "*")
_COMPARES = ("<", "<=", ">", ">=", "==", "!=")


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 0

    def put(self, text: str) -> None:
        self.lines.append("    " * self.depth + text)


class _Generator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.fresh = 0

    def name(self, prefix: str) -> str:
        self.fresh += 1
        return f"{prefix}{self.fresh}"

    # -- expressions ---------------------------------------------------------

    def int_expr(self, names: list[str], depth: int = 0) -> str:
        if depth > 2 or not names or self.rng.random() < 0.35:
            if names and self.rng.random() < 0.6:
                return self.rng.choice(names)
            return str(self.rng.randint(-9, 9))
        op = self.rng.choice(_INT_OPS)
        left = self.int_expr(names, depth + 1)
        right = self.int_expr(names, depth + 1)
        if op in {"//", "%"}:
            # Divide by something provably nonzero; ZeroDivisionError has its
            # own dedicated statement form below.
            right = f"({right} * {right} + 1)"
        return f"({left} {op} {right})"

    def float_expr(self, names: list[str], depth: int = 0) -> str:
        if depth > 2 or not names or self.rng.random() < 0.35:
            if names and self.rng.random() < 0.6:
                return self.rng.choice(names)
            return f"{self.rng.randint(-40, 40)}.{self.rng.randint(0, 9)}"
        op = self.rng.choice(_FLOAT_OPS)
        return f"({self.float_expr(names, depth + 1)} {op} {self.float_expr(names, depth + 1)})"

    def condition(self, names: list[str]) -> str:
        return f"{self.int_expr(names, 2)} {self.rng.choice(_COMPARES)} {self.int_expr(names, 2)}"

    # -- statements ----------------------------------------------------------

    def body(self, writer: _Writer, ints: list[str], lists: list[str], budget: int) -> None:
        # A block must hold at least one statement, whatever the budget says.
        budget = max(1, budget)
        while budget > 0:
            budget -= 1
            choice = self.rng.random()
            if choice < 0.22:
                target = self.name("n")
                writer.put(f"{target} = {self.int_expr(ints)}")
                ints.append(target)
            elif choice < 0.34 and ints:
                writer.put(f"{self.rng.choice(ints)} += {self.int_expr(ints, 2)}")
            elif choice < 0.46:
                writer.put(f"if {self.condition(ints)}:")
                writer.depth += 1
                # Copies on both sides: a name bound on one branch only
                # must not look bound to the code after the join.
                self.body(writer, list(ints), list(lists), min(budget, 2))
                writer.depth -= 1
                writer.put("else:")
                writer.depth += 1
                self.body(writer, list(ints), list(lists), 1)
                writer.depth -= 1
            elif choice < 0.58:
                index = self.name("i")
                writer.put(f"for {index} in range({self.rng.randint(1, 6)}):")
                writer.depth += 1
                # The loop may run zero times, so its bindings stay inside.
                self.body(writer, [*ints, index], list(lists), min(budget, 2))
                writer.depth -= 1
            elif choice < 0.68:
                target = self.name("xs")
                items = ", ".join(self.int_expr(ints, 2) for _ in range(self.rng.randint(1, 4)))
                writer.put(f"{target} = [{items}]")
                lists.append(target)
            elif choice < 0.78 and lists:
                source = self.rng.choice(lists)
                if self.rng.random() < 0.5:
                    writer.put(f"{source}.append({self.int_expr(ints, 2)})")
                else:
                    alias = self.name("ys")
                    writer.put(f"{alias} = {source}")
                    writer.put(f"{alias}.append({self.int_expr(ints, 2)})")
                    lists.append(alias)
            elif choice < 0.88 and lists:
                source = self.rng.choice(lists)
                total = self.name("t")
                writer.put(f"{total} = 0")
                item = self.name("v")
                writer.put(f"for {item} in {source}:")
                writer.depth += 1
                writer.put(f"{total} += {item}")
                writer.depth -= 1
                ints.append(total)
            else:
                target = self.name("r")
                source = self.rng.choice(lists) if lists else None
                writer.put("try:")
                writer.depth += 1
                if source is not None and self.rng.random() < 0.5:
                    writer.put(f"{target} = {source}[{self.int_expr(ints, 2)}]")
                    failure = "IndexError"
                else:
                    writer.put(f"{target} = {self.int_expr(ints, 2)} // {self.int_expr(ints, 2)}")
                    failure = "ZeroDivisionError"
                writer.depth -= 1
                writer.put(f"except {failure}:")
                writer.depth += 1
                writer.put(f"{target} = {self.rng.randint(0, 5)}")
                writer.depth -= 1
                ints.append(target)

    def function(self, writer: _Writer, name: str) -> int:
        arity = self.rng.randint(1, 3)
        params = [self.name("a") for _ in range(arity)]
        writer.put(f"def {name}({', '.join(f'{p}: int' for p in params)}) -> int:")
        writer.depth += 1
        ints = list(params)
        lists: list[str] = []
        self.body(writer, ints, lists, self.rng.randint(3, 6))
        pieces = " + ".join([*ints[-3:], *[f"len({xs})" for xs in lists[-2:]]])
        writer.put(f"return {pieces}")
        writer.depth -= 1
        writer.put("")
        writer.put("")
        return arity

    def program(self) -> str:
        writer = _Writer()
        functions: list[tuple[str, int]] = []
        for _ in range(self.rng.randint(1, 3)):
            name = self.name("f")
            functions.append((name, self.function(writer, name)))
        for name, arity in functions:
            for _ in range(self.rng.randint(1, 3)):
                arguments = ", ".join(str(self.rng.randint(-9, 9)) for _ in range(arity))
                writer.put(f"print({name}({arguments}))")
        # A tuple round-trip and a float reduction, outside any function.
        writer.put("pair = (3, 4.5)")
        writer.put("left, right = pair")
        writer.put("acc = 0.0")
        writer.put("for scale in [1.5, 2.5, 3.5]:")
        writer.put("    acc += right * scale + left")
        writer.put("print(left, right, acc)")
        return "\n".join(writer.lines) + "\n"


def generate_program(seed: int) -> str:
    """The program for this seed; identical on every machine and run."""
    return _Generator(seed).program()
