"""The `re` program as Codon compiles it: the same source, Codon's own `re`."""

import re
import time

WORD = re.compile(rb"[A-Za-z]+")
PAIR = re.compile(rb"(?P<key>\w+)\s*=\s*(\d+)")
HEX = re.compile(rb"0x[0-9a-f]+", re.IGNORECASE)


def count_words(text: bytes) -> int:
    return sum(1 for _ in WORD.finditer(text))


def longest_word(text: bytes) -> int:
    return max((m.end() - m.start() for m in WORD.finditer(text)), default=0)


def sum_values(text: bytes) -> int:
    return sum(int(m.group(2)) for m in PAIR.finditer(text))


def count_hex(text: bytes) -> int:
    return sum(1 for _ in HEX.finditer(text))


def make_text(lines: int) -> bytes:
    words = [b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta", b"eta", b"theta"]
    parts = []
    state = 12345
    for i in range(lines):
        state = (state * 1103515245 + 12345) % (1 << 31)
        word = words[state % len(words)]
        parts.append(word + b" = " + str(state % 1000).encode() + b"  " + word.upper())
        if i % 7 == 0:
            parts.append(b"0x" + format(state, "x").encode())
        parts.append(b"\n")
    return b"".join(parts)


def timed(label, run, text):
    best = 1e9
    answer = 0
    for _ in range(5):
        started = time.perf_counter()
        answer = run(text)
        best = min(best, time.perf_counter() - started)
    print(f"# {label}: {best * 1000:.2f} ms")
    return answer


def main():
    text = make_text(400_000)
    print(len(text))
    print(timed("count_words", count_words, text))
    print(timed("longest_word", longest_word, text))
    print(timed("sum_values", sum_values, text))
    print(timed("count_hex", count_hex, text))


main()
