"""`random`, `heapq`, `bisect`, and `math` in native code, CPython's answers kept.

`random`'s generator is CPython's Mersenne Twister and `random.py`'s
algorithms on top of it; under `ppy run` native code draws from the state
inside `random._inst`, so native and Python draws interleave as CPython's
would, and a native call that falls back after drawing hands Python the
state it started from. `heapq` and `bisect` make CPython's comparisons in
CPython's order; `math` gives CPython's results and raises where it does.
Each program is held to CPython under `ppy`, `ppy run`, a standalone binary,
and emitted C and C++ under AddressSanitizer with leak detection, with
`ppy explain` confirming the listed functions went native.
"""

from __future__ import annotations

import ctypes
import os
import random
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

RANDOM = """
import random


def draws(n: int) -> str:
    random.seed(12345)
    xs = [random.randint(-50, 50) for _ in range(n)]
    random.shuffle(xs)
    a = random.choice(xs)
    b = random.randrange(7)
    c = random.randrange(3, 90)
    d = random.randrange(100, 3, -7)
    e = random.getrandbits(40)
    f = random.getrandbits(5)
    ys = random.sample(xs, 5)
    zs = random.sample(range(1000), 12)
    big = random.sample(range(10, 100000, 3), 40)
    ws = random.choices(xs, k=4)
    vs = random.choices(["a", "b", "c"], [1, 5, 2], k=6)
    us = random.choices([1.5, 2.5], cum_weights=[0.25, 1.0], k=3)
    g = random.random() + random.uniform(-2.0, 3.5) + random.uniform(1, 10)
    h = random.triangular(0.0, 10.0, 7.0) + random.triangular()
    i = random.normalvariate(5.0, 2.0) + random.lognormvariate(0.0, 0.5) + random.expovariate(3.0)
    j = random.paretovariate(2.5) + random.weibullvariate(1.0, 1.5)
    return f"{xs} {a} {b} {c} {d} {e} {f} {ys} {zs} {sum(big)} {ws} {vs} {us} {g!r} {h!r} {i!r} {j!r}"


def main() -> None:
    print(draws(30))


main()
"""

HEAPS = """
import bisect
import heapq
from heapq import heappop, heappush


def heaps(n: int) -> str:
    h: list[int] = []
    for i in range(n):
        heappush(h, (i * 37) % 101)
    first = [heappop(h) for _ in range(5)]
    top = heapq.heappushpop(h, 1)
    top2 = heapq.heappushpop(h, 1000)
    rep = heapq.heapreplace(h, 3)
    xs = [(i * 53) % 29 for i in range(40)]
    heapq.heapify(xs)
    fs = [2.5, -1.0, 7.25, 0.0, -0.0, 3.5]
    heapq.heapify(fs)
    small = heapq.nsmallest(4, xs)
    large = heapq.nlargest(3, [5, 1, 9, 9, 2])
    return f"{first} {top} {top2} {rep} {h} {xs} {fs} {small} {large}"


def dijkstra(n: int) -> list[int]:
    dist = [1000000000] * n
    dist[0] = 0
    queue: list[tuple[int, int]] = [(0, 0)]
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue
        for step in [1, 3, 7]:
            v = (u * step + 1) % n
            w = (u + v) % 11 + 1
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(queue, (d + w, v))
    return dist


def searches(n: int) -> str:
    xs = sorted([(i * 31) % 50 for i in range(n)])
    a = bisect.bisect_left(xs, 10)
    b = bisect.bisect_right(xs, 10)
    c = bisect.bisect(xs, 10, 3, 20)
    for v in [5, 5, 49, -1, 25]:
        bisect.insort(xs, v)
    bisect.insort_left(xs, 7)
    words = ["apple", "fig", "kiwi"]
    bisect.insort(words, "grape")
    fl = [0.5, 1.5, 2.5]
    d = bisect.bisect_left(fl, 1)
    return f"{a} {b} {c} {xs} {words} {d}"


def main() -> None:
    print(heaps(30))
    print(dijkstra(20))
    print(searches(25))


main()
"""

ERRORS = """
import heapq
import random


def errs(n: int) -> list[str]:
    out: list[str] = []
    empty: list[int] = []
    pair = [1, 2]
    one = [1]
    zeros = [0, 0]
    try:
        random.randrange(n - n)
    except ValueError as e:
        out.append(str(e))
    try:
        random.randrange(n, 3)
    except ValueError as e:
        out.append(str(e))
    try:
        random.randrange(n, 3, 2)
    except ValueError as e:
        out.append(str(e))
    try:
        random.randrange(1, n, 0)
    except ValueError as e:
        out.append(str(e))
    try:
        random.randint(n, 2)
    except ValueError as e:
        out.append(str(e))
    try:
        random.choice(empty)
    except IndexError as e:
        out.append(str(e))
    try:
        random.sample(empty, 1)
    except ValueError as e:
        out.append(str(e))
    try:
        random.choices(pair, one, k=2)
    except ValueError as e:
        out.append(str(e))
    try:
        random.choices(pair, zeros, k=2)
    except ValueError as e:
        out.append(str(e))
    try:
        heapq.heappop(empty)
    except IndexError as e:
        out.append(str(e))
    try:
        random.getrandbits(-1)
    except ValueError as e:
        out.append(str(e))
    try:
        random.expovariate(0.0)
    except ZeroDivisionError as e:
        out.append(str(e))
    return out


def main() -> None:
    random.seed(1)
    for line in errs(10):
        print(line)
    print(random.random())


main()
"""

GAUSS = """
import random


def f(n: int) -> str:
    random.seed(n)
    a = random.gauss(0.0, 1.0)
    b = random.gauss(2.0, 0.5)
    c = random.gauss()
    random.seed(5)
    d = random.gauss(0.0, 1.0)
    e = random.gammavariate(0.5, 2.0) + random.gammavariate(1, 3.0) + random.gammavariate(4.5, 1.0)
    g = random.betavariate(2.0, 5.0)
    return f"{a!r} {b!r} {c!r} {d!r} {e!r} {g!r}"


def main() -> None:
    print(f(30))


main()
"""

MATH = """
import math
from math import inf, pi


def integers(n: int) -> list[str]:
    out: list[str] = []
    for i in range(1, n):
        out.append(f"{math.gcd(i, 12, 18)} {math.lcm(i, 6)} {math.isqrt(i * 1000)}")
        out.append(f"{math.comb(n, i)} {math.perm(n, i % 5)} {math.factorial(i % 20)}")
    return out


def floats(n: int, x: float) -> list[str]:
    out: list[str] = []
    for i in range(n):
        y = x + i
        out.append(f"{math.copysign(y, -1.0)!r} {math.atan2(y, i)!r} {math.hypot(y, i)!r}")
        out.append(f"{math.hypot(y, i, 3.0)!r} {math.dist((y, 1.0), (i, 2.0))!r}")
        out.append(f"{math.isclose(y, i / 12)} {math.isclose(y, y + 1e-12, rel_tol=1e-9)}")
        out.append(f"{math.log(y + 1, 2)!r} {math.degrees(y)!r} {math.radians(i)!r}")
        out.append(f"{math.asinh(y * i)!r} {math.erf(y)!r} {math.expm1(y / 10)!r}")
        out.append(f"{math.fmod(y, 1.5)!r} {pi * y + math.tau - math.e!r} {inf > y}")
    return out


def sums(n: int) -> str:
    xs = [0.1 * i for i in range(n)]
    xs.append(1e16)
    xs.append(-1e16)
    ys = [i + 1 for i in range(n % 15)]
    return f"{math.fsum(xs)!r} {sum(xs)!r} {math.prod(ys)} {math.dist(xs[:3], xs[3:6])!r}"


def main() -> None:
    for line in integers(25):
        print(line)
    for line in floats(12, 2.5):
        print(line)
    print(sums(40))


main()
"""

ITERTOOLS = """
import itertools
import string
from itertools import combinations, pairwise, permutations
from string import digits


def walks(n: int) -> str:
    t = 0
    out: list[str] = []
    for a, b in itertools.product(range(n), range(3)):
        t += a * b
    for p in permutations([1, 2, 3], 2):
        out.append(f"{p[0]}{p[1]}")
    for c in combinations(range(5), 3):
        t += c[0] * 100 + c[1] * 10 + c[2]
    for x, y in pairwise([4, 9, 1, 7]):
        t += x * y
    for w in itertools.combinations_with_replacement([1.5, 2.0, 3.0], 2):
        out.append(str(w[0] + w[1]))
    for z in itertools.chain([1, 2], [3], range(4, 6)):
        t += z
    for s in itertools.islice([10, 20, 30, 40, 50], 1, 4, 2):
        t += s
    acc = list(itertools.accumulate([1, 5, 2, 8]))
    fac = list(itertools.accumulate([0.5, 1.5], initial=2.0))
    rep = list(itertools.repeat("ab", 3))
    pairs = len(list(combinations(range(n), 2)))
    return f"{t} {out} {acc} {fac} {rep} {pairs}"


def letters(n: int) -> str:
    s = ""
    for i in range(n):
        s += string.ascii_lowercase[i % 26] + digits[i % 10]
    return s + str(len(string.punctuation)) + str("q" in string.ascii_letters)


def main() -> None:
    print(walks(20))
    print(letters(30))


main()
"""

DEQUE = """
from collections import deque


def queues(n: int) -> str:
    q: deque[int] = deque()
    for i in range(n):
        q.append(i)
        if i % 3 == 0:
            q.appendleft(q.pop())
    r: deque[int] = deque([1, 2, 3])
    r.rotate(1)
    r.extend([7, 8])
    s = 0
    while r:
        s = s * 10 + r.popleft()
    t = 0
    for x in q:
        t = t * 7 % 1000003 + x
    return f"{s} {len(q)} {t} {q[0]} {q[-1]} {q[-20]}"


def bfs(n: int) -> int:
    seen = [False] * n
    seen[0] = True
    frontier: deque[int] = deque([0])
    steps = 0
    while frontier:
        u = frontier.popleft()
        steps += u
        for v in [(u * 2 + 1) % n, (u * 3 + 2) % n]:
            if not seen[v]:
                seen[v] = True
                frontier.append(v)
    return steps


def empty(n: int) -> str:
    q: deque[int] = deque()
    try:
        q.popleft()
    except IndexError as e:
        return str(e) + str(n)
    return ""


def main() -> None:
    print(queues(20))
    print(bfs(50))
    print(empty(3))


main()
"""


MAPPINGS = """
from collections import Counter, OrderedDict, defaultdict, deque


def defaults(n: int) -> str:
    d: defaultdict[int, int] = defaultdict(int)
    for i in range(n):
        d[i % 4] += i
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for i in range(n):
        groups["odd" if i % 2 else "even"].append(i)
    seen = defaultdict(lambda: -1)
    for i in range(5):
        if seen[i % 3] < 0:
            seen[i % 3] = i * 10
    f: defaultdict[int, float] = defaultdict(float)
    f[3] += 1.5
    s: defaultdict[int, str] = defaultdict(str)
    s[1] += "ab"
    s[2] += s[1] + "c"
    sets: defaultdict[int, set[int]] = defaultdict(set)
    sets[1].add(5)
    sets[1].add(3)
    total = 0
    for k, v in d.items():
        total += k * v
    return f"{d} {groups} {dict(seen)} {f} {s} {sets} {total} {len(d)} {3 in d} {9 in d} {d.get(7, -5)}"


def counts(text: str) -> str:
    c = Counter(text)
    words = Counter(["a", "b", "a", "c", "b", "a"])
    nums = Counter([3, 1, 3, 2, 3, 1])
    top = nums.most_common(2)
    out: list[str] = []
    for w, k in words.most_common():
        out.append(f"{w}{k}")
    c.update("zzz")
    c.subtract("ll")
    words.update(["d"])
    both = nums + Counter([1, 1, 4])
    less = nums - Counter([3, 3, 3, 3, 1])
    least = nums & Counter([3, 2, 2])
    most = nums | Counter([2, 2, 5])
    return (
        f"{c} {c['q']} {c['z']} {words} {top} {out} {nums.total()} {sorted(nums.elements())} "
        f"{both} {less} {least} {most} {c.most_common(3)} {len(c)} {list(words)}"
    )


def ordered(n: int) -> str:
    od: OrderedDict[str, int] = OrderedDict()
    for i in range(n):
        od[str(i)] = i * i
    od.move_to_end("1")
    od.move_to_end("3", last=False)
    k, v = 0, 0
    nums: OrderedDict[int, int] = OrderedDict()
    for i in range(6):
        nums[i] = i + 10
    k, v = nums.popitem()
    a, b = nums.popitem(last=False)
    return f"{od} {list(od)} {k} {v} {a} {b} {nums} {OrderedDict()}"


def queue(n: int) -> str:
    q = deque([1, 2, 3])
    q.extend(range(n))
    q[1] = 50
    q.rotate(2)
    return f"{q} {list(q)} {deque()}"


def errors(n: int) -> list[str]:
    out: list[str] = []
    od: OrderedDict[int, int] = OrderedDict()
    try:
        od.popitem()
    except KeyError as e:
        out.append(str(e))
    try:
        od.move_to_end(n)
    except KeyError as e:
        out.append(str(e))
    return out


def main() -> None:
    print(defaults(10))
    print(counts("hello world"))
    print(ordered(5))
    print(queue(4))
    print(errors(7))


main()
"""

MEMO = """
import functools
from functools import cache, lru_cache


@cache
def paths(r: int, c: int) -> int:
    if r == 0 or c == 0:
        return 1
    return (paths(r - 1, c) + paths(r, c - 1)) % 1000000007


@lru_cache(maxsize=None)
def breaks(word: str, start: int) -> bool:
    if start == len(word):
        return True
    for end in range(start + 1, len(word) + 1):
        piece = word[start:end]
        if (piece == "ab" or piece == "abc" or piece == "c" or piece == "d") and breaks(word, end):
            return True
    return False


@functools.lru_cache(maxsize=3)
def noisy(n: int) -> int:
    print("computing", n)
    return n * n


@lru_cache
def label(n: int, x: float) -> str:
    print("label", n, x)
    return f"{n}:{x}"


@lru_cache(maxsize=0)
def never(n: int) -> int:
    print("never", n)
    return n + 1


@cache
def flag(n: int) -> bool:
    print("flag", n)
    return n % 3 == 0


def drive(n: int) -> int:
    total = 0
    for i in [1, 2, 3, 1, 4, 1, 2, 5, 3]:
        total += noisy(i)
    for i in range(3):
        total += never(i) + never(i)
    print(label(1, 0.5), label(1, 0.5), label(2, -0.0), label(2, 0.0))
    print(flag(3), flag(3), flag(4))
    return total + paths(n, n)


def main() -> None:
    print(paths(40, 40))
    print(breaks("abcdababcd", 0), breaks("abx", 0))
    print(drive(25))
    print(noisy(1), noisy(9))


main()
"""

FUNCTIONAL = """
import functools
import operator
from functools import reduce
from operator import attrgetter, itemgetter


class Job:
    def __init__(self, name: str, weight: int) -> None:
        self.name = name
        self.weight = weight


def combine(a: int, b: int) -> int:
    return a * 31 + b


def ops(xs: list[int], pairs: list[tuple[int, int]]) -> str:
    total = functools.reduce(operator.add, xs, 0)
    prod = reduce(lambda a, b: a * b, xs)
    folded = reduce(combine, xs, 7)
    biggest = reduce(max, xs)
    words = reduce(operator.add, ["ab", "c", "de"], "")
    joined = reduce(lambda s, w: s + "-" + w, ["x", "y", "z"])
    ordered = sorted(pairs, key=itemgetter(1))
    best = max(pairs, key=operator.itemgetter(1))
    least = min(pairs, key=itemgetter(0, 1))
    m = list(map(operator.neg, xs))
    f = list(filter(operator.truth, [0, 1, 2, 0, 3]))
    xs.sort(key=operator.neg)
    floats = reduce(operator.truediv, [100.0, 4.0, 5.0])
    return f"{total} {prod} {folded} {biggest} {words} {joined} {ordered} {best[0]} {best[1]} {least[0]} {m} {f} {xs} {floats}"


def jobs(n: int) -> str:
    js = [Job("a", 3), Job("b", 1), Job("c", 2)]
    js.sort(key=attrgetter("weight"))
    out = ""
    for j in js:
        out += j.name
    return out + str(n)


def errs(n: int) -> list[str]:
    out: list[str] = []
    empty: list[int] = []
    try:
        reduce(operator.add, empty)
    except TypeError as e:
        out.append(str(e))
    try:
        reduce(operator.floordiv, [n, 0])
    except ZeroDivisionError as e:
        out.append(str(e))
    return out


def main() -> None:
    print(ops([3, 1, 2], [(5, 2), (6, 1), (7, 3)]))
    print(jobs(4))
    print(errs(5))


main()
"""

COMPARED = """
from functools import cmp_to_key, partial, reduce


def scale(factor: int, offset: int, x: int) -> int:
    return factor * x + offset


def by_last_digit(a: int, b: int) -> int:
    if a % 10 != b % 10:
        return a % 10 - b % 10
    return b - a


def parts(xs: list[int], k: int) -> str:
    doubled = list(map(partial(scale, 2, k), xs))
    total = reduce(partial(scale, 3), xs, 1)
    picked = sorted(xs, key=partial(scale, -1, 0))
    ordered = sorted(xs, key=cmp_to_key(by_last_digit))
    xs.sort(key=cmp_to_key(lambda a, b: a - b), reverse=True)
    words = sorted(["bb", "a", "ccc", "dd"], key=cmp_to_key(lambda s, t: len(s) - len(t)))
    return f"{doubled} {total} {picked} {ordered} {xs} {words}"


def main() -> None:
    print(parts([13, 21, 3, 42, 11, 33], 5))


main()
"""

GENERATORS = """
import random


def roll(rng: random.Random, n: int) -> int:
    total = 0
    for _ in range(n):
        total = total * 7 + rng.randint(1, 6)
    return total


def draws(seed: int) -> str:
    r = random.Random(seed)
    s = random.Random(seed + 1)
    xs = [r.randint(-50, 50) for _ in range(20)]
    r.shuffle(xs)
    a = r.choice(xs)
    b = r.randrange(7) + s.randrange(3, 90) + r.randrange(100, 3, -7)
    c = r.getrandbits(40)
    ys = r.sample(xs, 5)
    zs = s.sample(range(1000), 6)
    ws = r.choices(xs, k=4)
    vs = s.choices(["a", "b", "c"], [1, 5, 2], k=6)
    g = r.random() + r.uniform(-2.0, 3.5) + s.gauss(0.0, 1.0) + s.gauss(1.0, 2.0) + s.gauss()
    h = r.triangular(0.0, 10.0, 7.0) + r.normalvariate(5.0, 2.0) + r.expovariate(3.0)
    j = r.gammavariate(2.5, 1.0) + r.betavariate(2.0, 3.0) + r.paretovariate(2.5)
    t = roll(r, 5)
    r.seed(99)
    k = r.random()
    random.seed(5)
    m = random.random()
    return f"{xs} {a} {b} {c} {ys} {zs} {ws} {vs} {g!r} {h!r} {j!r} {t} {k!r} {m!r}"


def main() -> None:
    print(draws(12345))
    print(draws(-7))


main()
"""

NESTED_MEMO = """
import functools


def min_distance_up_bottom(word1: str, word2: str) -> int:
    len_word1 = len(word1)
    len_word2 = len(word2)

    @functools.cache
    def min_distance(index1: int, index2: int) -> int:
        if index1 >= len_word1:
            return len_word2 - index2
        if index2 >= len_word2:
            return len_word1 - index1
        diff = int(word1[index1] != word2[index2])
        return min(
            1 + min_distance(index1 + 1, index2),
            1 + min_distance(index1, index2 + 1),
            diff + min_distance(index1 + 1, index2 + 1),
        )

    return min_distance(0, 0)


def fib_terms(n: int) -> list[int]:
    @functools.lru_cache(maxsize=None)
    def term(i: int) -> int:
        if i < 0:
            raise ValueError("n is negative")
        if i < 2:
            return i
        return term(i - 1) + term(i - 2)

    if n < 0:
        raise ValueError("n is negative")
    return [term(i) for i in range(n + 1)]


def counted(n: int) -> str:
    calls: list[int] = []

    @functools.lru_cache(maxsize=2)
    def square(x: int) -> int:
        calls.append(x)
        return x * x

    total = 0
    for x in [1, 2, 1, 3, 1, 2, 2]:
        total += square(x)
    return f"{total} {calls} {n}"


def main() -> None:
    print(min_distance_up_bottom("intention", "execution"))
    print(min_distance_up_bottom("zooicoarchaeologist", "zoologist" * 3))
    print(fib_terms(90)[-3:])
    print(counted(1))
    try:
        fib_terms(-1)
    except ValueError as e:
        print(e)


main()
"""

COUNTED = """
from collections import Counter, defaultdict


def majority_vote(votes: list[int], votes_needed_to_win: int) -> list[int]:
    majority_candidate_counter: Counter[int] = Counter()
    for vote in votes:
        majority_candidate_counter[vote] += 1
        if len(majority_candidate_counter) == votes_needed_to_win:
            majority_candidate_counter -= Counter(set(majority_candidate_counter))
    majority_candidate_counter = Counter(
        vote for vote in votes if vote in majority_candidate_counter
    )
    return [
        vote
        for vote in majority_candidate_counter
        if majority_candidate_counter[vote] > len(votes) / votes_needed_to_win
    ]


def inplace(words: list[str]) -> str:
    a = Counter(words)
    b = Counter(["x", "x", "y", "zz"])
    a += b
    a -= Counter(["x", "q"])
    c = Counter(words)
    c |= b
    d = Counter(words)
    d &= b
    return f"{a} {c} {d}"


def solution(limit: int) -> int:
    frequencies: defaultdict = defaultdict(int)
    for perimeter in range(1, limit):
        frequencies[perimeter % 37] += 1
    return sum(1 for frequency in frequencies.values() if frequency == 3)


def main() -> None:
    print(majority_vote([1, 2, 2, 3, 1, 3, 2], 3))
    print(majority_vote([1, 2, 2, 3, 1, 3, 2], 2))
    print(majority_vote([1, 2, 2, 3, 1, 3, 2], 4))
    print(inplace(["x", "y", "x", "w", "y", "y"]))
    print(solution(100))


main()
"""

PASSED = """
import random
from collections import Counter, defaultdict, deque


def fill(d: defaultdict[str, list[int]], n: int) -> None:
    for i in range(n):
        d[str(i % 3)].append(i)


def tally(words: list[str]) -> Counter[str]:
    return Counter(words)


def drain(q: deque[int]) -> int:
    total = 0
    while q:
        total = total * 3 + q.popleft()
    return total


def pick(r: random.Random, n: int) -> list[int]:
    return [r.randrange(100) for _ in range(n)]


def run(n: int) -> str:
    d: defaultdict[str, list[int]] = defaultdict(list)
    fill(d, n)
    c = tally(["b", "a", "b", "c", "a", "b"])
    c.update(tally(["c", "c"]))
    q = deque(range(n))
    r = random.Random(n)
    return f"{d} {c} {drain(q)} {q} {pick(r, 4)} {pick(r, 2)}"


def main() -> None:
    print(run(7))


main()
"""

VALUES = """
import operator
from collections.abc import Callable
from functools import partial
from operator import itemgetter


def fold(f: Callable[[int, int], int], xs: list[int], start: int) -> int:
    acc = start
    for x in xs:
        acc = f(acc, x)
    return acc


def scale(k: int, x: int) -> int:
    return k * x


def apply_all(n: int) -> str:
    a = fold(operator.add, [1, 2, 3], n)
    b = fold(operator.mul, [1, 2, 3], n)
    c = fold(max, [4, 9, 2], n)
    g: Callable[[int], int] = partial(scale, 3)
    pairs = [[3, 1], [1, 2], [2, 0]]
    first: Callable[[list[int]], int] = itemgetter(0)
    keyed = first(pairs[1]) + first(pairs[2])
    return f"{a} {b} {c} {g(5)} {keyed}"


def main() -> None:
    print(apply_all(10))


main()
"""

#: name -> (source, the functions `ppy explain` must call native).
PROGRAMS = {
    "random": (RANDOM, ("draws",)),
    "heaps": (HEAPS, ("heaps", "dijkstra", "searches")),
    "errors": (ERRORS, ("errs",)),
    # `gauss` holds a value back in Python's generator: native in a
    # standalone binary, Python's under `ppy run`.
    "gauss": (GAUSS, ()),
    "math": (MATH, ("integers", "floats", "sums")),
    "itertools": (ITERTOOLS, ("walks", "letters")),
    "deque": (DEQUE, ("queues", "bfs", "empty")),
    "mappings": (MAPPINGS, ("defaults", "counts", "ordered", "queue", "errors")),
    # Each cached function's entry looks its arguments up before the body runs.
    "memo": (MEMO, ("paths", "breaks", "noisy", "label", "never", "flag")),
    "functional": (FUNCTIONAL, ("ops", "jobs", "errs", "combine")),
    "compared": (COMPARED, ("parts",)),
    "generators": (GENERATORS, ("roll", "draws")),
    # A nested cached function's table is its closure's, new each time.
    "nested_memo": (NESTED_MEMO, ("min_distance_up_bottom", "fib_terms", "counted")),
    "counted": (COUNTED, ("majority_vote", "inplace", "solution")),
    # Native code's own: passed between native functions by handle.
    "passed": (PASSED, ("fill", "tally", "drain", "pick", "run")),
    # `operator.add`, `max`, and a `partial` passed where a function goes.
    "values": (VALUES, ("fold", "apply_all")),
}


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )


def _output(done: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(line for line in done.stdout.splitlines() if not line.startswith("compiling"))


def _expected(tmp_path: Path, source: str) -> str:
    _write(tmp_path, source)
    done = _run(tmp_path, "prog.ppy")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    source, natives = PROGRAMS[name]
    expected = _expected(tmp_path, source)
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    source = PROGRAMS[name][0]
    expected = _expected(tmp_path, source)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", "-x", "c", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@requires_llvm
@pytest.mark.parametrize("language", ["c", "cpp"])
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_emitted_source_frees_everything_once(tmp_path: Path, name: str, language: str):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    source = PROGRAMS[name][0]
    expected = _expected(tmp_path, source)
    emitted = tmp_path / f"prog.{language}"
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", "prog.ppy",
        "-o", emitted.name,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    subprocess.run(
        [compiler, standard, "-g", "-O1", "-fsanitize=address,undefined", str(emitted), "-lm",
         "-o", str(binary)],
        check=True,
    )  # fmt: skip
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


INTERLEAVED = """
import random


def native_draws(n: int) -> int:
    total = 0
    for _ in range(n):
        total = total * 7 + random.randint(1, 6)
    return total


def draw_then_overflow(k: int) -> int:
    a = random.randint(1, 100)
    b = a * k
    return b + random.randint(1, 10)


def reseeds(n: int) -> float:
    random.seed(n)
    return random.random()


def main() -> None:
    random.seed(7)
    print(random.vonmisesvariate(2.0, 3.0))
    print(native_draws(10))
    print(random.random(), random.gauss(0.0, 1.0))
    print(draw_then_overflow(3))
    print(draw_then_overflow(2 ** 62))
    print(random.randint(1, 1000))
    print(reseeds(99), random.gauss(0.0, 1.0), random.random())
    print(native_draws(5))


main()
"""


@requires_llvm
@requires_cc
def test_native_and_python_draws_share_one_generator(tmp_path: Path):
    """Python draws between native ones, a native call that falls back after
    it drew (an overflow past a word), and a native `seed`: the sequence is
    CPython's, and so is `gauss`'s held-back value after the reseed."""
    expected = _expected(tmp_path, INTERLEAVED)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    for function in ("native_draws", "draw_then_overflow", "reseeds"):
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


def _runtime(tmp_path: Path) -> ctypes.CDLL:
    from ppy_runtime.collections import library_source

    source = tmp_path / "runtime.c"
    source.write_text(library_source(), encoding="utf-8")
    library = tmp_path / "runtime.so"
    subprocess.run(
        [c_compiler() or "cc", "-O2", "-shared", "-fPIC", str(source), "-lm", "-o", str(library)],
        check=True,
    )
    return ctypes.CDLL(str(library))


@requires_cc
def test_the_generator_is_cpythons_draw_for_draw(tmp_path: Path):
    """Seeds of every size and sign, then `random()`, `getrandbits`,
    `_randbelow`, and `normalvariate` side by side with CPython's."""
    lib = _runtime(tmp_path)
    pointer, word = ctypes.c_void_p, ctypes.c_int64
    lib.ppy_random_state.restype = pointer
    lib.ppy_random_seed_int.argtypes = [pointer, word]
    lib.ppy_random_double.argtypes = [pointer]
    lib.ppy_random_double.restype = ctypes.c_double
    lib.ppy_random_bits.argtypes = [pointer, word]
    lib.ppy_random_bits.restype = word
    lib.ppy_random_below.argtypes = [pointer, word]
    lib.ppy_random_below.restype = word
    lib.ppy_random_normal.argtypes = [pointer, ctypes.c_double, ctypes.c_double]
    lib.ppy_random_normal.restype = ctypes.c_double
    state = lib.ppy_random_state()
    for seed in (0, 1, 42, -7, 2**32 - 1, 2**32, 2**40 + 5, 2**63 - 1, -(2**63)):
        lib.ppy_random_seed_int(state, seed)
        reference = random.Random(seed)
        for i in range(2000):
            if i % 4 == 0:
                assert lib.ppy_random_double(state) == reference.random()
            elif i % 4 == 1:
                assert lib.ppy_random_bits(state, i % 64) == reference.getrandbits(i % 64)
            elif i % 4 == 2:
                n = (i * 7919) % (2**62) + 1
                assert lib.ppy_random_below(state, n) == reference._randbelow(n)
            else:
                assert lib.ppy_random_normal(state, 1.5, 2.0) == reference.normalvariate(1.5, 2.0)


@requires_cc
def test_the_bound_state_is_random_inst(tmp_path: Path):
    """The layout `ppy run` binds to: `random._inst`'s index and words right
    after the object header, which a draw from either side moves."""
    from ppy_runtime.binding import _STATE_BYTES, _random_state_address

    address = _random_state_address()
    assert address is not None
    random.seed(12345)
    raw = ctypes.string_at(address, _STATE_BYTES)
    words = random.getstate()[1]
    assert struct.unpack("i", raw[:4])[0] == words[-1]
    assert struct.unpack("624I", raw[4:]) == words[:-1]
    lib = _runtime(tmp_path)
    lib.ppy_random_bind.argtypes = [ctypes.c_int64]
    lib.ppy_random_state.restype = ctypes.c_void_p
    lib.ppy_random_double.argtypes = [ctypes.c_void_p]
    lib.ppy_random_double.restype = ctypes.c_double
    lib.ppy_random_bind(address)
    bound = lib.ppy_random_state()
    drawn = [random.random() if i % 2 else lib.ppy_random_double(bound) for i in range(10)]
    random.seed(12345)
    assert drawn == [random.random() for _ in range(10)]


ACROSS_PYTHON = """
import random


def py_draw(n: int) -> float:
    return random.vonmisesvariate(1.0, 2.0) + n


def mixed(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += random.random()
        total += py_draw(i)
        total += random.uniform(0.0, 1.0)
    return total


def fallback_after_draw(k: int) -> int:
    a = random.randint(1, 100)
    return a * k + random.randint(1, 10)


def main() -> None:
    random.seed(99)
    print(mixed(8))
    print(random.random())
    print(fallback_after_draw(3))
    print(fallback_after_draw(2**62))
    print(random.random(), mixed(3))


main()
"""


@requires_llvm
@requires_cc
def test_draws_on_both_sides_of_a_call_into_python(tmp_path: Path):
    """Native draws, a call into Python that draws from the same generator (a
    barrier: nothing falls back after it), native draws again, and then a
    call that falls back after it drew: CPython's sequence throughout."""
    expected = _expected(tmp_path, ACROSS_PYTHON)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    for function in ("mixed", "fallback_after_draw"):
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


CACHE_INFO = """
from functools import lru_cache


@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n < 2:
        return n
    total = 0
    for k in range(1, 3):
        total += fib(n - k)
    return total


def main() -> None:
    print(fib(80))
    print(fib.cache_info())
    fib.cache_clear()
    print(fib.cache_info())
    print(fib(10), fib.cache_info().misses)


main()
"""


@requires_llvm
@requires_cc
def test_a_cache_the_program_reads_stays_in_python(tmp_path: Path):
    """`cache_info` and `cache_clear` read Python's cache: the function keeps
    it, and `ppy run` prints CPython's counts."""
    expected = _expected(tmp_path, CACHE_INFO)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.fib")
    assert "reads the cache Python keeps" in explained.stdout, explained.stdout


SHARED_CACHE = """
from functools import cache


@cache
def ways(n: int, k: int) -> int:
    if n == 0:
        return 1
    if n < 0 or k == 0:
        return 0
    total = 0
    for part in range(2):
        if part == 0:
            total += ways(n - k, k)
        else:
            total += ways(n, k - 1)
    return total % 1000000007


def main() -> None:
    print(ways(120, 120))
    print(ways(300, 300))


main()
"""


@requires_llvm
@requires_cc
def test_recursion_goes_through_the_native_table(tmp_path: Path):
    """A dynamic program that is exponential without its cache: the entry
    everything calls, Python's callers included, looks each subproblem up."""
    expected = _expected(tmp_path, SHARED_CACHE)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.ways")
    assert "llvm backend: native" in explained.stdout, explained.stdout


LENT = """
import random


def walk(rng: random.Random, n: int) -> int:
    position = 0
    for _ in range(n):
        position += rng.choice([-1, 1]) * rng.randint(1, 3)
    return position


def overflow_after_draw(rng: random.Random, k: int) -> int:
    a = 0
    for _ in range(3):
        a += rng.randint(1, 100)
    return a * k + rng.randint(1, 10)


def with_gauss(rng: random.Random, n: int) -> float:
    total = 0.0
    for _ in range(n):
        total += rng.random() + rng.gauss(0.0, 1.0)
    return total


def main() -> None:
    rng = random.Random(42)
    print(walk(rng, 1000))
    print(rng.random())
    print(walk(rng, 10), rng.gauss(0.0, 1.0))
    print(overflow_after_draw(rng, 3))
    print(overflow_after_draw(rng, 2**62))
    print(rng.randint(1, 1000))
    print(with_gauss(rng, 5), rng.gauss(0.0, 1.0))
    print(walk(rng, 5))
    other = random.Random(7)
    print(walk(other, 20), walk(rng, 20))


main()
"""


@requires_llvm
@requires_cc
def test_a_generator_python_lends_is_drawn_from_in_place(tmp_path: Path):
    """A `random.Random` Python passes in: native code draws from its state in
    place, interleaved with Python's draws; a call that falls back after it
    drew (a word overflowed) puts the state back first, and `gauss`, whose
    held value is Python's, falls back too."""
    expected = _expected(tmp_path, LENT)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    for function in ("walk", "overflow_after_draw", "with_gauss"):
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)
