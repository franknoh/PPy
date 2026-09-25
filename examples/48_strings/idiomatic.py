"""The same text processing with Python's own str, dict, and list."""

from collections import Counter, defaultdict

LINES = 200000
LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
PATHS = ["/api/users", "/api/orders", "/static/app.js", "/health", "/api/search"]


def make_log(count):
    return [
        f"2026-09-{i % 28 + 1:02d} {LEVELS[(i * 7) % 4]:<5} {PATHS[(i * 13) % 5]} "
        f"took={(i * 37) % 900 + 0.25:.2f}ms user=u{i % 997}"
        for i in range(count)
    ]


def slowest_paths(lines):
    total = defaultdict(float)
    count = Counter()
    for line in lines:
        fields = line.split()
        total[fields[2]] += float(fields[3].removeprefix("took=").removesuffix("ms"))
        count[fields[2]] += 1
    return "\n".join(
        f"{path:<16}{total[path] / count[path]:>9.2f} ms  x{count[path]:,}" for path in sorted(total)
    )


def levels(lines):
    seen = Counter(line[11:16].strip() for line in lines)
    return " ".join(["levels:", *(f"{level}={n}" for level, n in seen.items())])


def compress(text):
    if not text:
        return ""
    out = []
    last, run = text[0], 0
    for c in text:
        if c == last:
            run += 1
        else:
            out.append(f"{last}{run}")
            last, run = c, 1
    out.append(f"{last}{run}")
    return "".join(out)


def checksum(lines):
    total = 0
    for line in lines:
        for c in compress(line.upper()):
            total = (total * 31 + ord(c)) % 1000000007
    return total


def report(count):
    lines = make_log(count)
    return f"{slowest_paths(lines)}\n{levels(lines)}\n{compress('aaabccddddd')} {checksum(lines)}"


def main():
    print(report(LINES))


main()
