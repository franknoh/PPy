def mean(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def variance(values):
    m = mean(values)
    total = 0.0
    for value in values:
        total += (value - m) * (value - m)
    return total / len(values)


def standardize(values):
    m = mean(values)
    spread = variance(values) ** 0.5
    out = []
    for value in values:
        out.append((value - m) / spread)
    return out


def report(values, label):
    return f"{label}: mean={mean(values):.3f} var={variance(values):.3f}"


samples = [1.0, 2.0, 3.0, 4.0, 5.0]
print(report(samples, "samples"), [round(x, 4) for x in standardize(samples)])
