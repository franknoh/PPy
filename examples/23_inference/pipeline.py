import math


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def normalize(value, mean, spread):
    return clamp((value - mean) / spread, -3.0, 3.0)


def summarize(readings):
    count = len(readings)
    if not count:
        return None
    total = 0.0
    for i in range(count):
        total += readings[i]
    mean = total / count
    spread = 0.0
    for i in range(count):
        spread += (readings[i] - mean) * (readings[i] - mean)
    return Summary(mean, math.sqrt(spread / count), count)


class Summary:
    def __init__(self, mean, deviation, count):
        self.mean = mean
        self.deviation = deviation
        self.count = count

    def scaled(self, factor):
        return Summary(self.mean * factor, self.deviation * factor, self.count)

    def shifted(self, offset):
        return Summary(self.mean + offset, self.deviation, self.count)

    def describe(self):
        return f"n={self.count} mean={self.mean:.3f} sd={self.deviation:.3f}"


def standardized(readings, summary):
    out = []
    for reading in readings:
        out.append(normalize(reading, summary.mean, summary.deviation))
    return out


def report(readings):
    summary = summarize(readings)
    if summary is None:
        return "empty"
    values = standardized(readings, summary)
    return summary.describe() + " first=" + str(round(values[0], 4))


samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
print(report(samples))
print(report([]))

overall = summarize(samples)
if overall is not None:
    print(overall.scaled(2.0).describe())
