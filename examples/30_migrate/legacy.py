"""A little legacy telemetry script, dynamic in all the usual harmless ways."""

import importlib

math = importlib.import_module("math")

globals()["SCALE"] = 4


class Reading:
    def __init__(self, value):
        self.value = value
        self.flagged = False


def normalize(readings):
    out = []
    for reading in readings:
        out.append(reading.value * SCALE)
    return out


def spread(values):
    return math.ceil(max(values) - min(values))


samples = [Reading(0.5), Reading(1.25), Reading(2.0)]
first = samples[0]
setattr(first, "flagged", True)

scaled = normalize(samples)
print(getattr(first, "flagged"), scaled, spread(scaled))
