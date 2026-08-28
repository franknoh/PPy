import math


def line_value(item):
    return item[1] * item[2]


def total_value(items):
    total = 0.0
    for item in items:
        total += line_value(item)
    return total


def total_units(items):
    total = 0
    for item in items:
        total += item[1]
    return total


def discounted(price, discount):
    return price * (1.0 - discount)


def apply_discount(items, discount):
    out = []
    for item in items:
        out.append((item[0], item[1], discounted(item[2], discount)))
    return out


def below_stock(items, threshold):
    low = []
    for item in items:
        if item[1] < threshold:
            low.append(item[0])
    return low


def summary(items):
    value = total_value(items)
    count = len(items)
    if not count:
        return (0.0, 0.0, 0.0)
    return (value, value / count, math.sqrt(value))


def labeled(name, value):
    return name + "=" + str(round(value, 2))


def main():
    items = [
        ("bolt", 120, 0.25),
        ("nut", 340, 0.1),
        ("washer", 80, 0.05),
        ("bracket", 15, 12.5),
    ]

    value, average, root = summary(items)
    print(labeled("total", value), labeled("avg", average), labeled("root", root))
    print(apply_discount(items, 0.2)[0])
    print(below_stock(items, 100))
    print("units:", total_units(items))


main()
