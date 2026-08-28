def widest(rects):
    best = None
    for rect in rects:
        if best is None or rect.width > best.width:
            best = rect
    return best


class Rect:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def scaled(self, factor):
        return Rect(self.width * factor, self.height * factor)


def total_area(rects):
    total = 0.0
    for rect in rects:
        total += rect.area()
    return total


def label(rect, prefix):
    if rect is None:
        return prefix + ":none"
    return f"{prefix}:{rect.width}x{rect.height}"


def build(count):
    out = []
    for i in range(count):
        out.append(Rect(float(i), float(i) * 2.0))
    return out


boxes = build(4)
print(total_area(boxes))
print(label(widest(boxes), "max"))
print(label(widest([]), "none") if widest([]) is None else 0.0)
