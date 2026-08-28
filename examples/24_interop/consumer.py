import ppy

import geometry

print("area     :", geometry.area(3.0, 4.0))
print("perimeter:", geometry.perimeter(3.0, 4.0))
print("loaded   :", geometry.__file__.rsplit("/", 1)[-1])
print("hook     :", ppy.is_installed())
