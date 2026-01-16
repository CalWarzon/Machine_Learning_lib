import numpy as np
import copy
class x:
  def __init__(self, x):
    self.x = x
class y:
  def __init__(self, y):
    self.y = y
a = [y(2), x(3)]
b = copy.deepcopy(a)
a[1].x = 4
print(a[0].y, a[1].x)
print(b[0].y, b[1].x)