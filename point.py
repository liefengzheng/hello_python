import math as math
class Point:
  def __init__(self, x=0, y=0):
    super().__init__()
    self.x = x
    self.y = y
    print(self.__class__.__name__)

  def distance_from_original(self):
    return math.hypot(self.x,self.y)

  def __repr__(self):
    return "Point({0.x!r}, {0.y!r})".format(self)

  def __eq__(self,other):
    return self.x == other.x and self.y == other.y
  
  def __str__(self):
    return "({0.x!r},{0.y!r})".format(self)
