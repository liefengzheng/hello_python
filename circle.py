import point as Shape
import math

class Circle(Shape.Point):
  def __init__(self, radius, x=0, y=0):
    super().__init__(x,y)
    self.radius = radius
  
  @property
  def radius(self):
    """The circle's radius
    """
    return self.__radius
  
  @radius.setter
  def radius(self,radius):
    assert radius > 0,"radius must be nonzero and non-negative"
    self.__radius = radius


  def edge_distance_from_origin(self):
    return abs(self.distance_from_original() - self.radius)

  @property
  def area(self):
    return math.pi * (self.radius ** 2)

  def circumference(self):
    return 2 * math.pi * self.radius