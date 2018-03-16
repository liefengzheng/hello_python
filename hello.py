import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import array as array

print(sys.version)
msg ="Hello world"
print("nihao","wo shi zheng")

def func():       
    c = 1
    def foo():
      nonlocal c 
      c = 12
    foo()
    print(c)
func()  

x = np.linspace(0,20,100)

# plt.plot(x, np.sin(x))
# plt.show()
a = np.arange(15).reshape(3, 5)
print(a.shape)
print(a.ndim)
print(a.dtype)

x = np.array([
  [1, 1],
  [0, 1]
])

y = np.array([
  [2, 0, 5],
  [3, 4, 8]
])
# print('x * y:',x * y)
# print('np.multiply(x, y):',np.multiply(x, y))
print('np.dot(x,y):',np.dot(x,y))