import numpy as np

a = np.array([
  [5, 1, 3],
  [1, 1, 1],
  [1, 2, 1]
])

b = np.array([1,2,3])

# print (a @ b)
# print (a * b)
# print (np.dot(a,b))

p =(4, 'frog', 9, -3.1)
print(9 in p)
print('dog' not in p)

s = input("enter an integer:")
try:
  i = int(s)
  print("valid integer extered:", i)
except ValueError as err:
  print(err)
