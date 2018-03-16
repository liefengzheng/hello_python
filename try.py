a = ["1","b","2","aa"]
list.append(a,"extra")
list.insert(a,1,"second")
a.append("cannt")
print(a[1])

b = ("tuple1","tuple2")
print(b[1])

c = ["sunsine","flower"]
c +=["woshi"]
print(c)

import sys
print(sys.argv)

import random as rd
for num in range(5):
  print(rd.randint(1,100))

# print(rd.choice(["x","y","z"]))
s="he ate camel food"
print(s[1:])
print(s.count('e'))
print(s.split(' '))

a1 =(1,2)
a2= (3,4)
a1,a2 = a2,a1
print(a1)