import numpy as np
a = 5.5
b = np.floor(a)
# print(b)
# print(b.dtype)
c = np.random.randint(1, 10, 20)
print(c)
d = [1, 2, 3]
e = d.copy()
print(e)
e.append(4)
print(d, e)
print(e[0])
for i in range(10, 70):
    print(i)