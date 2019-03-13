import numpy as np
from numpy import linalg as la


a = np.array([[1,2],[3,4],[5,6]])

b = np.array([[1],[2]])

x = np.array([1,2,3])
y = ([1,2,3])

print(la.norm(y))


x = [1,2,3]
y=x
y[0] = 10
print(y)
print(x)