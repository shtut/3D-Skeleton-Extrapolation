import sympy as sy
import numpy as np
from IPython.display import display

w1, w2, w3, w4 = sy.symbols("w1 w2 w3 w4", real=True)
W = sy.Matrix([[w1, 0, w2], [0, w1, w3], [w2,w3,w4]])

v1 = np.array([1,1,1])
v2 = np.array([1,1,1])
v3 = np.array([1,1,1])

print v1
print v2
print v3
print W
display( v1.T * W * v2)