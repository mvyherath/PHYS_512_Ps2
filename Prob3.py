import numpy as np
import numpy.polynomial.chebyshev as cheb
from math import *


x = np.linspace(0.0001, 1, 600)  # x array from 0 to 1
y = np.log2(x)                   # getting the y values

print(len(x), len(y))

coeffs = cheb.chebfit(x, y, 3)    # getting the cheb coefficients by fitting between 0 and 1
  
#print(coeffs)

x2 = np.linspace(0.5, 1, 200)
f = cheb.chebval(x2, coeffs, tensor=True)  # finding the fit between 0.5 and 1 using the cheb coefficients from the previous fit. 

print(f)

#rescaling to find mantissa

y1,y2 = np.frexp(x2)
y = np.log2(y1) + y2


