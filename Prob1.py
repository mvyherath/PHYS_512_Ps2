import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import *


# Setting up a function to manually integrate

def my_integrate(f, a, b, n):

    #n is the step size
    dx = float(b - a) / n
    result = (0.5 * f(a)) + (0.5 * f(b))

    for i in range(1, n):
        result += f(a + (i*dx))
    result *= dx

    return result

# End of setting up manually integrating function



# The constants are set up over here before being used by the resulting integral

R = 2.0  #radius of the shell
E = 1.6 * (10**(-19))  # value of an electron-volt
sigma = 10000.0 * E  #charge density of the shell. Gave it a large value to make the calculations easier.
epsilon = 8.85 * (10**(-12)) 
K = (1.0 / (4.0 * pi * epsilon)) * (2.0 * pi * (R**2.0) * sigma)  #The constant outside the integral

# End of setting up the constants

# Set up the function to integrate. The function in this case is the integrand for finding the E-field from a spherical shell at a distance z. The integral was derived from Griffiths Problem 2.7.

def d_Ef(z,u):
    numerator = z - (R * u)
    denominator = (R**2.0) + (z**2.0) - (2.0 * R * z * u) 
    int = float(numerator) / float(denominator**(3.0/2.0)) 
    return K * int

z_vals_quad = []  # array to store the distance values with quad
E_vals_quad = []  # array to store the E-field values with quad
z_vals_MyInt = [] # array to store the distance values with MyInt
E_vals_MyInt = [] # array to store the E-field values with MyInt

for z in range(0,67,1):
    d_E = lambda u: d_Ef(z,u)      # an element of the electric field
    result_quad = quad(d_E, -1, 1) # integration happens here
    z_vals_quad.append(z)
    E_vals_quad.append(result_quad)

    result_MyInt = my_integrate(d_E, -1, 1, 600)
    z_vals_MyInt.append(z)
    E_vals_MyInt.append(result_MyInt)


plt.plot(z_vals_quad, E_vals_quad)
plt.xlabel("Radius (no units)")
plt.ylabel("E-field (no units)")
plt.show()
#plt.plot(z_vals_MyInt, E_vals_MyInt)
#plt.xlabel("Radius (no units)")
#plt.ylabel("E-field (no units)")
#plt.show()
