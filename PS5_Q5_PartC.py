import numpy as np
from matplotlib import pyplot as plt

# Solutions for PS5 - Question 5 - Part C
# COMMENTS at the BOTTOM

# We approximate a sine wave using exponential functions where.....
# Sin (xt) = (exp(jxt) - exp(-jxt)) / 2j

J = np.complex(0, 1)

# This is the function from 5a

def func_fft(k_dft, k_sin, N):
    fft = np.zeros(len(k_dft), dtype=complex)  # complex array of zeros
    for i in range(len(k_dft)):
        num1 = (1 - np.exp(-2 * np.pi * 1J * (k_dft[i] + k_sin)))
        den1 = (1 - np.exp(-2 * np.pi * 1J * (k_dft[i] + k_sin) / (N + 1)))
        num2 = (1 - np.exp(-2 * np.pi * 1J * (k_dft[i] - k_sin)))
        den2 = (1 - np.exp(-2 * np.pi * 1J * (k_dft[i] - k_sin) / (N + 1)))

        fft1 = num1 / den1  # first exponential
        fft2 = num2 / den2  # second exponential

        fft[i] = (fft1 - fft2) / (2*J)

    return np.abs(fft)


N = 16
f = 3.5  # frequency for Sine wave
sample = 100

# Making the Sine wave
x = np.arange(sample)
y = np.sin(2 * np.pi * (x - f) / N)
#y = y / y.sum()

# Analytical function
y_a = func_fft(x, f, N)

# Plots for both functions
plt.plot(x, y_a, color='blue', label='Analytical function')
plt.plot(x, y, color='orange', label='Sine wave')
plt.xlabel('Frequencies')
plt.ylabel('Magnitude')
plt.legend()
plt.show()


# COMMENTS:
# The plots from the analytic function and the sine function don't seem ......
# ...... to match up precisely.
# Not quite sure how to make them match up better.
# The main problem is the upwards shift of the analytic function.
# Normalizing does not seem to improve it. 


    
