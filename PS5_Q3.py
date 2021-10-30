import numpy as np
from matplotlib import pyplot as plt

# COMMENTS at the BOTTOM

def Gauss(x):
    sigma = 1.0
    num = np.exp(-0.5 * x**2 / (sigma**2))
    return num


def shift_array(arr, n):
    
    # if we are shifting by 1, then we need 2 for the zeros array
    # if shifting by 2, we need 3 in the array. n = n + 1
    
    n_new = n + 1
    arr_s = np.zeros(n_new)  # creating an array of zeros
    arr_s[-1] = 1            # make final value = 1
    arr_s = np.pad(arr_s, (0, np.size(arr) - np.size(arr_s)))  # pad with zeros

    # do the convolution by fft'ing each of the arrays and getting inverse
    f1 = np.fft.fft(arr)
    f2 = np.fft.fft(arr_s)
    arr_shifted = np.real(np.fft.ifft(f1 * f2)) # getting the inverse Fourier
    
    return arr_shifted



# The Correlation function is created

def corr(arr1, arr2):
    f1 = np.fft.fft(arr1)
    f2 = np.fft.fft(arr2)
    conj_f2 = np.conj(f2)
    corr_arr = np.real(np.fft.ifft(f1 * conj_f2))    
        
    return corr_arr



arr1 = np.arange(-10, 10, 0.1)  # creating initial array
arr2 = shift_array(arr1, 40)    # shifting the array by 20% of array length
arr3 = np.copy(arr2)            # copying the initial array

gauss1 = Gauss(arr1)            # Gaussian for initial array
gauss2 = Gauss(arr2)            # Gaussian for shifted array
gauss3 = Gauss(arr3)            # Ditto above

# correlation func between original Gaussian and shifted Gaussian
corr_gauss = corr(gauss1, gauss2)
# Correlation of shifted Gaussian with itself
corr_gauss_shifted = corr(gauss2, gauss3)

# plots of correlation function for shifted Gaussians
plt.plot(arr1, corr_gauss_shifted, label='Correlation function')
plt.plot(arr1, gauss1, label='Gaussian function')
plt.xlabel('X-values')
plt.ylabel('Y-values')   # Outputs given in "PS5_Q3_CorrShifted.png"
plt.legend()
plt.show()  # plot saved as "PS5_Q2_CorrelationFunc.png"

# plots of correlation function for original Gaussian and shifted Gaussian
plt.plot(arr1, corr_gauss, label='Correlation function')
plt.plot(arr1, gauss1, label='Gaussian function')
plt.plot(arr1, gauss2, label='Shifted Gaussian function')
plt.xlabel('X-values')
plt.ylabel('Y-values')   # Outputs given in "PS5_Q3_CorrOrigShift.png"
plt.legend()
plt.show()

# COMMENTS:
# The correlation function of the shifted Gaussians look the same as in Q2.
# The shift seems to be happening at the cusp of the wrap-around effect.
# No surprises here as it makes sense that despite the shit, the correlation....
# ..... func stays the same because it only looks at the deviation btwn.....
# ..... two functions.
# For comparison, look at the Corr func btwn the original un-shifted Gaussian...
# ..... and the shifted Gaussian.
# That particular correlation function peaks away from both Gaussians.
# The shift shows how much of a shift is required to align the two.....
# ..... Gaussian Functions.
