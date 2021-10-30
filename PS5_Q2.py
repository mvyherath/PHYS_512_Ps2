import numpy as np
from matplotlib import pyplot as plt

# COMMENTS at the BOTTOM

# The Gaussian function is created

def Gauss(x):
    sigma = 1.0
    num = np.exp(-0.5 * x**2 / (sigma**2))
    return num

# The Correlation function is created

def corr(arr1, arr2):
    f1 = np.fft.fft(arr1)  # Fourier trnasform of array 1
    f2 = np.fft.fft(arr2)  # Fourier trnasform of array 2
    conj_f2 = np.conj(f2)  # Complex conjugate of Fourier transformed array 2
    corr_arr = np.real(np.fft.ifft(f1 * conj_f2))  # Inverse Fourier transform  
        
    return corr_arr



arr1 = np.arange(-10, 10, 0.1)  # creating array
arr2 = np.copy(arr1)            # copying array

gauss = Gauss(arr1)             # making the Gaussian
corr_gauss = corr(gauss, gauss) # getting the correlation of Gaussian with itself

plt.plot(arr1, corr_gauss, label='Correlation function')  # plot of correlation function
plt.plot(arr1, gauss, label='Gaussian function')       # plot of original Gaussian
plt.xlabel('X-values')
plt.ylabel('Y-values')   # Outputs given in "PS5_Gauss_Shift.png"
plt.legend()
plt.show()  # plot saved as "PS5_Q2_CorrelationFunc.png"

# COMMENTS:
# The correlation function looks like a shifted Gaussian.
# The shift seems to be happening at the cusp of the wrap-around effect
# Correlations show how similar two functions are to each other
# So the Corr. of a Gaussian with itself would show near-zero values.......
# ...around the same point when the peaks of the two Gaussians overlap.
# The more we move away from the overlapping peaks, the more the correlation....
# .....func seems to deviate from the zero line.
# At least that's how I understand this. 
