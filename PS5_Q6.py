import numpy as np
from matplotlib import pyplot as plt

# COMMENTS at the BOTTOM of the script

# Here's the correlation function

def corr(arr1, arr2):
    f1 = np.fft.fft(arr1)
    f2 = np.fft.fft(arr2)
    conj_f2 = np.conj(f2)
    corr_arr = np.real(np.fft.ifft(f1 * conj_f2))    
        
    return corr_arr


n = 1000  # number of data points
# creating an array of cumulative sums of 1000 random numbers between 0 and 1
rando = np.cumsum(np.random.randn(n))

# rando is our random walk function

# getting the correlation function of the random walk with itself
corr_rando = corr(rando, rando)  
pow_rando = np.abs(np.fft.fft(corr_rando)) # the power spectrum of the random walk
pow_rando = pow_rando / pow_rando.sum()  # normalization of power spectrum


f = np.abs(np.fft.fftfreq(n, (1/n))) # getting frequency values for n
f = f[1:]                            # frequencies ignoring the first value
f_inv = 1 / (f**2)

plt.loglog(pow_rando[1:], color='blue', label='Random walk power spectrum')  # ignoring the first term
plt.loglog(f_inv, color='orange', label='1/K^2')
plt.xlabel('log-frequncy')
plt.ylabel('log-power')
plt.legend()
plt.show()


# COMMENTS:
# The power spectrum is the Fourier transform of the correlation function.
# In this case the correlation function is that of random jumps in a .......
# ..... random walk.
# It can be seen that the power spectrum of the random walk and the 1/k^2 .....
# ..... are following close trajectories.
# There we prove that the power spectrum of a random walk follows a 1/K^2 relation.
# The shift betweent he two curves probably corresponds to "C". 



