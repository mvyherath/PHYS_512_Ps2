import numpy as np
from matplotlib import pyplot as plt

# Comments at the BOTTOM of the script

# Here's the Gaussian function

def Gauss(x):
    sigma = 1.0
    num = np.exp(-0.5 * x**2 / (sigma**2))
    return num


def shift_array(arr, n):
    
    # if we are shifting by 1, then we need 2 for the zeros array
    # if shifting by 2, we need 3 in the array. n = n_0 + 1
    
    n_new = n + 1
    arr_s = np.zeros(n_new)  # creating an array of zeros
    arr_s[-1] = 1            # make final value = 1
    arr_s = np.pad(arr_s, (0, np.size(arr) - np.size(arr_s)))  # pad with zeros

    # do the convolution by fft'ing each of the arrays and getting inverse
    f1 = np.fft.fft(arr)
    f2 = np.fft.fft(arr_s)
    arr_shifted = np.real(np.fft.ifft(f1 * f2))
    
    return arr_shifted


a = np.arange(-10, 10, 0.1)   # our starting array
new_arr = shift_array(a, 100)  # shifting by half an array length (i.e. 100)
print(len(a),len(new_arr))


first_gauss = Gauss(a)   # using the function to make a Gaussain
second_gauss = Gauss(new_arr)  # shifting the Gaussian
plt.plot(a, first_gauss, label='Initial array')
plt.plot(a, second_gauss, label='Shifted array')
plt.xlabel('X-values')
plt.ylabel('Y-values')   # Outputs given in "PS5_Gauss_Shift.png"
plt.legend()
plt.show()

# COMMENTS:
# When shifted by half a wavelength, we start seeing the wrap-around effect...
# starting to appear.
# An additional plot with just a shift of 25% of array length is shown....
# for comparison (PS5_Gauss_First_Shift.png).

