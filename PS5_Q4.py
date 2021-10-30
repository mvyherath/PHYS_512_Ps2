import numpy as np
from matplotlib import pyplot as plt

# COMMENTS at BOTTOM of the script

# Here be our Gaussian

def Gauss(x):
    sigma = 1.0
    num = np.exp(-0.5 * x**2 / (sigma**2))
    return num

# Here's the regular convolution function

def conv(f, g):
    
    f1 = np.fft.fft(f)
    f2 = np.fft.fft(g)
    arr_con = np.real(np.fft.ifft(f1 * f2))
    
    return arr_con


# Here's the safe convolution function

def conv_safe(f, g):

    # the two arrays are initially of different sizes (size(g) = 0.5*size(f))
    # we take the f and g functions and run them through the Gaussians
    gauss1 = Gauss(f)
    gauss2 = Gauss(g)

    # the gaussian arrays are padded with zeros according to their sizes
    # the larger array is doubled in size, and the excess is padded with zeros
    # the smaller array is also padded with zeros
    pad_f = np.pad(gauss1, (0, np.size(gauss1)))
    pad_g = np.pad(gauss2, (0, np.size(pad_f) - np.size(gauss2)))
    pad_f = pad_f / pad_f.sum()
    pad_g = pad_g / pad_g.sum()

    # getting the Fourier transforms of the padded functions
    # rounded off with the inverse Fourier
    
    f1 = np.fft.fft(pad_f)
    f2 = np.fft.fft(pad_g)
    arr_con = np.real(np.fft.ifft(f1 * f2))
    
    return arr_con


a = np.linspace(-10, 10, 200)  # array f
b = np.linspace(-10, 10, 100)  # array g
b_o = np.linspace(-10, 10, 200) # a separate array to do a non-zero-padded convolution
c = np.linspace(-10, 10, 400) # array for convolved function

gauss1 = Gauss(a) # Gaussian for array f
gauss2 = Gauss(b) # Gaussian for array g
gauss1 = gauss1 / gauss1.sum()
gauss2 = gauss2 / gauss2.sum()

gauss3 = Gauss(b_o)  # Gaussian like gauss1 with the same array size
gauss3 = gauss3 / gauss3.sum()


new_gauss = conv_safe(a, b) # running the functions through conv_safe
# plotting the padded Gaussians and convolution
plt.plot(gauss1, color='blue', label='First Gaussian')
plt.plot(gauss2, color='orange', label='Second Gaussian')
plt.plot(new_gauss, color='green', label='Convolved function')
plt.xlabel('X-values')
plt.ylabel('Y-values')   # Outputs given in "PS5_Q4_ConvPadded.png"
plt.legend()
plt.show()

other_gauss = conv(gauss1, gauss3) # running the functions through conv
# plotting the padded Gaussians and convolution
plt.plot(gauss1, color='blue', label='First Gaussian')
plt.plot(gauss3, '--', color='orange', label='Second Gaussian')
plt.plot(other_gauss, color='green', label='Convolved function')
plt.xlabel('X-values')
plt.ylabel('Y-values')   # Outputs given in "PS5_Q4_ConvNonPadded.png"
plt.legend()
plt.show()


# COMMENTS:
# A comparison between the zero-padded and non-zero padded functions are made.
# The two arrays were made with different sizes.
# The size of the bigger array was doubled, and the excess was padded with zeros.
# The smaller array (g) was padded with zeros to match the size-doubled array (f)
# Convolution measures the total area resulting form the two functons.
# During convolution, one of the functions travels right across the domain.
# If array g was the only array that was padded, with array f having all.....
# ..... integer values with no padding, the wrap around effect would occur.
# This occurs because there is nowhere else for the excess of the bigger array to go.
# By padding the bigger array with excess zeros, it ensures that the excess ......
# ..... of the function gets damped out before getting convolved as a ......
# ..... wrap-around.
# In "PS5_Q4_ConvPadded.png", we see how the convolved function is different ....
# ..... from "PS5_Q4_ConvNonPadded.png" because of the zero padding.
