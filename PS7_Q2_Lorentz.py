import numpy as np
import matplotlib.pyplot as plt

# COMMENTS ON PROGRAMME OUTPUT WITHIN THE SCRIPT ITSELF.

alpha = 1.5

# Creating the Lorentzian function

def lorentzians(n):
    q = np.pi*(np.random.rand(n) + 0.5) 
    return np.tan(q)

n=10000000
t=lorentzians(n)


# The reason why t is used is that we are restricting the random nums to just..
# .. numbers that fall within the function that t is using (lorentzian, gauss,etc.).
# Then the random nums of y is multiplied by the function so that it has the..
# .. same height.

y = 1.0 / (1 + t**2) * np.random.rand(n) * 2 


bins=np.linspace(0,10,501)
aa,bb=np.histogram(t,bins)
cents=0.5*(bins[1:]+bins[:-1])
aa=aa/aa.sum()

# accepting only the values below the Lorentzian threshold. 

accept = y < np.exp(-alpha * t)
t_use = t[accept]

aa,bb=np.histogram(t_use,bins)
aa=aa/aa.sum()
pred = np.exp(-alpha * cents)
pred = pred / pred.sum()
plt.plot(cents, aa, '*', label='Random points')
plt.plot(cents, pred, 'r', label='predicted curve')
plt.xlabel('x values')
plt.ylabel('y values')
plt.legend()
plt.show()

# The figure is called "PS7_Q2_Lorentz_fig.png".

# It should be noted that if the y values get plotted along with the random......
#.. values of the Lorentz curve, it would seem like there are regions where..
#.... the Lorentzian dips below the exponential curve. 
# This is because both curves get normalized, and then plotted. The rejection..
#....method is implimented before the normalization is done. If the curves are..
#....plotted prior to being normalized, then the Lorentzian will be above the....
#....exp curve at all times. 


s = len(y)
x = len(t_use)
accp = (x / s) * 100
print("The efficiency is", accp, "%")

# Using the Lorentzian gives an efficiency of 57.16 %.
