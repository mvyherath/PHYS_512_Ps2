import numpy as np
import matplotlib.pyplot as plt

# COMMENTS ON PROGRAMME OUTPUT WITHIN THE SCRIPT ITSELF.

alpha2 = 1.5
alpha = 1.5  # using two alpha variables in case one of the variables need to..
             #.. be changed. 


n=10000000
q=np.random.rand(n) 
t=(q)**(1/(1-alpha2))

# The reason why t is used is that we are restricting the random nums to just..
# .. numbers that fall within the function that t is using (lorentzian, gauss,etc.).
# Then the random nums of y is multiplied by the function so that it has the..
# .. same height.

y = t**(-alpha2) * np.random.rand(n) * 2  


bins=np.linspace(1,10,501)
aa,bb=np.histogram(t,bins)
cents=0.5*(bins[1:]+bins[:-1])
aa=aa/aa.sum()

# accepting only the values below the power law threshold. 
accept = y < np.exp(-alpha * t)
t_use = t[accept]

aa,bb=np.histogram(t_use,bins)
aa=aa/aa.sum()
pred = np.exp(-alpha * cents)
pred = pred / pred.sum()
plt.plot(cents, aa, '*')
plt.plot(cents, pred, 'r')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

# The figure is called "PS7_Q2_PowerLaw.png".

# It should be noted that if the y values get plotted along with the random......
#.. values of the exponential curve, it would seem like there are regions where..
#.... the power law dips below the exponential curve. 
# This is because both curves get normalized, and then plotted. The rejection..
#....method is implimented before the normalization is done. If the curves are..
#....plotted prior to being normalized, then the power law will be above the....
#....exp curve at all times. 


s = len(y)
x = len(t_use)
accp = (x / s) * 100
print("The efficiency is", accp, "%")

# Using the power law gives an efficiency of 3.716 %.
