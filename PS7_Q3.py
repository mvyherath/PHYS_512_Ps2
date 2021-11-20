import numpy as np
from matplotlib import pyplot as plt

alpha = 1.0

u = np.linspace(0,1,2001)
u = u[1:]  # We skip the zero

# we have u < sqrt(exp(-alpha * r)) so
# log(u^2) = - alpha * (v/u)
# therefore v = (u / alpha) * (-log(u^2))

v = (u / alpha) * (-np.log(u**2))
print('max v is ',v.max())  # Finding the maximum value of v

plt.figure(1)
plt.plot(u,v,'k')
plt.plot(u,-v,'k')
plt.xlabel('u values')
plt.ylabel('v values')
plt.title('Range of v and u')
plt.show()


# The limits on v are between -0.73 and +0.73. 


N = 1000000
u = np.random.rand(N)
# 0.73 seems to be max value of v
v = (np.random.rand(N) * 2 - 1) * 0.73
r = v / u
accept = u < np.exp(-alpha * r)
exp = r[accept]

a, b = np.histogram(exp, 100, range=(0,3))
bb = 0.5 * (b[1:]+b[:-1])
pred = np.exp(-alpha * bb) / np.sqrt(2*np.pi)*np.sum(accept)*(bb[2]-bb[1])
plt.figure(2)
plt.bar(bb,a,0.05, label='Random points')
plt.plot(bb,pred,'r', label='predicted curve')
plt.xlabel('x values')
plt.ylabel('y values')
plt.legend()
plt.show()

# See the figures "PS7_Q3_fig1.png" and "PS7_Q3_fig2.png"


s = len(u)
x = len(exp)
accp = (x / s) * 100
print("The efficiency is", accp, "%")

# This method has an acceptance rate of 67.15%
# It should be noted that the predicted curve and the histograms do not overlap fully.
# The reason for it was not fully understood.
# However, the exponential values were mostly produced using the....
# .... ratio of uniforms method. 
