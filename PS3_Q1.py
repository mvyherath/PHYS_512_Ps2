import numpy as np
from matplotlib import pyplot as plt

# Comments at bottom of the script. 

def f(x,y):
    #dy/dx = y/(1+x^2)
    b = (1 + x**2)
    return y / b


def rk4_step(fun,x,y,h):
    rk4_step.counter += 1
    k0=fun(x,y)*h
    k1=fun(x+h/2,y+k0/2)*h
    k2=fun(x+h/2,y+k1/2)*h
    k3=fun(x+h,y+k2)*h
    return (k0+2*k1+2*k2+k3)/6

# Function for running different RK4 step sizes
# The delta value and the final output from rkf_stepd determined ....
# with section 17.1 of Numerical Recipes. 

def rk4_stepd(fun,x,y,h):
    rk4_stepd.counter += 1
    # running step size of h
    y1 = rk4_step(fun,x,y,h)
    # running two steps of h/2
    y2 = rk4_step(fun,x,y,h/2)
    y3 = rk4_step(fun,(x+h/2),y2,h/2)
    delta = y3 - y1
    return y3 + (delta/15)

rk4_step.counter=0   # Counters for the two RK4 functions
rk4_stepd.counter=0


nstep=200
x=np.linspace(-20,20,nstep+1)
y=0*x
y[0]=1  # y(-20) = 1

# The loops below can be commented out depending on which RK4 integrator....
# gets used

#for i in range(nstep):
#    h=x[i+1]-x[i] 
#    y[i+1]=y[i]+rk4_step(f,x[i],y[i],h)

for i in range(nstep):
    h=x[i+1]-x[i] 
    y[i+1]=y[i]+rk4_stepd(f,x[i],y[i],h)

y_pred=np.exp(np.arctan(x))

#print('RK4_step called', rk4_step.counter, 'times')
print('RK4_stepd called', rk4_stepd.counter, 'times')

print(np.std(y-y_pred))

# Outputs in accompanying .txt file.
# Q1 PART b)
# RK4_step uses 4 function evaluations per step.
# RK4_stepd uses 12 function evaluations per step.
# Using RK4_step gives a standard deviation of 7 for 200 steps.
# Using RK4_stepd gives a standard deviation of 2 for 200 steps.
# RK4_stepd is better. 


