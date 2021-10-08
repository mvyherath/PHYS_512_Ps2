import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import time

# Function created to integrate all 15 reaction chains in the decay of U-238 to Pb-206.
# I included all half lives in the list call inside the function. 
# Plots included in repository (names of plots next to plt.plot commands)
# Answers to questions right at the bottom

def fun(x,y,half_life=[4.468e9,6.6e-2,7.6e-4,2.45e5,7.5e4,1.6e3,1e-2,5.9e-6,5.1e-5,3.8e-5,5.2e-12,22.3,5e3,3.8e-1]):
    #The half lives are all in a scale of years.
    const = np.log(2)
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]*const/half_life[0]                          # Eqn for Uranium-238
    dydx[1]=y[0]*const/half_life[0]-y[1]*const/half_life[1]   # Eqn for Thorium-234
    dydx[2]=y[1]*const/half_life[1]-y[2]*const/half_life[2]   # Eqn for Protactinum-234
    dydx[3]=y[2]*const/half_life[2]-y[3]*const/half_life[3]   # Eqn for Uranium-234
    dydx[4]=y[3]*const/half_life[3]-y[4]*const/half_life[4]   # Eqn for Thorium-230
    dydx[5]=y[4]*const/half_life[4]-y[5]*const/half_life[5]   # Eqn for Radium-226
    dydx[6]=y[5]*const/half_life[5]-y[6]*const/half_life[6]   # Eqn for Radon-222
    dydx[7]=y[6]*const/half_life[6]-y[7]*const/half_life[7]   # Eqn for Polonium-218
    dydx[8]=y[7]*const/half_life[7]-y[8]*const/half_life[8]   # Eqn for Lead-214
    dydx[9]=y[8]*const/half_life[8]-y[9]*const/half_life[9]   # Eqn for Bismuth-214
    dydx[10]=y[9]*const/half_life[9]-y[10]*const/half_life[10] # Eqn for Polonium-214
    dydx[11]=y[10]*const/half_life[10]-y[11]*const/half_life[11] # Eqn for Lead-210
    dydx[12]=y[11]*const/half_life[11]-y[12]*const/half_life[12] # Eqn for Bismuth-210
    dydx[13]=y[12]*const/half_life[12]-y[13]*const/half_life[13] # Eqn for Polonium-210
    dydx[14]=y[13]*const/half_life[13]                           # Eqn for Lead-206
    return dydx


y0=np.asarray([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])  # Storage array for all quantities of elements
x0=0
x1=16000000000                                  # 16 billion years as the final time step
step_size = 1000000                             # Step size of a million years
t = np.arange(x0, x1, step_size)                # Setting the time steps

t1=time.time()
ans_stiff=solve_ivp(fun,[x0,x1],y0,method='Radau', t_eval= t)  # Radau solving the ODE's
t2=time.time()
print('took ',ans_stiff.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly')
print('final values were ',ans_stiff.y[0,-1],' with truth ',np.exp(-1*(x1-x0)))

time = ans_stiff.t        # Extracting the time array from the Radau output matrix
y_new = ans_stiff.y.T     # Extracting the Y array from the matrix and transposing to align the time and y arrays
y_U_238 = y_new[:,0]      # Extracting array for U-238
y_Pb = y_new[:,14]        # Extracting array for Pb-206
y_U_234 = y_new[:,3]      # Extracting array for U-234
y_Th = y_new[:,4]         # Extracting array for Th-230

ratio_Pb_U238 = np.divide(y_Pb,y_U_238)   # Creating list of ratios of Pb-206/U-238
ratio_Th_U234 = np.divide(y_Th,y_U_234)   # Creating list of ratios of Th-230/U-234

#plt.plot(t, y_new)   # Plotting all the element chains 
#plt.yscale('log')    # Plotting all the element chains logarithmically
#plt.show()

#Plot of U-238 and Pb-206. SEE FIGURE PS3_Q2_Plot1
plt.plot(t,y_U_238,label='Uranium-238')
plt.plot(t,y_Pb,label='Lead-206')

#Plot of U-234 and Th-230. SEE FIGURE PS3_Q2_Plot2
#plt.plot(t,y_U_234,label='Uranium-234',color='black')
#plt.plot(t,y_Th,label='Thorium-230',color='purple')

#Plot of U-238 and Pb-206 ratio. SEE FIGURE PS3_Q2_Plot3
#plt.plot(t, ratio_Pb_U238,label='ratio:Pb-206/U-238',color='green')

#Plot of U-234 and Th-230. SEE FIGURE PS3_Q2_Plot4
#plt.plot(t, ratio_Th_U234,label='ratio:Th-230/U-234',color='red')
# NOTE: Spikes at beginning of Th-230 and U-234 plots on account of their starting values being zero.
plt.legend()
plt.xlabel('Time (t-years)')
plt.ylabel('Quantity (N-unitless)')
plt.show()

# Q2: PART a)
# Used the solve_ivp Radau solver.

#Q2: PART b)
# Plot of U-238 and Pb-206 intersect around 5 billion years. 
# I guess 5 billion years makes sense analytically as U-238 has a 4.4 billion year half life. 
# The ratios between U-238 and Pb-206 show a steady increase.  
# This is because Pb-206 increases as U-238 decreases
# Ratio of U-234 to Th-230 stays costant at a value of Th_230/U-234 = 0.306

