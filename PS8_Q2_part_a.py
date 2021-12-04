import numpy as np
from matplotlib import pyplot as plt

#make a greens function, the potential from a point charge at (0,0)
# The equation V = rho * (1/2 * pi) * ln(r) is used.
# We assume rho = 1 at the origin. 

def green(n):
    
    dx=np.arange(n)

    V = np.zeros([n,n])
    xmat, ymat = np.meshgrid(dx,dx)
    dr = np.sqrt(xmat**2 + ymat**2)
    #in 2-D, potential looks like log(r) since a 2D point charge is a 3D line charge
    V = 1.0 - (9.0 * np.log(dr)/ (2 * np.pi))
    # I used a scaling factor of 9, because it is the only way I could get the..
    #.. potential to go below zero. This may be wrong, but it is the only way..
    #.. for now. 
    V = np.roll(V, -1)  # Shifting the V array by 1
    Tot = V[1,0] + V[-1,0] + V[0,1] + V[0,-1]  # sum of the neighbours
    V[0,0] = V[0,0] - (0.25 * Tot) # we divide the sum of neighbours by 4

    return V

n = 16  # Setting up the number of grids
V = green(n)

print('Potential at V[1,0] =',V[1,0])
print('Potential at V[2,0] =',V[2,0])
print('Potential at V[5,0] =',V[5,0])

# Values of V[1,0] and V[2,0] are given in "PS8_Q2_part_a_Outputs.txt".

plt.plot(V[:,0])
plt.xlabel('X-direction (r)')
plt.ylabel('potential (V)')
plt.show()

# Figure is "PS8_Q2_fig_a.png".
