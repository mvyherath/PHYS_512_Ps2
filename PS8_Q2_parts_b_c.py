import numpy as np
from matplotlib import pyplot as plt

# This script gives the code for both parts (b) and (c) of PS8-Q2.
# The figure "PS8_Q2_BOX_ref.png" shows the square box used for parts b and c.
# Comments given in the script itself.


def green(n):
    
    dx=np.arange(n)
    dx[n//2:]=dx[n//2:]-n
    V = np.zeros([n,n])
    xmat, ymat = np.meshgrid(dx,dx)
    dr = np.sqrt(xmat**2 + ymat**2)
    dr[0,0]=1
    #in 2-D, potential looks like log(r) since a 2D point charge is a 3D line charge
    V = 1.0 - (9.0 * np.log(dr)/ (2 * np.pi))
    V = V - V[n//2,n//2]
    # I used a scaling factor of 9, because it is the only way I could get the..
    #.. potential to go below zero. This may be wrong, but it is the only way..
    #.. for now. 
    V = np.roll(V, -1)  # Shifting the V array by 1
    Tot = V[1,0] + V[-1,0] + V[0,1] + V[0,-1]  # sum of the neighbours
    V[0,0] = V[0,0] - (0.25 * Tot) # we divide the sum of neighbours by 4

    return V
    

#convolve the density with the Greens function to get the potential

def rho2pot(rho,kernelft):
    tmp=rho.copy()
    tmp=np.pad(tmp,(0,tmp.shape[0]))

    tmpft = np.fft.rfftn(tmp)
    tmp = np.fft.irfftn(tmpft * kernelft)
    if len(rho.shape) == 2:
        tmp = tmp[:rho.shape[0],:rho.shape[1]]
        return tmp
    if len(rho.shape) == 3:
        tmp = tmp[:rho.shape[0],:rho.shape[1],:rho.shape[2]]
        return tmp
    print("error in rho2pot - unexpected number of dimensions")

# Solving for Ax = b. "x" is the charge, while b = potential on the boundary. 
# A convolves the charge with the Green's function for the current charge distribution..

def rho2pot_masked(rho, mask, kernelft,return_mat = False):
    rhomat = np.zeros(mask.shape)
    rhomat[mask] = rho
    potmat = rho2pot(rhomat,kernelft)

    if return_mat:
        return potmat
    else:
        return potmat[mask]

# The Conjugate Gradient function is written following the equations in the ..
# .. Wiki page

def cg(rhs,x0,mask,kernelft,niter,fun=rho2pot_masked,show_steps=False,step_pause=0.01):

    # A is the Laplacian Operator in matrix form.
    # b = contribution from boundary conditions.
    Ax = fun(x0, mask, kernelft)
    r = rhs - Ax
    p = r.copy()
    x = x0.copy()
    rsqr = np.sum(r*r)
    
    print('starting rsqr is ',rsqr)
    for k in range(niter):

        Ap = fun(p, mask, kernelft)
        alpha = np.sum(r * r) / np.sum(Ap * p)
        x = x + (alpha * p)
        r = r - alpha * Ap
        rsqr_new = np.sum(r * r)
        beta = rsqr_new / rsqr
        p = r + beta * p
        rsqr = rsqr_new
        
    print('final rsqr is ',rsqr)
    return x


# Setting up boundary conditions and masks
n=1024
bc=np.zeros([n,n])
mask=np.zeros([n,n],dtype='bool')
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True
bc[0,:]=0.0
bc[0,0]=0.0
bc[0,-1]=0.0

# Set up a square box for calculations

bc[n//4:3*n//4,(1*n//5)] = 1.0
mask[n//4:3*n//4,(1*n//5)] = True

bc[n//4:3*n//4,(4*n//5)] = -1.0
mask[n//4:3*n//4,(4*n//5)] = True

bc[(1*n//5), n//4:3*n//4] = 1.0
mask[(1*n//5), n//4:3*n//4] = True

bc[(4*n//5),n//4:3*n//4] = -1.0
mask[(4*n//5),n//4:3*n//4] = True

# COMEMNTS:
# I used charges of +1 and -1 on opposite ends of each side because having a ..
# .. negative and positively charged side gave a smaller rsqr value (9.576) ..
# .. compared to a rsqr = 66 when I used just +1 charges on all four sides.
# However, I did run the script for +1 charges on all 4 sides as well.
# The figures for all + charges are marked as POST among the figures. 

# --------------------------------- Part (b) --------------------------------- #


# Deriving the green's function. 
kernel = green(2 * n)
kernelft = np.fft.rfft2(kernel)

# The right-hand side is the potential on the mask.  
rhs = bc[mask]
x0 = 0 * rhs

# Getting the charge on the boundary that matches the potential
# We get the charge density on one side of the box.
rho_out = cg(rhs, x0, mask, kernelft, 40, show_steps=True, step_pause=0.25)

plt.plot(rho_out)
plt.xlabel('x - direction')
plt.ylabel('Charge density (rho)')
plt.show()

# Output figure given as "PS8_Q2_fig_b.png".
# FOR ALL +ve charges it is "PS8_Q2_fig_b_POST.png"
# Data outputs given in "PS8_Q2_parts_b_c_Outputs.txt".



# --------------------------------- Part (c) --------------------------------- #


#convert the charge on the boundary to the potential everywhere in space

pot = rho2pot_masked(rho_out, mask, kernelft, True)

# Field inside the box. y = 500 is used to show the field in the x-direction..
#.. directly through the middle of the box. 
plt.plot(pot[200:800,500])
plt.xlabel('x - direction')
plt.ylabel('Potential (V)')
plt.title('Field inside the box')
plt.show()  # Figure "PS8_Q2_fig_c_1.png"
# FOR ALL +ve charges it is "PS8_Q2_fig_c_POST_1.png"

# Field in the x-direction through the middle both inside and ouside the box.
plt.plot(pot[:,500])
plt.xlabel('x - direction')
plt.ylabel('Potential (V)')
plt.title('Field inside/outside the box')
plt.show()  # Figure "PS8_Q2_fig_c_2.png"

# Field inside the box. x = 500 is used to show the field in the y-direction..
#.. directly through the middle of the box. 
plt.plot(pot[500,200:800])
plt.xlabel('y - direction')
plt.ylabel('Potential (V)')
plt.title('Field inside the box')
plt.show()  # Figure "PS8_Q2_fig_c_3.png"
# FOR ALL +ve charges it is "PS8_Q2_fig_c_POST_2.png"

# Field in the y-direction through the middle both inside and ouside the box.
plt.plot(pot[500,:])
plt.xlabel('y - direction')
plt.ylabel('Potential (V)')
plt.title('Field inside/outside the box')
plt.show()  # Figure "PS8_Q2_fig_c_4.png"


# Field inside/outside the box. Here the field is closer to the edge in the x-direction..
plt.plot(pot[:,300])
plt.xlabel('x - direction')
plt.ylabel('Potential (V)')
plt.title('Field inside/outside the box')
plt.show()  # Figure "PS8_Q2_fig_c_5.png"

# Field inside/outside the box. Here the field is closer to the edge in the y-direction..
plt.plot(pot[300,:])
plt.xlabel('y - direction')
plt.ylabel('Potential (V)')
plt.title('Field inside/outside the box')
plt.show()  # Figure "PS8_Q2_fig_c_6.png"

# COMMENTS
#The potential is not exactly constant inside the box. But this was to be ..
#.. expected when using a positive and negative field.

# Potential field just outside the positive part of the box.
# Both the x and y components are shown
plt.plot(pot[1:200, 500], label='x-component')
plt.plot(pot[500, 1:200], label='y-component')
plt.legend()
plt.xlabel('Axis')
plt.ylabel('Potential (V)')
plt.title('Field outside the box')
plt.show()  # Figure "PS8_Q2_fig_c_7_XandY.png"
# FOR ALL +ve charges it is "PS8_Q2_fig_c_POST_XandY.png"

# The potential outside the box for both x and y components are as expected.
# When the charges are all +ve, the potential hovers around V = +1 inside ..
# .. the box. 
