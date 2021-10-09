import numpy as np
import matplotlib.pyplot as plt

#SEE PS3_Q3_PART_a_MATH.txt for details on the simplification of the dish equation.

# Loading the data into x,y,z arrays
data = np.loadtxt('dish_zenith.txt')
x = data[:,0]
y = data[:,1]
z = data[:,2]

# For the A matrix we have the X1, X2 and X3 values along with X4 = 1
# X1 = x**2 + y**2
# X2 = x
# X3 = y
# X4 = 1

rows = len(x)
cols = 4

# Creation of the A matrix

A = np.zeros([rows,cols])
for i in range(0, len(x), 1):
    # adding valurs to the columns following the simplification procedure
    # looping through each row per column and adding the relevant values
    A[i,0] = (x[i]**2 + y[i]**2)
    A[i,1] = x[i]
    A[i,2] = y[i]
    A[i,3] = 1

# The y matrix consists of all the z values
Y = np.zeros([rows,1])

for j in range(0, len(y), 1):
    Y[j,0] = z[j]

# Now we find the m matrix through the svd equation    
lhs = A.T @ A
rhs = A.T @ Y
m = np.linalg.inv(lhs) @ rhs

print(np.shape(m))
print('Best fit parametrs are', m)

# Now we find the values of the constants using the best fit parameters
a = m[0,0]
x0 = (-1) * m[1,0] / (2 * a)
y0 = (-1) * m[2,0] / (2 * a)
z0 = m[3,0] - (x0**2 + y0**2)
print('The constants are', 'a =',a, 'x0 =',x0, 'y0 =',y0, 'z0 =',z0)

# Here we do the noise calculation
predictions = A @ m
noise = np.std(Y - predictions)
print('The per-point noise is', noise)

# For the noise covariance matrix
N = np.eye(len(x))*noise**2
Ninv=np.eye(len(x))*noise**-2
# To get the errors
mat=A.T @ Ninv @ A
errors = np.linalg.inv(mat)
print('parameter errors are ',np.sqrt(np.diag(errors)))
    
