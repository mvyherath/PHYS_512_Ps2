import numpy as np
import camb
from matplotlib import pyplot as plt
import csv

# Comments on the operation of the code in separate test file (PS4_Q2_Comments.txt)
# Outputs in separate file (PS4_Q2_Newton_Outputs.txt)


def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

# Loading the parameters from the Plank satellite
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

curv_matr = []
text_file = []

# Creating the function to get differentiated values for the 6 parameters

def diff_this(pars):
    derivs = np.empty([data_pts,len(pars)])
    
    for n in range(0, len(pars),1):
        # making a copy of the original list of parameters
        pars_copy = np.copy(pars)
        # setting the value of dx which can be subjected to change
        dx = 0.01 * pars[n]
        
        pars[n] = pars[n] + dx      
        elem1 = get_spectrum(pars)  # deriving f(x + dx)
        pars = pars_copy            # getting the parameters back to their original form
        pars[n] = pars[n] - dx      
        elem2 = get_spectrum(pars)  # deriving f(x - dx)
        deriv = (elem1 - elem2) / (2 * dx)  # getting f(x + dx) - f(x - dx) / 2 * dx
        
        derivs[:, n] = deriv        # updating the corresponding columns in the derivs matrix
        pars = pars_copy            # resetting the parameter array
        
    derivs = derivs[:len(spec)]

    return derivs



def newton_fit(pars, y, errors, model, niter=2):


    for i in range(niter):
        
        grads = diff_this(pars)
        r = y - model
        chisq_ini = np.sum((r / errors)**2)  # getting chi square values, taking the errors into consideration
        lhs = grads.T @ grads
        rhs = grads.T @ r
        dm = np.linalg.inv(lhs) @ rhs
        
        pars = pars + dm  # updating the parameters
        
        model = get_spectrum(pars)  # rederiving the model using updated parameters
        model=model[:len(spec)]
        r = y-model
        chisq_new = np.sum((r / errors)**2) # updating chi square with updated parameters
        
        print('Iteration =',i, 'ChiSq_NEW =',chisq_new, 'ChiSq_OLD =',chisq_ini, 'step of',dm)
        print(pars)

        best_fit_params = np.empty([niter, len(pars)])
        param_errs = np.empty([niter, len(pars)])
        best_fit_params[i, :] = pars              # storing the best fit parameters

        cov_mat = np.linalg.inv(lhs)              # deriving the covariance matrix
        curv_matr.append(cov_mat)                 # deriving the curvature matrix
        diag = np.diagonal(cov_mat)               # storing the errors
        error_vals = np.sqrt(diag)
        param_errs[i, :] = error_vals

        with open('planck_fit_params.txt', 'w+') as x:  # writing the parameters to a text file
            writer = csv.writer(x, delimiter='\t')
            writer.writerows(zip(pars, error_vals))


    return pars, cov_mat


pars = [60.0, 0.02, 0.1, 0.05, 2.00e-9, 1.0]
power_spec = get_spectrum(pars)
data_pts = len(power_spec)
model = get_spectrum(pars)
model=model[:len(spec)]
r = spec-model

m_fit, cov = newton_fit(pars, spec, errs, model)
print(m_fit)

new_model = get_spectrum(m_fit)
new_model = new_model[:len(spec)]

planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
plt.plot(ell,new_model)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.xlabel('Spectral values')
plt.ylabel('Spectral powers')
#plt.show()

    


