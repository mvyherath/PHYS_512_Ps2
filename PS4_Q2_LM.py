import numpy as np
import camb
from matplotlib import pyplot as plt
import csv

# Comments on the operation of the code in separate text file (PS4_Q2_Comments.txt)
# Outputs in separate file (PS4_Q2_LM_Outputs.txt)

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
        elem1 = get_spectrum(pars)   # deriving f(x + dx)
        pars = pars_copy             # getting the parameters back to their original form

        pars[n] = pars[n] - dx
        elem2 = get_spectrum(pars)    # deriving f(x - dx)
        deriv = (elem1 - elem2) / (2 * dx)    # getting f(x + dx) - f(x - dx) / 2 * dx
        
        derivs[:, n] = deriv    # updating the corresponding columns in the derivs matrix  
        pars = pars_copy        # resetting the parameter array
        
    derivs = derivs[:len(spec)]

    return derivs


def update_lamda(lamda,success):
    
    if success:
        #lamda = lamda + 0.2  # changing the lambda value to experiment with steps
        lamda = lamda/1.5
        if lamda < 0.5:
            lamda = 0
    else:
        if lamda == 0:
            lamda = 1
        else:
            lamda=lamda*1.5**2
            
    return lamda


def fit_lm(pars, y, errors, niter=10):
    
    lamda = 0
    grads = diff_this(pars)
    # original model using original parameters
    model = get_spectrum(pars)
    model = model[:len(spec)]
    r = y - model
    # original chi square
    chisq_ini = np.sum((r / errors)**2)
    
    for i in range(niter):
        grads = diff_this(pars)
        model = get_spectrum(pars)  
        model = model[:len(spec)]
        r = y - model
        lhs = grads.T @ grads
        lhs = lhs + lamda * np.diag(np.diag(lhs))

        rhs = grads.T @ r
        dm = np.linalg.inv(lhs) @ rhs
        
        # Do a trial step to figure out if the current lambda works
        
        m_trial = pars + dm
        grads = diff_this(pars)
        model = get_spectrum(pars)  # updating the model using new params
        model = model[:len(spec)]
        r = y - model        
        chisq_new = np.sum((r / errors)**2)

        # if the new chisq is lower than the previous chisq, the params are upated
        # if Chisq_new > Chisq_ini then params are not updated
        
        if (chisq_new < chisq_ini):
            lamda = update_lamda(lamda, True)
            pars = pars + dm
            print('accepting step with new/old chisq ', chisq_new, chisq_ini)
            chisq_ini = chisq_new
        else:
            lamda = update_lamda(lamda, False)
            print('rejecting step with new/old chisq ',chisq_new, chisq_ini)

        print('on iteration ',i,' chisq is ',chisq_new,' with step ',dm,' and lamda ',lamda)
        print(pars)
        
    return pars



pars = [60.0, 0.02, 0.1, 0.05, 2.00e-9, 1.0]
power_spec = get_spectrum(pars)
data_pts = len(power_spec)
model = get_spectrum(pars)
model=model[:len(spec)]

m_fit = fit_lm(pars, spec, errs)
print(m_fit)
    


