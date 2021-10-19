import numpy as np
import camb
from matplotlib import pyplot as plt
import PS4_Q2
import csv


steps = []
H0_chain = []
Omb_chain = []
Omc_chain = []
Tau_chain = []
As_chain = []
ns_chain = []

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


def chisq(pars, y, errors):
    
    pred = get_spectrum(pars)
    pred = pred[:len(spec)]
    r = y - pred
    chi_sqr = np.sum((r / errors)**2)

    return chi_sqr


def prior_chisq(pars,par_priors,par_errs):
    if par_priors is None:
        return 0
    par_shifts=pars-par_errs
    return np.sum((par_shifts/par_errs)**2)


# The MCMC function gets upgraded to include two variables in the function for........
# parameter priors and parameter errors.

def mcmc(pars, y, errors, cov, nstep, parm_priors=None, parm_errs=None):

    # getting the current chi square values and setting up matrices for chains
    chi_cur = chisq(pars, y, errors) + prior_chisq(pars, parm_priors, parm_errs)
    npar = len(pars)
    chain = np.zeros([nstep,npar])
    chivec = np.zeros(nstep)
    
    for i in range(nstep):

        # using the Cholesky method to determine the step size.
        # the matrix used here is the curvature matrix form Problem 2
        
        L = np.linalg.cholesky(cov)
        new_cov = np.dot(L, np.random.randn(len(pars)))
        
        trial_pars = pars + (new_cov * 0.5)
        if pars[3] < 0.0:
            pars[3] = np.abs(pars[3])  # replacing Tau with a positive value if it is negative
            
        # using a trial chisq to determine whether to accept this step or not
        trial_chisq = chisq(trial_pars, y, errors) + prior_chisq(pars, parm_priors, parm_errs)
        delta_chisq = trial_chisq - chi_cur
        # acceptance probability based on a Gaussian distribution 
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1) < accept_prob
        
        if accept:
            
            pars=trial_pars      # update params only if this step was accepted
            chi_cur=trial_chisq
            
        chain[i,:]=pars
        chivec[i]=chi_cur
        steps.append(i)
        
    return chain,chivec

planck = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell = planck[:,0]
spec = planck[:,1]
errs = 0.5 * (planck[:,2] + planck[:,3])
nstep = 500

pars1 = [60.0, 0.02, 0.1, 0.05, 2.00e-9, 1.0]
model = get_spectrum(pars1)
model=model[:len(spec)]
par_prev, cov_mat = PS4_Q2.newton_fit(pars1, spec, errs, model)

# OR....does the cholesky function go here???, outside the loop.
# Setting the step size here instead of inside the function, putting a variable for the matrix in the function.

#L = np.linalg.cholesky(cov)
#new_cov = np.dot(L, np.random.randn(len(pars)))

pars2 = [68.0, 0.022, 0.12, 0.06, 2.10e-9, 0.96]
chains, chi_sq = mcmc(pars2, spec, errs, cov_mat, nstep)
print(np.shape(chi_sq), np.shape(steps))
print(np.shape(chains))


expected_pars = 0 * pars2
expected_pars[3] = 0.054
par_errs = 0 * starting_pars + 1e20
par_errs[1]=0.5

new_chains, new_chi_sq = mcmc(pars2, spec, errs, cov_mat, nstep, parm_priors=expected_pars, parm_errs=par_errs)

# Do the importance sampling
nsamp = chains.shape[0]
weight = np.zeros(nsamp)
chivec = np.zeros(nsamp)
for i in range(nsamp):
    chisq = prior_chisq(chain[i,:],expected_pars,par_errs)
    chivec[i]=chisq
#    weight[i]=np.exp(-0.5*chisq)
chivec=chivec-chivec.mean()
weight=np.exp(0.5*chivec)



