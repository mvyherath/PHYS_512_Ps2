import numpy as np
import camb
from matplotlib import pyplot as plt
import PS4_Q2
import csv

# Outputs from the MCMC can be found in PS4_Q3_Outputs.txt 
# Comments on the output of this code can be found in PS4_Q3_Comments.txt

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


def mcmc(pars, y, errors, cov, nstep):

    # getting the current chi square values and setting up matrices for chains
    chi_cur = chisq(pars, y, errors)
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
        trial_chisq = chisq(trial_pars, y, errors)
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

pars2 = [68.0, 0.022, 0.12, 0.06, 2.10e-9, 0.96]
chains, chi_sq = mcmc(pars2, spec, errs, cov_mat, nstep)
print(np.shape(chi_sq), np.shape(steps))
print(np.shape(chains))

H0 = np.mean(chains[:,0])
Omega_b = np.mean(chains[:,1])
Omega_c = np.mean(chains[:,2])
Tau = np.mean(chains[:,3])
A_s = np.mean(chains[:,4])
n_s = np.mean(chains[:,5])



print('H0 is', H0, 'and its uncertainty is',np.std(chains[:,0])/(np.sqrt(nstep)))
print('Omega_b is', Omega_b, 'and its uncertainty is',np.std(chains[:,1])/(np.sqrt(nstep)))
print('Omega_c is', Omega_c, 'and its uncertainty is',np.std(chains[:,2])/(np.sqrt(nstep)))
print('Tau is', Tau, 'and its uncertainty is',np.std(chains[:,3])/(np.sqrt(nstep)))
print('A_s is', A_s, 'and its uncertainty is',np.std(chains[:,4])/(np.sqrt(nstep)))
print('n_s is', n_s, 'and its uncertainty is',np.std(chains[:,5])/(np.sqrt(nstep)))

chainft1 = np.fft.rfft(chains[:,0])
chainft2 = np.fft.rfft(chains[:,1])
chainft3 = np.fft.rfft(chains[:,2])
chainft4 = np.fft.rfft(chains[:,3])
chainft5 = np.fft.rfft(chains[:,4])
chainft6 = np.fft.rfft(chains[:,5])

# The plots are all available in the GitHub page

plt.loglog(np.abs(chainft1), color='blue')
plt.xlabel('Steps')
plt.ylabel('H0')
plt.show()


plt.loglog(np.abs(chainft2), color='green')
plt.xlabel('Steps')
plt.ylabel('Omega_b')
plt.show()
plt.savefig('PS4_Q3_OmB_FFT.png')


plt.loglog(np.abs(chainft3), color='red')
plt.xlabel('Steps')
plt.ylabel('Omega_c')
plt.show()
plt.savefig('PS4_Q3_OmC_FFT.png')

plt.loglog(np.abs(chainft4), color='orange')
plt.xlabel('Steps')
plt.ylabel('Tau')
plt.show()
plt.savefig('PS4_Q3_Tau_FFT.png')

plt.loglog(np.abs(chainft5), color='purple')
plt.xlabel('Steps')
plt.ylabel('A_s')
plt.show()
plt.savefig('PS4_Q3_As_FFT.png')

plt.loglog(np.abs(chainft6), color='black')
plt.xlabel('Steps')
plt.ylabel('n_s')
plt.show()
plt.savefig('PS4_Q3_ns_FFT.png')




plt.plot(steps, chi_sq, color='blue')
plt.xlabel('Steps')
plt.ylabel('Chi Square')
plt.show()
plt.savefig('PS4_Q3_ChiSquare.png')

plt.plot(steps, chains[:,0], color='blue')
plt.xlabel('Steps')
plt.ylabel('H0')
plt.show()
plt.savefig('PS4_Q3_H0_Chain.png')

plt.plot(steps, chains[:,1], color='green')
plt.xlabel('Steps')
plt.ylabel('Omega_b')
plt.show()
plt.savefig('PS4_Q3_OmB_Chain.png')

plt.plot(steps, chains[:,2], color='red')
plt.xlabel('Steps')
plt.ylabel('Omega_c')
plt.show()
plt.savefig('PS4_Q3_OmC_Chain.png')

plt.plot(steps, chains[:,3], color='orange')
plt.xlabel('Steps')
plt.ylabel('Tau')
plt.show()
plt.savefig('PS4_Q3_Tau_Chain.png')

plt.plot(steps, chains[:,4], color='purple')
plt.xlabel('Steps')
plt.ylabel('A_s')
plt.show()
plt.savefig('PS4_Q3_As_Chain.png')

plt.plot(steps, chains[:,5], color='black')
plt.xlabel('Steps')
plt.ylabel('n_s')
plt.show()
plt.savefig('PS4_Q3_ns_Chain.png')

with open('planck_chains.txt', 'w+') as x:
    writer = csv.writer(x, delimiter='\t')
    writer.writerows(zip(chains[:,0],chains[:,1],chains[:,2],chains[:,3],chains[:,4],chains[:,5]))
   
        
Om_b = Omega_b / (H0 / 100)**2
Om_c = Omega_c / (H0 / 100)**2
dark_energy = 1 - Om_b - Om_c
print('Dark Energy =', dark_energy)
