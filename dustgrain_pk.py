######################################################################
## CHANGES:
## Extended h_max from 100 to 1000, to avoid extrapolations
## Changed l range in 10-100000, with 5000 points instead of 10000
## Changed the purpose of the script
######################################################################

import numpy as np
import sys
sys.path.insert(0, '/home/alessandro/code')

from MGCAMB import camb

from scipy import integrate, interpolate, constants as cs

import matplotlib.pyplot as plt

import csv

import os

from time import time

import pyreact


# Define cosmological parameters for DUSTGRAIN-pathfinder simulations
# as they appear in https://doi:10.1093/mnras/sty2465 for LCDM background

Omega_b = 0.0481
h = 0.6731
H0 = h*100 # Hubble constant in km/s/Mpc
# Omega_Lambda = 0.68655 # Taken from CAMB results
A_s = 2.199e-9
n_s = 0.9658
w0 = -1.0
w_a = 0.0

# Define the range for k (just k, not k/h)
k_min = 10**(-4.1) # for lower values CAMB does not compute P(z,k)
k_max = 1000 # value high enough to safely use the interpolator

# Define the redshift range
z_range = np.linspace(0, 4, 100)
# k_points = 300

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Number of values of ell to integrate over
ell_points = 5000
l_min = 1
l_max = 5

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Methods to compute nonlinear P(z,k)
methods = ['RHF', 'RHM']

# Set k/h and P(k) in (Mpc/h)^3 if True, k and P(k) in Mpc^3 if False
hunits = True

# Define output folder
outpath = 'Dustgrain_outs/'
pk_out = outpath+f'Pknl/'
cls_out = outpath+f'Cls/'

print('Creating necessary directories\n')

os.makedirs(outpath, exist_ok=True)
os.makedirs(pk_out, exist_ok=True)
os.makedirs(cls_out, exist_ok=True)

# directory for Vincenzo
os.makedirs(outpath+'Vincenzo/Pknl/', exist_ok=True)
os.makedirs(outpath+'Vincenzo/Cls/', exist_ok=True)

# Angular power spectrum without tomography
def C_l(ell, zs):
    
    c = cs.c/1e3 # in km/s
    
    def E(z): # dimensionless Hubble parameter
        return res.hubble_parameter(z)/H0
    
    def r(z): # comoving radial distance in Mpc
        return res.comoving_radial_distance(z)

    def W(z):
        # Lensing efficiency for sources on the same plane
        return 1.5*Omega_M*(H0/c)**2*(1+z)*r(z)*(1-(r(z)/r(zs)))

    integrand = lambda z: W(z)**2 * P_zk(z, (ell+.5)/r(z)) / r(z)**2 / E(z)
    return (c/H0)*integrate.quad(integrand, 0., zs)[0]


t1 = time()

for cosmo in cosmos:

    print(f'Starting computation for {cosmo}\n')

    # Density parameters and neutrino masses

    # LCDM parameters values
    if cosmo == 'lcdm':

        Omega_M = 0.31345 # Matter density parameter
        Omega_CDM = Omega_M - Omega_b #CDM density parameter

        Omega_nuh2 = 0. # Neutrino density parameter * h^2

        m_nu = 0 # Neutrino mass (eV)

        plot_label = r'$\Lambda$CDM'

    # f(R) models with massless neutrinos
    elif cosmo == 'fr4':

        Omega_M = 0.31345
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.

        m_nu = 0

        plot_label = 'fR4'

    elif cosmo ==  'fr5':

        Omega_M = 0.31345
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.

        m_nu = 0

        plot_label = 'fR5'

    elif cosmo ==  'fr6':

        Omega_M = 0.31345
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.

        m_nu = 0

        plot_label = 'fR6'

    # f(R) models with different neutrino masses
    elif cosmo == 'fr4_0.3':

        Omega_M = 0.30630
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00715

        m_nu = 0.3

        plot_label = 'fR4 0.3 eV'

    elif cosmo == 'fr5_0.1':

        Omega_M = 0.31107
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00238

        m_nu = 0.1

        plot_label = 'fR5 0.1 eV'

    elif cosmo == 'fr5_0.15':

        Omega_M = 0.30987
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00358

        m_nu = 0.15

        plot_label = 'fR5 0.15 eV'

    elif cosmo == 'fr6_0.1':

        Omega_M = 0.31107
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00238

        m_nu = 0.1

        plot_label = 'fR6 0.1 eV'

    elif cosmo == 'fr6_0.06':

        Omega_M = 0.31202
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00143

        m_nu = 0.06

        plot_label = 'fR6 0.06 eV'

    # Getting non-linear P(z,k) for LCDM from CAMB
    if cosmo == 'lcdm':
        pars = camb.CAMBparams(WantTransfer=True, 
                                Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                                NonLinear = 'NonLinear_pk',
                                omnuh2=Omega_nuh2,
                                WantTensors=False, WantVectors=False, WantCls=False, WantDerivedParameters=False,
                                want_zdrag=False, want_zstar=False,
                                MG_flag = 0)
        
        pars.set_cosmology(H0=h*100, ombh2=Omega_b*h**2, omch2=Omega_CDM*h**2, omk=0, mnu=m_nu)
        pars.set_initial_power(camb.initialpower.InitialPowerLaw(As=A_s, ns=n_s))
        pars.set_dark_energy(w=w0,wa=w_a)
        
        pars.set_matter_power(redshifts=z_range[::-1], kmax=k_max, nonlinear=True) # changed from nonlinear=False 
        
        res = camb.get_results(pars)

        res.Params.NonLinearModel.set_params(halofit_version='mead2020')
        res.calc_power_spectra()

        # Computing the non linear Pk in LCDM
        # kh_camb, z_camb, pk_nonlin = res.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=k_points)
        kh_camb, z_camb, pk_nonlin = res.get_nonlinear_matter_power_spectrum(hubble_units=hunits, k_hunit=hunits)

        method = 'HM'

        # sigma_state = f'sigma8 for {cosmo} with {method} is {sigma_8:.3f}'

        # with open(pk_out+'sigma8.txt','a',newline='\n') as sfile:
        #     swriter = csv.writer(sfile)
        #     swriter.writerows(sigma_state)

        # Saving power spectra, k, and z arrays to file
        with open(pk_out+f'{cosmo}_{method}.txt','w',newline='\n') as file:
            writer = csv.writer(file)
            writer.writerows(pk_nonlin)
        
        with open(outpath+'Vincenzo/Pknl/'+f'logPk_{cosmo}_{method}.txt','w',newline='\n') as file:
            writer = csv.writer(file,delimiter=' ')
            writer.writerows(np.log10(pk_nonlin))
        
        np.savetxt(pk_out+f'k_{cosmo}_{method}.txt',kh_camb)
        np.savetxt(outpath+'Vincenzo/Pknl/'+f'logk_{cosmo}_{method}.txt',np.log10(kh_camb))
        np.savetxt(pk_out+f'z_{cosmo}_{method}.txt',z_camb)

        # Setting a fixed grid interpolator to be able to use the Limber approximation
        pk_interp = interpolate.RectBivariateSpline(z_camb, kh_camb, pk_nonlin, kx=5,ky=5)

        # Renaming the interpolator
        P_zk = pk_interp

        for zs in zs_values:
            print(f'Computing C(l) for {cosmo} at zs={zs}\n')
            # Setting the array for l with logarithically equispaced values
            l_array = np.logspace(l_min,l_max,ell_points)
            dl = np.log(l_array[1]/l_array[0]) # Discrete Hankel transform step

            # Compute the C(l)
            cl = np.fromiter((C_l(l,zs) for l in l_array), float)

            np.savetxt(cls_out+f'{cosmo}_{method}_{zs}.txt',cl)

            # for Vincenzo
            np.savetxt(outpath+f'Vincenzo/Cls/{cosmo}_{method}_{zs}.txt',np.column_stack((np.log10(l_array),cl)),delimiter=',',header='log ell, C(ell)')

    # Getting linear P(z,k) for f(R) from CAMB -> to be fed to ReAct
    else:
        pars = camb.CAMBparams(WantTransfer=True, 
                                Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                                NonLinear = 'NonLinear_none',
                                omnuh2=Omega_nuh2,
                                WantTensors=False, WantVectors=False, WantCls=False, WantDerivedParameters=False,
                                want_zdrag=False, want_zstar=False,
                                MG_flag = 0)
        
        pars.set_cosmology(H0=h*100, ombh2=Omega_b*h**2, omch2=Omega_CDM*h**2, omk=0, mnu=m_nu)
        pars.set_initial_power(camb.initialpower.InitialPowerLaw(As=A_s, ns=n_s))
        pars.set_dark_energy(w=w0,wa=w_a)
        
        pars.set_matter_power(redshifts=z_range[::-1], kmax=k_max, nonlinear=False) 
        
        res = camb.get_results(pars)
        sigma_8 = res.get_sigma8_0()
        # To use ReAct we must compute also z and k/h
        # In f(R) we compute the P_NL and then we interpolate to use the Limber approximation
        # kh_camb, z_camb, pk_camb = res.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=k_points)
        kh_camb, z_camb, pk_camb = res.get_linear_matter_power_spectrum(hubble_units=hunits, k_hunit=hunits)
        
        # Model selection and ReAct parameters -> f(R)
        mg_model = 'f(R)'
        Omega_rc = None
        massloop = 30

        react = pyreact.ReACT()

        # Only compute the reaction up to z=2.5 -> ReAct is unstable for z>=2.5
        z_camb = np.array(z_camb)
        z_react = z_camb[z_camb < 2.5]

        # Value of fR0 for different MG models (must be absolute value)
        if 'fr4' in cosmo:
            fR0 = 1e-4
        elif 'fr5' in cosmo:
            fR0 = 1e-5
        elif 'fr6' in cosmo:
            fR0 = 1e-6
        
        # Computing the pseudo power spectrum through nPPF (see ReAct documentation for details)
        model_pseudo = 'fulleftppf' # Full linear EFTofDE model with a nPPF screening nonlinear Poisson modification

        # Setting parameters for pseudo power spectrum and reaction
        # Set linear theory

        #Hu-Sawicki with n=1 (EFT functions)
        alphak0 = 0.
        alphab0 = -fR0 # note the minus sign, as above fR0 is the absolute value
        alpham0 = -fR0 # note the minus sign
        alphat0 = 0.
        m2 = -fR0 # note the minus sign

        # For nPPF
        extrapars = np.zeros(20)
            
        # Set nonlinear theory

        # For nPPF we use the expressions in Eq. 5.6 of arXiv:1608.00522
        alpha = 0.5 
        omegabd = 0

        extrapars[0] = alphak0
        extrapars[1] = alphab0
        extrapars[2] = alpham0
        extrapars[3] = alphat0
        extrapars[4] = m2

        extrapars[5] = 3
        extrapars[6] = 1
        extrapars[7] = (4.-alpha)/(1.-alpha)
        extrapars[8] = Omega_M**(1/3) * ((Omega_M + 4*(1-Omega_M))**(1/(alpha-1)) *extrapars[5]/3/fR0)**(1./extrapars[7])
        extrapars[9] = -1
        extrapars[10] = 2/(3*extrapars[7])
        extrapars[11] = 3/(alpha-4)
        extrapars[12] = -0.8 # Yukawa suppression

        # This parameter scales the background function c(a). 
        # Because we assume LCDM H(a), c(a) will not be identically 0 so we shoulds set it by hand
        c0 = 0. 

        extrapars[19] = c0

        # Compute reaction for f(R) with exact solution

        print(f'Computing reaction for {cosmo}\n')

        R, _, _ = react.compute_reaction(
                                        h, n_s, Omega_M, Omega_b, sigma_8, z_react, kh_camb, pk_camb[0], model=mg_model, 
                                        fR0=fR0, Omega_rc=Omega_rc, w=w0, wa=w_a, 
                                        is_transfer=False, mass_loop=massloop,
                                        verbose=False)
        
        print('\n')
        
        # Compute pseudo power spectrum with nPPF

        print(f'Computing pseudo power spectrum for {cosmo}\n')

        _, _, _, pseudo = react.compute_reaction_ext(
                                        h, n_s, Omega_M, Omega_b, sigma_8, z_react, kh_camb, pk_camb[0], model_pseudo, 
                                        extrapars, 
                                        is_transfer=False, mass_loop=massloop,
                                        verbose=False)
        
        print('\n')

        print(f'Computing non-linear power spectrum for {cosmo}\n')

        # Non-linear power spectrum for f(R) from CAMB, to have P(z,k) for z > 2.5
        pars_CAMB = camb.CAMBparams(WantTransfer=True, 
                            Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                            NonLinear="NonLinear_pk",
                            omnuh2=Omega_nuh2,
                            WantTensors=False, WantVectors=False, WantCls=False, WantDerivedParameters=False,
                            MG_flag=3, QSA_flag=4, F_R0=fR0, FRn=1,
                            want_zdrag=False, want_zstar=False)

        pars_CAMB.set_cosmology(H0=h*100, ombh2=Omega_b*h**2, omch2=Omega_CDM*h**2, omk=0, mnu=m_nu)
        pars_CAMB.set_initial_power(camb.initialpower.InitialPowerLaw(As=A_s, ns=n_s))
        pars_CAMB.set_dark_energy(w=w0,wa=w_a)

        pars_CAMB.set_matter_power(redshifts=z_range[::-1], kmax=k_max, nonlinear=True) # changed from nonlinear=False

        res_CAMB = camb.get_results(pars_CAMB)

        res_CAMB.Params.NonLinearModel.set_params(halofit_version='mead2020')
        res_CAMB.calc_power_spectra()
        sigma8_CAMB = res_CAMB.get_sigma8_0()

        # We compute the non linear power spectrum with MGCAMB in the full z range
        # kh_nlcamb, z_nlcamb, pk_nlcamb = res.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=k_points)
        kh_nlcamb, z_nlcamb, pk_nlcamb = res.get_nonlinear_matter_power_spectrum(hubble_units=hunits, k_hunit=hunits)

        for method in methods:

            # f(R) non linear power spectrum up to z=2.5
            if method == 'RHF':
                pk_partial = R*pseudo
            elif method == 'RHM':
                pk_partial = R*pk_nlcamb[0:62]

            # sigma_state = f'sigma8 for {cosmo} with {method} is {sigma_8:.3f}'

            # with open(pk_out+'sigma8.txt','a',newline='\n') as sfile:
            #     swriter = csv.writer(sfile)
            #     swriter.writerows(sigma_state)
        
            # The full non linear Pk is built from ReAct for z<2.5 and MGCAMB-HMCode for z>=2.5
            pk_nonlin = np.append(pk_partial,pk_nlcamb[62::],axis=0)

            # Saving power spectra, k, and z arrays to file
            with open(pk_out+f'{cosmo}_{method}.txt','w',newline='\n') as file:
                writer = csv.writer(file)
                writer.writerows(pk_nonlin)
            
            with open(outpath+'Vincenzo/Pknl/'+f'logPk_{cosmo}_{method}.txt','w',newline='\n') as file:
                writer = csv.writer(file,delimiter=' ')
                writer.writerows(np.log10(pk_nonlin))
            
            np.savetxt(pk_out+f'k_{cosmo}_{method}.txt',kh_camb)
            np.savetxt(outpath+'Vincenzo/Pknl/'+f'logk_{cosmo}_{method}.txt',np.log10(kh_camb))
            np.savetxt(pk_out+f'z_{cosmo}_{method}.txt',z_camb)

            # Setting a fixed grid interpolator to be able to use the Limber approximation
            pk_interp = interpolate.RectBivariateSpline(z_camb, kh_camb, pk_nonlin, kx=5,ky=5)

            # Renaming the interpolator
            P_zk = pk_interp

            for zs in zs_values:
                print(f'Computing C(l) for {cosmo} with {method} at zs={zs}\n')
                # Setting the array for l with logarithically equispaced values
                l_array = np.logspace(l_min,l_max,ell_points)
                dl = np.log(l_array[1]/l_array[0]) # Discrete Hankel transform step

                # Compute the C(l)
                cl = np.fromiter((C_l(l,zs) for l in l_array), float)

                np.savetxt(cls_out+f'{cosmo}_{method}_{zs}.txt',cl)
                
                # for Vincenzo
                np.savetxt(outpath+'Vincenzo/Cls/'+f'{cosmo}_{method}_{zs}.txt',np.column_stack((np.log10(l_array),cl)),delimiter=',',header='log ell, C(ell)')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')