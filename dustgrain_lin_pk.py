import numpy as np
import sys
sys.path.insert(0, '/home/alessandro/code')

from MGCAMB import camb

import csv

import os

from time import time

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
k_min = 1e-4
k_max = 1e3

# Define the redshift range
z_range = np.linspace(0, 4, 100)
k_points = 300

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Set k/h and P(k) in (Mpc/h)^3 if True, k and P(k) in Mpc^3 if False
# hunits = True

# Define output folder
outpath = 'Dustgrain_outs/'
pk_out = outpath+f'Pk_lin/'

print('Creating necessary directories\n')

os.makedirs(outpath, exist_ok=True)
os.makedirs(pk_out, exist_ok=True)

# directory for Vincenzo
os.makedirs(outpath+'Vincenzo/Pk_lin/', exist_ok=True)

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

        plot_label = 'fR4 0.3eV'

    elif cosmo == 'fr5_0.1':

        Omega_M = 0.31107
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00238

        m_nu = 0.1

        plot_label = 'fR5 0.1eV'

    elif cosmo == 'fr5_0.15':

        Omega_M = 0.30987
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00358

        m_nu = 0.15

        plot_label = 'fR5 0.15eV'

    elif cosmo == 'fr6_0.1':

        Omega_M = 0.31107
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00238

        m_nu = 0.1

        plot_label = 'fR6 0.1eV'

    elif cosmo == 'fr6_0.06':

        Omega_M = 0.31202
        Omega_CDM = Omega_M - Omega_b

        Omega_nuh2 = 0.00143

        m_nu = 0.06

        plot_label = 'fR6 0.06eV'

    # Getting linear P(z,k) for LCDM from CAMB
    if cosmo == 'lcdm':
        pars = camb.CAMBparams(WantTransfer=True, 
                                Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                                NonLinear = 'NonLinear_none',
                                omnuh2=Omega_nuh2,
                                WantTensors=False, WantVectors=False, WantCls=False, WantDerivedParameters=False,
                                want_zdrag=False, want_zstar=False,
                                MG_flag = 0)

    # Getting linear P(z,k) for f(R) from CAMB
    else:

        # Value of fR0 for different MG models (absolute value)
        if 'fr4' in cosmo:
            fR0 = 1e-4
        elif 'fr5' in cosmo:
            fR0 = 1e-5
        elif 'fr6' in cosmo:
            fR0 = 1e-6

        pars = camb.CAMBparams(WantTransfer=True, 
                            Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                            NonLinear="NonLinear_none",
                            omnuh2=Omega_nuh2,
                            WantTensors=False, WantVectors=False, WantCls=False, WantDerivedParameters=False,
                            MG_flag=3, QSA_flag=4, F_R0=fR0, FRn=1,
                            want_zdrag=False, want_zstar=False)

    pars.set_cosmology(H0=h*100, ombh2=Omega_b*h**2, omch2=Omega_CDM*h**2, omk=0, mnu=m_nu)
    pars.set_initial_power(camb.initialpower.InitialPowerLaw(As=A_s, ns=n_s))
    pars.set_dark_energy(w=w0,wa=w_a)

    pars.set_matter_power(redshifts=z_range[::-1], kmax=k_max, nonlinear=False)

    res = camb.get_results(pars)

    # res.Params.NonLinearModel.set_params(halofit_version='mead2020')
    res.calc_power_spectra()
    sigma8 = res.get_sigma8_0()

    # We compute the linear power spectrum with MGCAMB in the full z range
    kh_camb, z_camb, pk_lin = res.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=k_points)
    # kh_nlcamb, z_nlcamb, pk_nlcamb = res.get_nonlinear_matter_power_spectrum(hubble_units=hunits, k_hunit=hunits)

    # Saving power spectra, k, and z arrays to file
    with open(pk_out+f'{cosmo}.txt','w',newline='\n') as file:
        writer = csv.writer(file)
        writer.writerows(pk_lin)
    
    with open(outpath+'Vincenzo/Pk_lin/'+f'logPk_{cosmo}.txt','w',newline='\n') as file:
        writer = csv.writer(file,delimiter=' ')
        writer.writerows(np.log10(pk_lin))
    
    np.savetxt(pk_out+f'k_{cosmo}.txt',kh_camb)
    np.savetxt(outpath+'Vincenzo/Pk_lin/'+f'logk_{cosmo}.txt',np.log10(kh_camb))
    np.savetxt(pk_out+f'z_{cosmo}.txt',z_camb)

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')