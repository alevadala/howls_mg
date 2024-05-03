import numpy as np

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

# Define the redshift range
z_range = np.linspace(0, 4, 100)

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Define output folder
outpath = 'Dustgrain_outs/'
files_path = '/home/alessandro/code/camb_fR/Pzk/'
pk_plots = outpath+'Pknl_plots/'
pk_out = outpath+f'Pknl/'

print('Creating necessary directories\n')

os.makedirs(outpath, exist_ok=True)
os.makedirs(pk_out, exist_ok=True)
os.makedirs(pk_plots, exist_ok=True)
# directories for Vincenzo
os.makedirs(outpath+'Vincenzo/Pknl/', exist_ok=True)

t1 = time()

for cosmo in cosmos:

    print(f'Starting computation for {cosmo}\n')

    # Density parameters and neutrino mass

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

    file_root = cosmo.replace(' ','_')

    # Importing P(z,k) at z=0
    fr_pzk0 = np.loadtxt(files_path+f'{file_root}_pzk_z={z_range[0]:.2f}.dat',usecols=1)

    pk_nonlin = np.zeros((len(z_range),len(fr_pzk0)))

    # Filling the P(z,k) array in the CAMB form
    for i in range(len(z_range)):
        pk_nonlin[i,:] = np.loadtxt(files_path+f'{file_root}_pzk_z={z_range[i]:.2f}.dat',usecols=1)

    # Importing the k array
    kh_camb = np.loadtxt(files_path+f'{file_root}_pzk_z={z_range[i]:.2f}.dat',usecols=0)

    # Saving power spectra, k, and z arrays to file
    with open(pk_out+f'{cosmo}_Win.txt','w',newline='\n') as file:
        writer = csv.writer(file)
        writer.writerows(pk_nonlin)

    # np.savetxt(outpath+f'{cosmo}_Pk_nonlin.txt',pk_nonlin)
    np.savetxt(pk_out+f'k_{cosmo}_Win.txt',kh_camb)
    np.savetxt(pk_out+f'z_{cosmo}_Win.txt',z_range)

    with open(outpath+'Vincenzo/Pknl/'+f'logPk_{cosmo}_Win.txt','w',newline='\n') as file:
        writer = csv.writer(file,delimiter=' ')
        writer.writerows(np.log10(pk_nonlin))
    
    np.savetxt(outpath+'Vincenzo/Pknl/'+f'logk_{cosmo}_Win.txt',np.log10(kh_camb))

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')