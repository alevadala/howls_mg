import numpy as np

from scipy import integrate, interpolate, constants as cs

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

# Speed of light in vacuum in km/s
c = cs.c/1e3

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Define input/output folder
outpath = 'Dustgrain_outs/'
cls_out = outpath+f'Cls/'
pk_path = outpath+'Pknl/'

print('Creating necessary directories\n')

os.makedirs(outpath, exist_ok=True)
os.makedirs(cls_out, exist_ok=True)

# directory for Vincenzo
os.makedirs(outpath+'Vincenzo/Cls/', exist_ok=True)

# Dimensionless Hubble parameter
def E(z):
    return np.sqrt(Omega_M*(1+z)**3+Omega_DE)

# Comoving radial distance
def r(z):
    integrand = lambda z_prime: 1 / E(z_prime)
    result = integrate.quad(integrand, 0., z)[0]
    return (c/H0)*result

def W(z):
    # Lensing efficiency for sources on the same plane
    return 1.5*Omega_M*(H0/c)**2*(1+z)*r(z)*(1-(r(z)/r(zs)))

# Angular power spectrum
def C_l(ell, zs):

    integrand = lambda z: W(z)**2 * P_zk(z, (ell+.5)/r(z)) / r(z)**2 / E(z)
    return (c/H0)*integrate.quad(integrand, 0., zs)[0]

t1 = time()

for cosmo in cosmos:

    if cosmo == 'lcdm':
        methods = ['HM', 'EE2', 'Win']
    else:
        methods = ['RHM','RHF', 'Win']

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

    # Omega Lambda for the selected cosmology
    Omega_DE = 1-Omega_M-(Omega_nuh2/h**2)

    for method in methods:

        print(f'Importing power spectrum for {cosmo} with {method}\n')

        k = np.loadtxt(pk_path+f'k_{cosmo}_{method}.txt')
        z = np.loadtxt(pk_path+f'z_{cosmo}_{method}.txt')
        pk_in = f'{cosmo}_{method}.txt'
        plt_label = f'{method}'
        
        with open(pk_path+pk_in,'r',newline='\n') as pk_impo:
            pk_nonlin = np.zeros((len(z),len(k)))
            reader = csv.reader(pk_impo)
            for i,row in enumerate(reader):
                pk_nonlin[i,:] = row

        # Changing units from CAMB defaults
        pk_nonlin = pk_nonlin/h**3
        k = k*h

        # Setting a fixed grid interpolator to be able to use the Limber approximation
        pk_interp = interpolate.RectBivariateSpline(z, k, pk_nonlin, kx=5,ky=5)

        # Renaming the interpolator
        P_zk = pk_interp

        for zs in zs_values:

            print(f'Computing C(l) for {cosmo} with {method} at zs={zs}\n')
            # Setting the array for l with logarithically equispaced values
            l_array = np.logspace(l_min,l_max,ell_points)

            # Compute the C(l)
            cl = np.fromiter((C_l(l,zs) for l in l_array), float)

            # Saving on file
            np.savetxt(cls_out+f'{cosmo}_{method}_{zs}.txt',cl)

            # for Vincenzo
            np.savetxt(outpath+f'Vincenzo/Cls/{cosmo}_{method}_{zs}.txt',np.column_stack((np.log10(l_array),cl)),delimiter=',',header='log ell, C(ell)')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')