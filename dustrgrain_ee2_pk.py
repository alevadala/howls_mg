import numpy as np
import sys

import euclidemu2 as ee2

import classy

import csv

from scipy import integrate, interpolate, fft, constants as cs

import os
import matplotlib.pyplot as plt

from time import time

# Define output folder
outpath = 'Dustgrain_outs/'
pk_plots = outpath+'Pknl_plots/'
cls_plots = outpath+'Cls_plots/'
pk_out = outpath+f'Pknl/'
cls_out = outpath+f'Cls/'

print('Creating necessary directories\n')

os.makedirs(outpath, exist_ok=True)
os.makedirs(pk_out, exist_ok=True)
os.makedirs(cls_out, exist_ok=True)
os.makedirs(pk_plots, exist_ok=True)
os.makedirs(cls_plots, exist_ok=True)
# directory for Vincenzo
os.makedirs(outpath+'Vincenzo/Pknl/', exist_ok=True)
os.makedirs(outpath+'Vincenzo/Cls/', exist_ok=True)

# Cosmological parameters according to DUSTGRAIN-pathfinder simulations for LCDM
cosmo_par = {
            'As':2.199e-9,
            'ns':0.9658,
            'Omb':0.0481,
            'Omm':0.31345,
            'h':0.6731,
            'mnu':0.0,
            'w':-1.0,
            'wa':0.0 
            }

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

Omega_M = 0.31345 # Matter density parameter
Omega_CDM = Omega_M - Omega_b #CDM density parameter

Omega_nuh2 = 0. # Neutrino density parameter * h^2

Ode = 1-Omega_M-(Omega_nuh2/h**2)

m_nu = 0 # Neutrino mass (eV)

plot_label = r'$\Lambda$CDM'

cosmo = 'lcdm'
method = 'EE2'

# Define the range for k (just k, not k/h)
k_min = 10**(-4.1) # for lower values CAMB does not compute P(z, k)
k_max = 1000 #10**(1.5) # value high enough to safely use the interpolator

# Define the redshift range
z_range = np.linspace(0, 4, 50)

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

k_range = np.geomspace(k_min,k_max,300)

# Number of values of ell to integrate over
ell_points = 5000
l_min = 1
l_max = 5

def E(z):
    return np.sqrt(Omega_M*(1+z)**3+Ode)

def r(z):
    c = cs.c/1e3
    integrand = lambda z_prime: 1 / E(z_prime)
    result = integrate.quad(integrand, 0., z)[0]
    return (c/H0)*result

# Angular power spectrum without tomography
def C_l(ell, zs):
    
    c = cs.c/1e3

    def W(z):
        
        return 1.5*Omega_M*(H0/c)**2*(1+z)*r(z)*(1-(r(z)/r(zs)))

    integrand = lambda z: W(z)**2 * P_zk(z, (ell+.5)/r(z)) / r(z)**2 / E(z)
    return (c/H0)*integrate.quad(integrand, 0., zs)[0]

print('Computing P(k) with EuclidEmulator2\n')

k_emu, pnl, plin, b = ee2.get_pnonlin(cosmo_par, z_range, k_range)

print('\n')

pk_nonlin = np.zeros((len(z_range),len(k_emu)))

for z in range(len(pnl)):
    pk_nonlin[z,:] = pnl[z]

# Saving power spectra, k, and z arrays to file
with open(pk_out+f'{cosmo}_{method}.txt','w',newline='\n') as file:
    writer = csv.writer(file)
    writer.writerows(pk_nonlin)

# np.savetxt(outpath+f'{cosmo}_Pk_nonlin.txt',pk_nonlin)
np.savetxt(pk_out+f'k_{cosmo}_{method}.txt',k_emu)
np.savetxt(pk_out+f'z_{cosmo}_{method}.txt',z_range)

with open(outpath+'Vincenzo/Pknl/'+f'logPk_{cosmo}_{method}.txt','w',newline='\n') as file:
    writer = csv.writer(file,delimiter=' ')
    writer.writerows(np.log10(pk_nonlin))

np.savetxt(outpath+'Vincenzo/Pknl/'+f'logk_{cosmo}_{method}.txt',np.log10(k_emu))

pk_interp = interpolate.RectBivariateSpline(z_range, k_emu, pk_nonlin, kx=5,ky=5)

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