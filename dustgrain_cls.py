import numpy as np
import sys
sys.path.insert(0, '/home/alessandro/code')

from MGCAMB import camb

from scipy import integrate, interpolate, constants as cs

import matplotlib.pyplot as plt

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

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Number of values of ell to integrate over
ell_points = 5000
l_min = 1
l_max = 4.61

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Define output folder
outpath = 'Dustgrain_outs/'
plot_out = outpath+'Cls_plots/'
print('Creating necessary directories\n')
os.makedirs(outpath, exist_ok=True)
os.makedirs(outpath+'Cls/', exist_ok=True)
os.makedirs(plot_out, exist_ok=True)

# directory for Vincenzo
os.makedirs(outpath+'Vincenzo/Cls/', exist_ok=True)

# Angular power spectrum without tomography
def C_l(ell, zs):
    
    c = cs.c/1e3 # in km/s
    
    def E(z): # dimensionless Hubble parameter
        return res.hubble_parameter(z)/H0
    
    def r(z): # comoving radial distance in Mpc
        return res.comoving_radial_distance(z)

    def W(z):
        # Lensing efficiency for sources on the same plae
        return 1.5*Omega_M*(H0/c)**2*(1+z)*r(z)*(1-(r(z)/r(zs)))

    integrand = lambda z: W(z)**2 * P_zk(z, (ell+.5)/r(z)) / r(z)**2 / E(z)
    return (c/H0)*integrate.quad(integrand, 0., zs)[0]

# Correction to the Limber approximation for low ls
# def limber_correction(l):
#     return (l+2)*(l+1)*l*(l-1)*(2/(2*l+1))**4
# unused if l_min => 10

t1 = time()

kh_camb = np.loadtxt(outpath+f'/Pknl/lcdm_k.txt')
z_camb = np.loadtxt(outpath+f'/Pknl/lcdm_z.txt')

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

        # Value of fR0 for different MG models (must be absolute value)
        if 'fr4' in cosmo:
            fR0 = 1e-4
        elif 'fr5' in cosmo:
            fR0 = 1e-5
        elif 'fr6' in cosmo:
            fR0 = 1e-6
    
    with open(outpath+f'Pknl/'+cosmo.replace(' ','_')+'.txt','r',newline='\n') as file:
        pk_nonlin = np.zeros((100,300))
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            pk_nonlin[i:] = row
        
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
        # cl = np.fromiter((limber_correction(l)*C_l(l,zs) for l in l_array), float)
        # only if l_min <= 10
        cl = np.fromiter((C_l(l,zs) for l in l_array), float)

        np.savetxt(outpath+rf'Cls/{cosmo}_{zs}.txt',cl)

        # Plotting and saving C(l) plots
        plt.figure(figsize=(8,6))
        plt.loglog(l_array,l_array*(l_array+1)*cl/2/np.pi,label= rf'{plot_label}, $z_s={zs}$',color='k')
        plt.xlim(l_array.min(),l_array.max())
        plt.title(r'$\kappa$ - Angular power spectrum',fontsize=16)
        plt.xlabel(r'$\ell$',fontsize=14)
        plt.ylabel(r'$\ell \, (\ell + 1) \, P_{\kappa}(\ell) / (2\pi)$',fontsize=14)
        plt.legend()
        plt.savefig(plot_out+f'Cls_{cosmo}_zs={zs}.png', dpi=300) 
        plt.clf()
        plt.close('all')
        
        # for Vincenzo
        np.savetxt(outpath+f'Vincenzo/Cls/{cosmo}_{zs}.txt',np.column_stack((np.log10(l_array),cl)),delimiter=',',header='log ell, C(ell)')

    # Saving power spectra, k, and z arrays to file
    with open(outpath+f'Pknl/{cosmo}.txt','w',newline='\n') as file:
        writer = csv.writer(file)
        writer.writerows(pk_nonlin*h**3)

    # np.savetxt(outpath+f'{cosmo}_Pk_nonlin.txt',pk_nonlin)
    np.savetxt(outpath+f'/Pknl/{cosmo}_k.txt',kh_camb*h)
    np.savetxt(outpath+f'/Pknl/{cosmo}_z.txt',z_camb)

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')