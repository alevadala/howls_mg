import numpy as np

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

# Define the redshift range
z_range = np.linspace(0, 4, 100)

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Number of values of ell to integrate over
ell_points = 5000
l_min = 1
l_max = 5

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Define output folder
outpath = 'Dustgrain_outs/fofr_fit/'
plot_out = outpath+'Cls_plots/'
files_path = '/home/alessandro/code/camb_fR/Pzk/'
print('Creating necessary directories\n')
os.makedirs(outpath, exist_ok=True)
os.makedirs(outpath+'Pknl/', exist_ok=True)
os.makedirs(outpath+'Cls/', exist_ok=True)
os.makedirs(plot_out, exist_ok=True)


# directory for Vincenzo
os.makedirs(outpath+'Vincenzo/Cls/', exist_ok=True)

def E(z):
    return np.sqrt(Omega_M*(1+z)**3+Ode)

def r(z):
    c = cs.c/1e3
    integrand = lambda z_prime: 1 / E(z_prime)
    result = integrate.quad(integrand, 0., z)[0]
    return (c/H0)*result

# Angular power spectrum without tomography
def C_l(ell, zs):
    
    c = cs.c/1e3 # in km/s

    def W(z):
        # Lensing efficiency for sources on the same plae
        return 1.5*Omega_M*(H0/c)**2*(1+z)*r(z)*(1-(r(z)/r(zs)))

    integrand = lambda z: W(z)**2 * P_zk(z, (ell+.5)/r(z)) / r(z)**2 / E(z)
    return (c/H0)*integrate.quad(integrand, 0., zs)[0]


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
        pk = np.loadtxt(files_path+f'{file_root}_pzk_z={z_range[i]:.2f}.dat',usecols=1)
        pk_nonlin[i,:] = pk*h**3

    # Importing the k array
    kh = np.loadtxt(files_path+f'{file_root}_pzk_z={z_range[i]:.2f}.dat',usecols=0)
    kh_camb = kh*h

    # Omega Lambda for the cosmology
    Ode = 1-Omega_M-(Omega_nuh2/h**2)

    # Setting a fixed grid interpolator to be able to use the Limber approximation
    pk_interp = interpolate.RectBivariateSpline(z_range, kh_camb, pk_nonlin, kx=5,ky=5)

    # Renaming the interpolator
    P_zk = pk_interp

    for zs in zs_values:
        print(f'Computing C(l) for {cosmo} at zs={zs}\n')
        # Setting the array for l with logarithically equispaced values
        l_array = np.logspace(l_min,l_max,ell_points)
        dl = np.log(l_array[1]/l_array[0]) # Discrete Hankel transform step

        # Compute the C(l)
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
        writer.writerows(pk_nonlin)

    # np.savetxt(outpath+f'{cosmo}_Pk_nonlin.txt',pk_nonlin)
    np.savetxt(outpath+f'/Pknl/{cosmo}_k.txt',kh_camb)
    np.savetxt(outpath+f'/Pknl/{cosmo}_z.txt',z_range)

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')