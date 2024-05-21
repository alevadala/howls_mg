import numpy as np

from scipy import fft, integrate
from scipy.special import spherical_jn as jn

import os

from time import time

# Output path
outpath = 'Dustgrain_outs/'
pk_path = outpath+'Pknl/'
cls_path = outpath+'Cls/'
outpath_bins = outpath+'25bins/'
two_pt_out = outpath_bins+'2PCF/'
vincenzo = outpath_bins+'Vincenzo/2PCF/'

print('Creating directories\n')
os.makedirs(two_pt_out, exist_ok=True)
os.makedirs(vincenzo, exist_ok=True)

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

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

# Setting the array for l with logarithically equispaced values
ell_points = 5000
l_min = 1
l_max = 5
l_array = np.logspace(l_min,l_max,ell_points)
dl = np.log(l_array[1]/l_array[0]) # Discrete Hankel transform step

# Settings for the FHT
init = -1.4 # Initial step for the offset
bs = -0.3 # Bias
mu_j = 0 # Order of the Bessel function
conv_factor = 206265/60 # Conversion factor from radians to arcmins
# correction = np.sqrt(np.pi/2)*conv_factor # Total correction to account for conversion and spherical Bessel

offset = fft.fhtoffset(dl, initial=init, mu=mu_j, bias=bs) # Setting offset for low ringing condition
theta = np.exp(offset)*conv_factor/l_array[::-1] # Output theta array

# Settings for the Bessel function integral
off_bess = fft.fhtoffset(dl,initial=0,mu=0,bias=0)
th = np.exp(off_bess)/l_array[::-1]

# Shift to plot the 2PCF in the correct interval
shift = 0.01/th[0]
th_tr = th*shift

# Switch the Limber correction on/off
correct_limber = 1

# Correction to the Limber approximation for low ls
def limber_correction(l):
    return (l+2)*(l+1)*l*(l-1)*(2/(2*l+1))**4

# Defining the 2PCF through integral over J_0
def xi(theta_range, lrange):
    j_integ = np.fromiter((jn(0,theta_range*l)*l for l in lrange), float)
    return integrate.simpson(y=cl*j_integ, x=lrange)/np.pi

t1 = time()

for cosmo in cosmos:

    if cosmo == 'lcdm':
        methods = ['HM', 'EE2', 'Win']
    else:
        methods = ['RHM','RHF', 'Win']

    print(f'Starting computation for {cosmo}\n')

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

    for zs in zs_values:

        print(f'Importing C(l) for {cosmo} at zs={zs}\n')
        
        # Computing the 2PCF through Hankel transform (with FHT/Bessel integral)
        # The expression of the discrete Hankel transform (FHT) has been modified to match
        # the canonical definition of the 2PCF        

        for method in methods:
            print(f'Computing 2PCF for {cosmo} with {method} at zs={zs}\n')

            if correct_limber:
                limber_term = np.fromiter((limber_correction(l) for l in l_array),float)
                cl = np.loadtxt(cls_path+f'{cosmo}_{method}_{zs}.txt')
                cl = limber_term*cl
            else:
                cl = np.loadtxt(cls_path+f'{cosmo}_{method}_{zs}.txt')
            # FHT. Changed from correction to conversion factor only
            xi_fft = fft.fht(cl*l_array*conv_factor,dln=dl,mu=mu_j,offset=offset,bias=bs)/theta/2/np.pi
            # Bessel integral
            xi_theta = np.fromiter((xi(theta_val,l_array) for theta_val in th), float)

            # Saving the 2PCF values
            print('Saving on file\n')

            np.savetxt(two_pt_out+f'{cosmo}_{method}_{zs}_FHT_theta.txt',theta)
            np.savetxt(two_pt_out+f'{cosmo}_{method}_{zs}_Bes_theta.txt',th_tr)
            np.savetxt(two_pt_out+f'{cosmo}_{method}_{zs}_FHT_2pcf.txt',xi_fft)
            np.savetxt(two_pt_out+f'{cosmo}_{method}_{zs}_Bes_2pcf.txt',xi_theta)

            # File for Vincenzo
            np.savetxt(vincenzo+f'{cosmo}_{method}_{zs}_FHT.txt',np.column_stack((np.log10(theta),xi_fft)),delimiter=' ',header='log theta, xi(theta)')
            np.savetxt(vincenzo+f'{cosmo}_{method}_{zs}_Bes.txt',np.column_stack((np.log10(th_tr),xi_theta)),delimiter=' ',header='log theta, xi(theta)')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')