import numpy as np

from scipy import integrate, interpolate, constants as cs
# from scipy.special import j1
from scipy.optimize import curve_fit

import os

import csv

from time import time

# Output path
outpath = 'Dustgrain_outs/'
pk_path = outpath+'Pknl/'
# dv_path = '/home/alessandro/phd/kappa_moments/'
dv_path = '/home/alessandro/phd/MG_Paper_outputs_FINAL_mean_DVs/outputs_FINAL_mean_DVs/mean_DVs/kappa_moments/'

homs_out = outpath+'HOMs/'

print('Creating directories\n')
os.makedirs(homs_out, exist_ok=True)

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm']#, 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Smoothing scale range (in arcmins)
theta_s = np.linspace(.5,22,50)

# Speed of light in vacuum (in km/s)
c = cs.c/1e3

# Conversion factor rads-arcmins
conv_factor = 60/206265

# Fiducial values for Q parameters (in LCDM)
fid_Q2 = 1
fid_Q3 = 3
fid_Q4 = 12*7.29+4*16.23
# To list
fid_Q = [fid_Q2, fid_Q3, fid_Q4]

# Headers of the HOMs file
homs_cols = ['smoothing','k2','sigk2','k3','sigk3','k4','sigk4']#,'S3','sigS3','S4','sigS4']

# Define cosmological parameters for DUSTGRAIN-pathfinder simulations
# as they appear in https://doi:10.1093/mnras/sty2465 for LCDM background
Omega_b = 0.0481
h = 0.6731
H0 = h*100 # Hubble constant in km/s/Mpc
# Omega_Lambda = 0.68655 # Computed for different cosmologies
A_s = 2.199e-9
n_s = 0.9658
w0 = -1.0
w_a = 0.0

# Dimensionless Hubble paremeter
def E(z):
    return np.sqrt(Omega_M*(1+z)**3+Ode)

# Comoving radial dinstance
def r(z):
    integrand = lambda z_prime: 1 / E(z_prime)
    result = integrate.quad(integrand, 0., z, limit = 300)[0]
    return (c/H0)*result

# Fourier transform of top-hat filter
def W_th(ell,theta_sm):
    theta_sm = theta_sm*conv_factor
    win =  3*(np.sin(ell*theta_sm) - ell*theta_sm*np.cos(ell*theta_sm))/(ell*theta_sm)**3
    # win = 2*j1(ell*theta_sm)/(ell*theta_sm) # In terms of Bessel function
    return win/2/np.pi # NOTE: added 1/2pi term in defining Fourier transform

# Correction to Limber approximation for low ls
def limber_correction(l):
    return (l+2)*(l+1)*l*(l-1)*(2/(2*l+1))**4

# Smoothed convergence field
def k_sm(z,theta_sm):
    integrand = lambda ell: limber_correction(ell) * P_zk(z, (ell+.5)/r(z)) * W_th(ell,theta_sm)**2 * ell
    return 2*np.pi*integrate.quad(integrand, 10, 1e5, limit = 300)[0]

# Lensing efficiency
def W(z):
    return 1.5*Omega_M*(H0/c)**2*(1+z)*r(z)*(1-(r(z)/r(zs)))

# Functional for the three different moments
def C_t(theta_sm,t):
    integrand = lambda z: W(z)**t * k_sm(z,theta_sm)**(t-1) / E(z) / r(z)**(2*(t-1))
    return (c/H0)*integrate.quad(integrand, 0., zs, limit = 300)[0]

# Function to fit the value of the Q parameters
def k_fit(theta_sm, Q):
    return Q*np.fromiter((C_t(theta,t) for theta in theta_sm),float)

t1 = time()

for cosmo in cosmos:

    # Methods for the P(z,k)
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

    if cosmo == 'lcdm':
        name_cosmo = 'LCDM'
    else:
        name_cosmo = plot_label.replace(' ','_')
    
    # Dark energy density
    Ode = 1-Omega_M-(Omega_nuh2/h**2)

    for method in methods:

        print(f'Importing P(z,k) for {name_cosmo} with {method}\n')

        k = np.loadtxt(pk_path+f'k_{cosmo}_{method}.txt')
        z = np.loadtxt(pk_path+f'z_{cosmo}_{method}.txt')
        pk_in = f'{cosmo}_{method}.txt'
        plt_label = f'{method}'

        with open(pk_path+pk_in,'r',newline='\n') as pk_impo:
            pk_nonlin = np.zeros((len(z),len(k)))
            reader = csv.reader(pk_impo)
            for i,row in enumerate(reader):
                pk_nonlin[i,:] = row

        # Changing in h units
        pk_nonlin = pk_nonlin/h**3
        k = k*h

        # Building interpolator for Limber approximation
        P_zk = interpolate.RectBivariateSpline(z, k, pk_nonlin, kx=5, ky=5)

        for zs in zs_values:

            print(f'Importing measured moments for {name_cosmo} at zs={zs}\n')

            hom_file = dv_path+f'LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_filter_tophat_scales_[ 4  8 16 32]_pixels_kappa_moments.txt'
            hom_meas = np.loadtxt(hom_file)

            arcmins = np.fromiter((hom_meas[i][0] for i in range(len(hom_meas))),float)

            for t in [2,3,4]:

                print(f'Computing C_{t} for {name_cosmo} with {method}\n')

                Ct = np.fromiter((C_t(theta,t) for theta in theta_s),float)

                k_index = homs_cols.index('k'+f'{t}')
                err_index = homs_cols.index('sigk'+f'{t}')
                k_meas = np.fromiter((hom_meas[i][k_index] for i in range(len(hom_meas))),float)
                k_err = np.fromiter((hom_meas[i][err_index] for i in range(len(hom_meas))),float)

                print(f'Fitting Q{t} value to measured moment k{t}\n')

                Q_fit, _ = curve_fit(k_fit, arcmins, k_meas, sigma=k_err)

                print('Saving on file\n')

                np.savetxt(homs_out+f'{cosmo}_{method}_{zs}_C{t}.txt', Ct)
                np.savetxt(homs_out+f'{cosmo}_{method}_{zs}_Q{t}_fit.txt',Q_fit)

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')

