##################################################################
## CHANGES:
## Removed plot for individual maps and for xi(theta), just of theta^2.xi(theta).
## Changed l range in 10-100000, with 5000 points instead of 10000.
## Order of the Bessel returned to 0, due to better agreement with data.
## Reduced theta range to ~ 0.5-100 (it depends on the shift w.r.t l range) 
## Introduced the limber correction term, with the possibility to switch it off.
## Reduced the correction on the conversion factor.
##################################################################

import numpy as np
import sys
sys.path.insert(0, '/home/alessandro/code')

from MGCAMB import camb

from scipy import fft

import os
import matplotlib.pyplot as plt

from time import time

# Measured DVs path
inpath = '/home/alessandro/phd/outputs_howls-like_filter_settings/'
dv_path = inpath+'DVs/kappa_2pcf/'
mean_path = inpath+'mean_DVs/kappa_2pcf/'

# Output path
outpath = 'Dustgrain_outs/'
two_pt_out = outpath+'2PCF/'
plot_out = outpath+'2PCF_plots/'
mean_out = plot_out+'2PCF_mean/'
lin_out = plot_out+'2PCF_lin/'
vincenzo = outpath+'Vincenzo/2PCF/'
os.makedirs(two_pt_out, exist_ok=True)
os.makedirs(lin_out, exist_ok=True)
os.makedirs(mean_out, exist_ok=True)
os.makedirs(vincenzo, exist_ok=True)



# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Setting the array for l with logarithically equispaced values
ell_points = 5000
l_min = 1
l_max = 5
l_array = np.logspace(l_min,l_max,ell_points)
dl = np.log(l_array[1]/l_array[0]) # Discrete Hankel transform step

# Limits of measured theta
# npix = 2048 #pixels
# pix_res = (5 / npix) * 60 #arcmins
# twopt_min  = pix_res
# twopt_max  = pix_res * npix * np.sqrt(2)

# Settings for the FHT
init = -1.2 # Initial step for the offset
bs = -0.5 # Bias
mu_j = 0 # Order of the Bessel function
conv_factor = 206265/60 # Conversion factor from radians to arcmins
correction_fr = np.sqrt(np.pi/2)*conv_factor**1.06 # Total correction to account for conversion and spherical Bessel
correction_lcdm = np.sqrt(np.pi/2)*conv_factor**0.97
offset = fft.fhtoffset(dl, initial=init, mu=mu_j, bias=bs) # Setting offset for low ringing condition
theta = np.exp(offset)*conv_factor/l_array[::-1] # Output theta array

# Switch the Limber correction on/off
correct_limber = 1

# Correction to the Limber approximation for low ls
def limber_correction(l):
    return (l+2)*(l+1)*l*(l-1)*(2/(2*l+1))**4

t1 = time()

for cosmo in cosmos:

    print(f'Starting computation for {cosmo}\n')
    # Plot labels
    # LCDM 
    if cosmo == 'lcdm':

        plot_label = r'$\Lambda$CDM'

    # f(R) models with massless neutrinos
    elif cosmo == 'fr4':

        plot_label = 'fR4'

    elif cosmo ==  'fr5':

        plot_label = 'fR5'

    elif cosmo ==  'fr6':

        plot_label = 'fR6'

    # f(R) models with different neutrino masses
    elif cosmo == 'fr4_0.3':

        plot_label = 'fR4 0.3eV'

    elif cosmo == 'fr5_0.1':

        plot_label = 'fR5 0.1eV'

    elif cosmo == 'fr5_0.15':

        plot_label = 'fR5 0.15eV'

    elif cosmo == 'fr6_0.1':

        plot_label = 'fR6 0.1eV'

    elif cosmo == 'fr6_0.06':

        plot_label = 'fR6 0.06eV'

    for zs in zs_values:

        print(f'Importing C(l) for {cosmo} at zs={zs}\n')

        # Import pre-computed the C(l)s
        cls = np.loadtxt(outpath+f'Cls/{cosmo}_{zs}.txt')

        if correct_limber:
            limber_term = np.fromiter((limber_correction(l) for l in l_array),float)
            cls = limber_term*cls
        
        # Computing the 2PCF through Hankel transform
        # The expression of the discrete Hankel transform has been modified to match
        # the canonical definition of the two-point c.f.
        print(f'Computing 2PCF for {cosmo} at zs={zs}\n')

        if cosmo == 'lcdm':
            correction = correction_lcdm
        else:
            correction = correction_fr

        xi_fft = fft.fht(cls*l_array*correction,dln=dl,mu=mu_j,offset=offset,bias=bs)/theta/2/np.pi
        # Saving the 2PCF values
        np.savetxt(two_pt_out+f'{cosmo}_{zs}_2pcf.txt',xi_fft)

        # File for Vincenzo
        np.savetxt(vincenzo+f'{cosmo}_{zs}.txt',np.column_stack((np.log10(theta),xi_fft)),delimiter=',',header='log theta, xi(theta)')

        print(f'Saving plots for {cosmo} at zs={zs}\n')
        
        # Loading mean DVs
        if cosmo == 'lcdm':
            two_pt = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_true_LCDM_z_{zs}_kappa_2pcf.txt',usecols=1)
            xr = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_true_LCDM_z_{zs}_kappa_2pcf.txt',usecols=0)
            err = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_true_LCDM_z_{zs}_kappa_2pcf.txt',usecols=2)
        else:
            two_pt = np.loadtxt(mean_path+'LCDM_DUSTGRAIN_convergence_true_'+ plot_label.replace(' ','_') + f'_z_{zs}_kappa_2pcf.txt',usecols=1)
            xr = np.loadtxt(mean_path+'LCDM_DUSTGRAIN_convergence_true_'+ plot_label.replace(' ','_') + f'_z_{zs}_kappa_2pcf.txt',usecols=0)
            err = np.loadtxt(mean_path+'LCDM_DUSTGRAIN_convergence_true_'+ plot_label.replace(' ','_') + f'_z_{zs}_kappa_2pcf.txt',usecols=2)

        # Plotting and saving measurements/theory comparison for mean values
        plt.figure(figsize=(8,6))
        plt.scatter(xr,two_pt*xr**2,label='Measured',color='k',marker='o')
        plt.errorbar(xr,two_pt*xr**2,yerr=err*1e4,fmt='none',color='k')
        plt.plot(theta,xi_fft*theta**2,label='Theory',color='r',linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.1,50)
        plt.ylim(1e-6,)
        plt.title(rf'$\kappa$ - 2PCF, {plot_label}, Mean values, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\theta^{2}\,\xi \, (\theta)$',fontsize=14)
        plt.xticks([0.1,1,10,50],[0.1,1,10,50])
        plt.legend(loc='lower right')
        plt.savefig(mean_out+f'{cosmo}_zs={zs}.png', dpi=300) 
        plt.clf()
        plt.close('all')

        # plt.figure(figsize=(8,6))
        # plt.scatter(xr,two_pt,label='Measured',color='k',marker='o')
        # plt.errorbar(xr,two_pt,yerr=err*1e3,fmt='none',color='k')
        # plt.plot(theta,xi_fft,label='Theory',color='r',linestyle='--')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlim(0.1,50)
        # plt.ylim(1e-6,)
        # plt.title(rf'$\kappa$ - 2PCF, {plot_label}, Mean values, $z_s$={zs}',fontsize=16)
        # plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        # plt.ylabel(r'$\xi \, (\theta)$',fontsize=14)
        # plt.xticks([0.1,1,10],[0.1,1,10])
        # plt.legend(loc='upper right')
        # plt.savefig(lin_out+f'{cosmo}_zs={zs}_lin.png', dpi=300) 
        # plt.clf()
        # plt.close('all')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')