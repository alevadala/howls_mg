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
# inpath = '/home/alessandro/phd/outputs_howls-like_filter_settings/'
inpath = '/home/alessandro/phd/new_k2pcf/outputs/'
dv_path = inpath+'DVs/kappa_2pcf/'
mean_path = inpath+'mean_DVs/kappa_2pcf/'

# Output path
outpath = 'Dustgrain_outs/'
outpath_bins = outpath+'25bins/'
two_pt_out = outpath_bins+'2PCF/'
plot_out = outpath_bins+'2PCF_plots/'
mean_out = plot_out+'2PCF_mean/'
noisy_out = plot_out+'2PCF_noisy/'
lin_out = plot_out+'2PCF_lin/'
vincenzo = outpath_bins+'Vincenzo/2PCF/'
os.makedirs(two_pt_out, exist_ok=True)
os.makedirs(lin_out, exist_ok=True)
os.makedirs(mean_out, exist_ok=True)
os.makedirs(noisy_out, exist_ok=True)
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
init = -1.4 # Initial step for the offset
bs = -0.3 # Bias
mu_j = 0 # Order of the Bessel function
conv_factor = 206265/60 # Conversion factor from radians to arcmins
correction = np.sqrt(np.pi/2)*conv_factor # Total correction to account for conversion and spherical Bessel

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

        xi_fft = fft.fht(cls*l_array*correction,dln=dl,mu=mu_j,offset=offset,bias=bs)/theta/2/np.pi
        # Saving the 2PCF values
        np.savetxt(two_pt_out+f'{cosmo}_{zs}_2pcf.txt',xi_fft)

        # File for Vincenzo
        np.savetxt(vincenzo+f'{cosmo}_{zs}.txt',np.column_stack((np.log10(theta),xi_fft)),delimiter=',',header='log theta, xi(theta)')
        
        # Loading mean DVs
        if cosmo == 'lcdm':
            name_cosmo = 'LCDM'
        else:
            name_cosmo = plot_label.replace(' ','_')

        two_pt = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_kappa_2pcf.txt',usecols=1)
        xr = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_kappa_2pcf.txt',usecols=0)

        two_pt_noisy = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_noisy_{name_cosmo}_z_{zs}_kappa_2pcf.txt',usecols=1)
        xr_noisy = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_noisy_{name_cosmo}_z_{zs}_kappa_2pcf.txt',usecols=0)

        print(f'Computing errors and covariance matrix for {cosmo} at zs={zs}\n')

        # Defining arrays for errors
        sigma_xi = np.zeros(25)
        sigma_xi_noisy = np.zeros(25)

        for n_bin in range(25):    
            bin_err = []
            bin_err_noisy = []
            
            for n_map in range(256):

                noise_tag = f'noisy_shapenoiseseed_{n_map+1}'                
                map_path_noisy = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_{noise_tag}_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
                map_err_noisy = np.loadtxt(map_path_noisy,usecols=4)
                bin_err_noisy.append(map_err_noisy[n_bin])

                map_path = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
                map_err = np.loadtxt(map_path,usecols=4)
                bin_err.append(map_err[n_bin])
                
            sigma_xi[n_bin] = np.std(np.array(bin_err))/np.sqrt(256)
            sigma_xi_noisy[n_bin] = np.std(np.array(bin_err_noisy))/np.sqrt(256)

        # Defining the values matrices
        bin_matrix = np.zeros((25,256))
        bin_matrix_noise = np.zeros((25,256))
        
        for n_map in range(256):
            noise_tag = f'noisy_shapenoiseseed_{n_map+1}'                
            map_noise = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_{noise_tag}_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
            xi_noise = np.loadtxt(map_path,usecols=3)
            bin_matrix_noise[:,n_map] = xi_noise

            map_path = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
            xi_map = np.loadtxt(map_path,usecols=3)
            bin_matrix[:,n_map] = xi_map

        # Defining the covariance matrices
        cov_matr  = np.cov(bin_matrix)
        cov_matr_noisy = np.cov(bin_matrix_noise)

        print(f'Saving plots for {cosmo} at zs={zs}\n')

        # Plotting and saving measurements/theory comparison for mean values
        plt.figure(figsize=(10,8))
        plt.scatter(xr,two_pt*xr**2,label='Measured',color='k',marker='.')
        plt.errorbar(xr,two_pt*xr**2,yerr=sigma_xi*xr**2/np.log(10)/two_pt,color='k',label=r'Mean error')
        plt.errorbar(xr,two_pt*xr**2,yerr=np.diagonal(cov_matr)*xr**2/np.log(10)/two_pt,fmt='none',color='magenta',label='Cov. matrix error')
        plt.plot(theta,xi_fft*theta**2,label='Theory',color='r',linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.ylim(1e-6,)
        plt.title(rf'{plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\theta^{2}\,\xi \, (\theta)$',fontsize=14)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='lower right')
        plt.savefig(mean_out+f'{cosmo}_zs={zs}.png', dpi=300) 
        plt.clf()
        plt.close('all')

        plt.figure(figsize=(10,8))
        plt.scatter(xr_noisy,two_pt_noisy*xr_noisy**2,label='Measured',color='k',marker='.')
        plt.errorbar(xr_noisy,two_pt_noisy*xr_noisy**2,yerr=sigma_xi_noisy*xr_noisy**2/np.log(10)/two_pt,color='k',label=r'Mean error')
        plt.errorbar(xr_noisy,two_pt_noisy*xr_noisy**2,yerr=np.diagonal(cov_matr_noisy)*xr_noisy**2/np.log(10)/two_pt_noisy,fmt='none',color='magenta',label='Cov. matrix error')
        plt.plot(theta,xi_fft*theta**2,label='Theory',color='r',linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.ylim(1e-6,)
        plt.title(rf'Noisy, {plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\theta^{2}\,\xi \, (\theta)$',fontsize=14)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='lower right')
        plt.savefig(noisy_out+f'{cosmo}_zs={zs}.png', dpi=300) 
        plt.clf()
        plt.close('all')

        plt.figure(figsize=(10,8))
        plt.scatter(xr,two_pt,label='Measured',color='k',marker='.')
        plt.errorbar(xr,two_pt,yerr=sigma_xi/np.log(10)/two_pt,color='k',label=r'Mean error')
        plt.errorbar(xr,two_pt,yerr=np.diagonal(cov_matr)/np.log(10)/two_pt,fmt='none',color='magenta',label='Cov. matrix error')
        plt.plot(theta,xi_fft,label='Theory',color='r',linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.title(rf'Mean values, {plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\xi \, (\theta)$',fontsize=14)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='upper right')
        plt.savefig(lin_out+f'{cosmo}_zs={zs}_lin.png', dpi=300) 
        plt.clf()
        plt.close('all')

        plt.figure(figsize=(10,8))
        plt.scatter(xr,two_pt,label='Measured',color='k',marker='.')
        plt.errorbar(xr_noisy,two_pt_noisy,yerr=sigma_xi_noisy/np.log(10)/two_pt,color='k',label=r'Mean error')
        plt.errorbar(xr_noisy,two_pt_noisy,yerr=np.diagonal(cov_matr_noisy)/np.log(10)/two_pt_noisy,fmt='none',color='magenta',label='Cov. matrix error')
        plt.plot(theta,xi_fft,label='Theory',color='r',linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.title(rf'Noisy, {plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\xi \, (\theta)$',fontsize=14)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='upper right')
        plt.savefig(lin_out+f'{cosmo}_zs={zs}_lin_noise.png', dpi=300) 
        plt.clf()
        plt.close('all')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')