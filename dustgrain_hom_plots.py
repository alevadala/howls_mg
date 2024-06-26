import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

from time import time

# Output path
outpath = 'Dustgrain_outs/'
homs_path = outpath+'HOMs/'
dv_path = '/home/alessandro/phd/kappa_moments/'

homs_plots = outpath+'HOMs_plots/'

print('Creating directories\n')
os.makedirs(homs_plots, exist_ok=True)

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Smoothing scale range (in arcmins)
theta_s = np.linspace(2,10,50)

# Fiducial values for Q parameters (in LCDM)
fid_Q2 = 1
fid_Q3 = 3
fid_Q4 = 12*7.29+4*16.23
# To list
fid_Q = [fid_Q2, fid_Q3, fid_Q4]

# Headers of the HOMs file
homs_cols = ['smoothing','k2','sigk2','k3','sigk3','k4','sigk4','S3','sigS3','S4','sigS4']

t1 = time()

for cosmo in cosmos:

    if cosmo == 'lcdm':
        methods = ['HM', 'EE2', 'Win']
    else:
        methods = ['RHM','RHF', 'Win']

    print(f'Starting plotting for {cosmo}\n')

    # LCDM parameters values
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
    
    if cosmo == 'lcdm':
        name_cosmo = 'LCDM'
    else:
        name_cosmo = plot_label.replace(' ','_')

    # HOMs grid plots
    fig, axs = plt.subplots(3, len(zs_values), figsize=(30,20), dpi=300, sharex=True, sharey=False)
    sns.set_theme(style='whitegrid')
    sns.set_palette(palette='colorblind')
    for i,t in enumerate([2,3,4]):

        for j,zs in enumerate(zs_values):

            hom_file = dv_path+f'LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_filter_tophat_scales_[15.97 32.02 64.03]_pixels_kappa_moments.txt'
            hom_meas = np.loadtxt(hom_file)
            arcm = np.fromiter((hom_meas[i][0] for i in range(3)),float)

            k_index = homs_cols.index('k'+f'{t}')
            err_index = homs_cols.index('sigk'+f'{t}')
                        
            k_meas = np.fromiter((hom_meas[i][k_index] for i in range(3)),float)
            k_err = np.fromiter((hom_meas[i][err_index] for i in range(3)),float)

            for method in methods:
                
                Ct = np.loadtxt(homs_path+f'{cosmo}_{method}_{zs}_C{t}.txt')

                Q_fit = np.loadtxt(homs_path+f'{cosmo}_{method}_{zs}_Q{t}_fit.txt')
                
                axs[i,j].semilogy(theta_s,Q_fit*Ct,label=f'{method} - '+r'$\mathcal{Q}$'+f'={Q_fit:.2f}')
                # axs[i,j].semilogy(theta_s,fid_Q[i]*Ct,label=r'$\mathcal{Q}$ fid'+f'={fid_Q[i]}',linestyle='--')

            axs[i,j].scatter(arcm,k_meas,marker='.',color='k',label='Measurements')
            axs[i,j].errorbar(arcm,k_meas,yerr=k_err,fmt='None',color='k')
            axs[i,j].set_xlim(2,10)
            axs[i,j].legend()
            axs[i,j].set(xlabel=r'$\theta_s$ (arcmin)')
            axs[i,j].set_title(rf'{plot_label} - $\langle \kappa^{t} \rangle$ - $z_s$={zs}')
    plt.savefig(homs_plots+f'{cosmo}_grid.png', dpi=300)
    plt.clf()
    plt.close('all')

    for zs in zs_values:

        hom_file = dv_path+f'LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_filter_tophat_scales_[15.97 32.02 64.03]_pixels_kappa_moments.txt'
        hom_meas = np.loadtxt(hom_file)
        arcm = np.fromiter((hom_meas[i][0] for i in range(3)),float)

        for i,t in enumerate([2,3,4]):

            k_index = homs_cols.index('k'+f'{t}')
            err_index = homs_cols.index('sigk'+f'{t}')
            k_meas = np.fromiter((hom_meas[i][k_index] for i in range(3)),float)
            k_err = np.fromiter((hom_meas[i][err_index] for i in range(3)),float)

            # HOMs single plot
            plt.figure(figsize=(8,6))
            sns.set_theme(style='whitegrid')
            sns.set_palette('colorblind')
            plt.scatter(arcm,k_meas,color='k',marker='.',label='Measurements')
            plt.errorbar(arcm,k_meas,yerr=k_err,fmt='None')

            for method in methods:
                Ct = np.loadtxt(homs_path+f'{cosmo}_{method}_{zs}_C{t}.txt')
                Q_fit = np.loadtxt(homs_path+f'{cosmo}_{method}_{zs}_Q{t}_fit.txt')
                plt.semilogy(theta_s,Q_fit*Ct, label=f'{method} - '+r'$\mathcal{Q}$'+f'={Q_fit:.2f}')
                plt.semilogy(theta_s,fid_Q[i]*Ct,label=r'$\mathcal{Q}$ fid'+f'={fid_Q[i]:.2f}',linestyle='--')
            
            plt.xlim(2,10)
            plt.title(rf'{plot_label} - $\langle \kappa^{t} \rangle$ - $z_s$={zs}',fontsize=16)
            plt.xlabel(r'$\theta_s$ (arcmin)',fontsize=14)
            plt.legend()
            plt.savefig(homs_plots+f'{cosmo}_k{t}_zs={zs}.png', dpi=300) 
            plt.clf()
            plt.close('all')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')