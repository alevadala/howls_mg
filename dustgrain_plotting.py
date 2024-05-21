import numpy as np

import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from time import time

# Measured DVs path
# inpath = '/home/alessandro/phd/outputs_howls-like_filter_settings/'
inpath = '/home/alessandro/phd/new_k2pcf/outputs/'
dv_path = inpath+'DVs/kappa_2pcf/'
mean_path = inpath+'mean_DVs/kappa_2pcf/'

# Output path
outpath = 'Dustgrain_outs/'
pk_path = outpath+'Pknl/'
cls_path = outpath+'Cls/'
outpath_bins = outpath+'25bins/'
twopt_path = outpath_bins+'2PCF/'
plots_path = outpath+'Plots/'

pk_plots = plots_path+'Pknl_plots/'
cls_plots = plots_path+'Cls_plots/'
twopt_plots = plots_path+'2PCF_plots/'
mean_plots = twopt_plots+'Mean/'
noisy_plots = twopt_plots+'Noisy/'

squared_plots = mean_plots+'2PCF_squared/'
lin_plots = mean_plots+'2PCF_lin/'
grid_plots = mean_plots+'2PCF_grid/'

squared_noisy = noisy_plots+'2PCF_squared/'
lin_noisy = noisy_plots+'2PCF_lin/'
grid_noisy = noisy_plots+'2PCF_grid/'

print('Creating directories\n')
os.makedirs(pk_plots, exist_ok=True)
os.makedirs(cls_plots, exist_ok=True)
os.makedirs(twopt_plots, exist_ok=True)
os.makedirs(mean_plots, exist_ok=True)
os.makedirs(noisy_plots, exist_ok=True)

os.makedirs(squared_plots, exist_ok=True)
os.makedirs(lin_plots, exist_ok=True)
os.makedirs(grid_plots, exist_ok=True)

os.makedirs(squared_noisy, exist_ok=True)
os.makedirs(lin_noisy, exist_ok=True)
os.makedirs(grid_noisy, exist_ok=True)

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Setting the array for l with logarithically equispaced values
ell_points = 5000
l_min = 1
l_max = 5
l_array = np.logspace(l_min,l_max,ell_points)

# Switch the Limber correction on/off
correct_limber = 1

# Correction to the Limber approximation for low ls
def limber_correction(l):
    return (l+2)*(l+1)*l*(l-1)*(2/(2*l+1))**4

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

    # Plotting P(z,k)
    plt.figure(figsize=(8,6))
    sns.set_theme(style='whitegrid')
    sns.set_palette('colorblind')

    for method in methods:
        k = np.loadtxt(pk_path+f'k_{cosmo}_{method}.txt')
        z = np.loadtxt(pk_path+f'z_{cosmo}_{method}.txt')
        pk_in = f'{cosmo}_{method}.txt'
        plt_label = f'{method}'
        
        with open(pk_path+pk_in,'r',newline='\n') as pk_impo:
            pk_nonlin = np.zeros((len(z),len(k)))
            reader = csv.reader(pk_impo)
            for i,row in enumerate(reader):
                pk_nonlin[i,:] = row

        for i, (redshift, line) in enumerate(zip(z,['-'])):
            plt.loglog(k, pk_nonlin[i,:], ls = line, label=plt_label)

    plt.xlabel(r'k/h $[Mpc^{-1}]$', fontsize=14)
    plt.title(plot_label+', P(z,k) [(Mpc/h)^3]', fontsize=16)
    plt.legend()
    plt.savefig(pk_plots+f'{cosmo}.png', dpi=300) 
    plt.clf()
    plt.close('all')

    # 2PCF grid plots
    for noisy in [0,1]:
        fig, axs = plt.subplots(len(methods), len(zs_values), figsize=(30,20), dpi=200, sharex=True, sharey=False)
        sns.set_theme(style='whitegrid')
        sns.set_palette(palette='colorblind')
        for i,method in enumerate(methods):
            for j,zs_val in enumerate(zs_values):
                        
                if noisy:
                    mean_tag = 'noisy'
                else:
                    mean_tag = 'true'
                    
                if cosmo == 'lcdm':
                    name_cosmo = 'LCDM'
                else:
                    name_cosmo = plot_label.replace(' ','_')
                    
                two_pt = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_{mean_tag}_{name_cosmo}_z_{zs_val}_kappa_2pcf.txt',usecols=1)
                xr = np.loadtxt(mean_path+f'LCDM_DUSTGRAIN_convergence_{mean_tag}_{name_cosmo}_z_{zs_val}_kappa_2pcf.txt',usecols=0)
                    
                for alg in ['FHT', 'Bes']:
                    xi_theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs_val}_{alg}_2pcf.txt')
                    theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs_val}_{alg}_theta.txt')
                    axs[i,j].plot(theta,xi_theta,label=f'{alg}',linestyle='--')
                
                bin_matrix = np.zeros((25,256))
                
                for n_map in range(256):
                    if noisy:
                        noise_tag = f'noisy_shapenoiseseed_{n_map+1}'
                    else:
                        noise_tag = 'true'
                    map_path = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_{noise_tag}_{name_cosmo}_z_{zs_val}_kappa_2pcf.txt'
                    xi_map = np.loadtxt(map_path,usecols=3)
                    bin_matrix[:,n_map] = xi_map

                cov_matr = np.cov(bin_matrix)

                axs[i,j].scatter(xr,two_pt,color='k',marker='.',label='Measurements')
                axs[i,j].errorbar(xr,two_pt,yerr=np.diagonal(cov_matr)/np.log(10)/two_pt,color='k')
                if noisy:
                    axs[i,j].set_title(rf'Noisy, {plot_label} - {method}, $z_s$={zs_val}')
                else:
                    axs[i,j].set_title(rf'{plot_label} - {method}, $z_s$={zs_val}')
                axs[i,j].set_xscale('log')
                axs[i,j].set_yscale('log')
                axs[i,j].set(xlabel=r'$\theta$ (arcmin)', ylabel=r'$\xi \, (\theta)$')
                axs[i,j].set_xlim(0.5,50)
                axs[i,j].set_xticks([0.5,1,10,50],[0.5,1,10,50])
                axs[i,j].legend()
        if noisy:
            plt.savefig(grid_noisy+f'{cosmo}_noise.png', dpi=200) 
        else:
            plt.savefig(grid_plots+f'{cosmo}.png', dpi=200) 
        plt.clf()
        plt.close('all')

    for zs in zs_values:

        print(f'Importing C(l) for {cosmo} at zs={zs}\n')
        
        # Plotting and saving C(l) plots
        plt.figure(figsize=(8,6))
        sns.set_theme(style='whitegrid')
        sns.set_palette('colorblind')

        # Import pre-computed the C(l)s
        for method in methods:
            if correct_limber:
                limber_term = np.fromiter((limber_correction(l) for l in l_array),float)
                cls = np.loadtxt(cls_path+f'{cosmo}_{method}_{zs}.txt')
                cls = limber_term*cls
            else:
                cls = np.loadtxt(cls_path+f'{cosmo}_{method}_{zs}.txt')
            plt.loglog(l_array,l_array*(l_array+1)*cls/2/np.pi,label=f'{method}')
        plt.xlim(l_array.min(),l_array.max())
        plt.title(rf'{plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\ell$',fontsize=14)
        plt.ylabel(r'$\ell \, (\ell + 1) \, P_{\kappa}(\ell) / (2\pi)$',fontsize=14)
        plt.legend()
        plt.savefig(cls_plots+f'{cosmo}_zs={zs}.png', dpi=300) 
        plt.clf()
        plt.close('all')
        
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

        # # Defining arrays for errors
        # sigma_xi = np.zeros(25)
        # sigma_xi_noisy = np.zeros(25)

        # for n_bin in range(25):    
        #     bin_err = []
        #     bin_err_noisy = []
            
        #     for n_map in range(256):

        #         noise_tag = f'noisy_shapenoiseseed_{n_map+1}'                
        #         map_path_noisy = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_{noise_tag}_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
        #         map_err_noisy = np.loadtxt(map_path_noisy,usecols=4)
        #         bin_err_noisy.append(map_err_noisy[n_bin])

        #         map_path = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
        #         map_err = np.loadtxt(map_path,usecols=4)
        #         bin_err.append(map_err[n_bin])
                
        #     sigma_xi[n_bin] = np.std(np.array(bin_err))/np.sqrt(256)
        #     sigma_xi_noisy[n_bin] = np.std(np.array(bin_err_noisy))/np.sqrt(256)

        # Defining the values matrices
        bin_matrix = np.zeros((25,256))
        bin_matrix_noise = np.zeros((25,256))
        
        for n_map in range(256):
            noise_tag = f'noisy_shapenoiseseed_{n_map+1}'                
            map_noise = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_{noise_tag}_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
            xi_noise = np.loadtxt(map_noise,usecols=3)
            bin_matrix_noise[:,n_map] = xi_noise

            map_path = dv_path+f'{str(n_map).zfill(3)}_LCDM_DUSTGRAIN_convergence_true_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
            xi_map = np.loadtxt(map_path,usecols=3)
            bin_matrix[:,n_map] = xi_map

        # Defining the covariance matrices
        cov_matr  = np.cov(bin_matrix)
        cov_matr_noisy = np.cov(bin_matrix_noise)

        print(f'Saving 2PCF plots for {cosmo} at zs={zs}\n')

        # Plotting and saving measurements/theory comparison for mean values

        # 2PCF squared plot
        plt.figure(figsize=(8,6))
        sns.set_theme(style='whitegrid')
        sns.set_palette('colorblind')
        plt.scatter(xr,two_pt*xr**2,color='k',marker='.',label='Measurements')
        for method in methods:
            xi_fft = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_2pcf.txt')
            xi_theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_2pcf.txt')
            theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_theta.txt')
            th = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_theta.txt')
            plt.plot(th,xi_theta*th**2,label=f'{method} - Bes',linestyle='-')
            plt.plot(theta,xi_fft*theta**2,label=f'{method} - FHT',linestyle='--')
        plt.errorbar(xr,two_pt*xr**2,yerr=np.diagonal(cov_matr)*xr**2/np.log(10)/two_pt,color='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.ylim(1e-6,)
        plt.title(rf'{plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\theta^2 \; \xi \, (\theta)$',fontsize=14)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='lower right')
        plt.savefig(squared_plots+f'{cosmo}_zs={zs}.png', dpi=300) 
        plt.clf()
        plt.close('all')

        # 2PCF squared plots noisy
        plt.figure(figsize=(8,6))
        sns.set_theme(style='whitegrid')
        sns.set_palette('colorblind')
        for method in methods:
            cl = np.loadtxt(cls_path+f'{cosmo}_{method}_{zs}.txt')
            xi_fft = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_2pcf.txt')
            xi_theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_2pcf.txt')
            theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_theta.txt')
            th = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_theta.txt')
            plt.plot(th,xi_theta*th**2,label=f'{method} - Bes',linestyle='-')
            plt.plot(theta,xi_fft*theta**2,label=f'{method} - FHT',linestyle='--')
        plt.scatter(xr_noisy,two_pt_noisy*xr_noisy**2,label='Measurements',color='k',marker='.')
        plt.errorbar(xr_noisy,two_pt_noisy*xr_noisy**2,yerr=np.diagonal(cov_matr_noisy)*xr_noisy**2/np.log(10)/two_pt_noisy,color='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.ylim(1e-6,)
        plt.title(rf'Noisy, {plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\theta^{2}\,\xi \, (\theta)$',fontsize=14)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='lower right')
        plt.savefig(squared_noisy+f'{cosmo}_zs={zs}_noise.png', dpi=300) 
        plt.clf()
        plt.close('all')

        # 2PCF linear plots
        plt.figure(figsize=(8,6))
        sns.set_theme(style='whitegrid')
        sns.set_palette('colorblind')
        plt.scatter(xr,two_pt,label='Measured',color='k',marker='.')
        plt.errorbar(xr,two_pt,yerr=np.diagonal(cov_matr)/np.log(10)/two_pt,color='k')
        for method in methods:
            cl = np.loadtxt(cls_path+f'{cosmo}_{method}_{zs}.txt')
            xi_fft = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_2pcf.txt')
            xi_theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_2pcf.txt')
            theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_theta.txt')
            th = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_theta.txt')
            plt.plot(th,xi_theta,label=f'{method} - Bes',linestyle='-')
            plt.plot(theta,xi_fft,label=f'{method} - FHT',linestyle='--')   
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.ylim(1e-6,)
        plt.title(rf'{plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=16)
        plt.ylabel(r'$\xi \, (\theta)$',fontsize=16)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='upper right')
        plt.savefig(lin_plots+f'{cosmo}_zs={zs}.png', dpi=300) 
        plt.clf()
        plt.close('all')

        # 2PCF linear plots noisy
        plt.figure(figsize=(8,6))
        sns.set_theme(style='whitegrid')
        sns.set_palette('colorblind')
        for method in methods:
            cl = np.loadtxt(cls_path+f'{cosmo}_{method}_{zs}.txt')
            xi_fft = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_2pcf.txt')
            xi_theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_2pcf.txt')
            theta = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_FHT_theta.txt')
            th = np.loadtxt(twopt_path+f'{cosmo}_{method}_{zs}_Bes_theta.txt')
            plt.plot(th,xi_theta,label=f'{method} - Bes',linestyle='-')
            plt.plot(theta,xi_fft,label=f'{method} - FHT',linestyle='--')  
        plt.scatter(xr_noisy,two_pt_noisy,label='Measurements',color='k',marker='.')
        plt.errorbar(xr_noisy,two_pt_noisy,yerr=np.diagonal(cov_matr_noisy)/np.log(10)/two_pt_noisy,color='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.5,50)
        plt.title(rf'Noisy, {plot_label}, $z_s$={zs}',fontsize=16)
        plt.xlabel(r'$\theta$ (arcmin)',fontsize=14)
        plt.ylabel(r'$\xi \, (\theta)$',fontsize=14)
        plt.xticks([0.5,1,10,50],[0.5,1,10,50])
        plt.legend(loc='upper right')
        plt.savefig(lin_noisy+f'{cosmo}_zs={zs}_noise.png', dpi=300) 
        plt.clf()
        plt.close('all')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')