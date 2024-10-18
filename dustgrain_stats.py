import numpy as np

from scipy.stats import chisquare, kstest

from scipy.interpolate import CubicSpline

from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

import os

from time import time

# Output path
outpath = '/home/alessandro/phd/scripts/Dustgrain_outs/'
stats_out = outpath+'stats/'
plots_out = outpath+'Plots/Stats/'
os.makedirs(stats_out, exist_ok=True)
os.makedirs(plots_out, exist_ok=True)

# Input path
twoth_path = outpath+'25bins/2PCF/'
homsth_path = outpath+'HOMs/'
dv_path = '/home/alessandro/phd/MG_Paper_outputs_FINAL_mean_DVs/outputs_FINAL_mean_DVs/mean_DVs/'
homs_path = dv_path+'kappa_moments/'
twopt_path = dv_path+'kappa_2pcf/'

# DUSTGRAIN redshift values
zs_values = [0.5, 1.0, 2.0, 4.0]

# Cosmologies tag
cosmos = ['lcdm', 'fr4','fr5', 'fr6', 'fr4_0.3', 'fr5_0.1', 'fr5_0.15', 'fr6_0.1', 'fr6_0.06']

# Smoothing scale range (in arcmins)
theta_s = np.linspace(.2,22,50)

# Headers of the DVs file
homs_cols = ['smoothing','k2','sigk2','k3','sigk3','k4','sigk4']
twopt_cols = ['theta', 'xi', 'sigxi']

# Select noisy or noisless DVs
noisy = True

# SMAPE in range 0-200%
def SMAPE(measured, predicted):
    abs_diff = np.abs(predicted - measured)
    abs_sum = np.abs(measured) + np.abs(predicted)
    smape = 200 * np.mean((abs_diff/abs_sum))
    return smape

if noisy:
    noise_tag = 'noisy'
else:
    noise_tag = 'true'

# Stats file header
stats_header = 'K-S, RMSE, MAPE, R2-score, RMSE, SMAPE'

t1 = time()

for cosmo in cosmos:

    print(f'Starting computation for {cosmo} in case:{noise_tag}\n')

    # Selecting methods
    if cosmo == 'lcdm':
        methods = ['HM', 'EE2', 'Win']
    else:
        methods = ['RHM','RHF', 'Win']

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
        
    r2_2pcf = np.zeros(shape=(6,4))
    smape_2pcf = np.zeros(shape=(6,4))

    r2_homs = np.zeros(shape=(9,4))
    smape_homs = np.zeros(shape=(9,4))

    labels_2pcf = []
    labels_homs = []

    for j,zs in enumerate(zs_values):

        i_2pcf = 0
        i_homs = 0

        homs_file = homs_path+f'LCDM_DUSTGRAIN_convergence_{noise_tag}_{name_cosmo}_z_{zs}_filter_tophat_scales_[ 4  8 16 32]_pixels_kappa_moments.txt'        
        smooth_index = homs_cols.index('smoothing')
        arcm = np.loadtxt(homs_file,usecols=smooth_index)

        twopt_file = twopt_path+f'LCDM_DUSTGRAIN_convergence_{noise_tag}_{name_cosmo}_z_{zs}_kappa_2pcf.txt'
        theta_index = twopt_cols.index('theta')
        xi_index = twopt_cols.index('xi')
        xierr_index = twopt_cols.index('sigxi')

        # Selecting elements != 0 fro which theoretical prediction is available
        theta_meas = np.loadtxt(twopt_file,usecols=theta_index)
        theta_meas = theta_meas[5:35] # ~1-60
        twopt = np.loadtxt(twopt_file,usecols=xi_index)
        twopt = twopt[5:35]
        err_xi = np.loadtxt(twopt_file,usecols=xierr_index)
        err_xi = err_xi[5:35]

        for method in methods:
            for alg in ['FHT', 'Bes']:
                xi_theta = np.loadtxt(twoth_path+f'{cosmo}_{method}_{zs}_{alg}_2pcf.txt')
                theta_th = np.loadtxt(twoth_path+f'{cosmo}_{method}_{zs}_{alg}_theta.txt')
                xi_spline = CubicSpline(theta_th, xi_theta)
                xi_intp = xi_spline(theta_meas)
                
                # chi2 = chisquare(twopt, xi_intp)
                KS = kstest(twopt, xi_intp)
                mape = mean_absolute_percentage_error(twopt, xi_intp)
                r2 = r2_score(twopt, xi_intp)
                rmse = root_mean_squared_error(twopt, xi_intp)
                smape = SMAPE(twopt, xi_intp)

                r2_2pcf[i_2pcf,j] = r2
                smape_2pcf[i_2pcf,j] = smape

                labels_2pcf.append(f'{method} - {alg}')

                i_2pcf+=1

                np.savetxt(stats_out+f'2PCF_{cosmo}_{zs}_{method}_{alg}_fit_stats.txt',np.column_stack((KS.statistic,mape,r2,rmse,smape)),header=stats_header)

            for i,t in enumerate([2,3,4]):
                k_index = homs_cols.index('k'+f'{t}')
                kerr_index = homs_cols.index('sigk'+f'{t}')
                k_meas = np.loadtxt(homs_file,usecols=k_index)
                k_err = np.loadtxt(homs_file,usecols=kerr_index)

                Ct = np.loadtxt(homsth_path+f'{cosmo}_{method}_{zs}_C{t}.txt')
                Q_fit = np.loadtxt(homsth_path+f'{cosmo}_{method}_{zs}_Q{t}_fit.txt')
                k_fit = Q_fit*Ct
                k_spline = CubicSpline(theta_s, k_fit)
                k_intp = k_spline(arcm)

                # chi2 = chisquare(k_meas, k_intp)
                KS = kstest(k_meas, k_intp)
                mape = mean_absolute_percentage_error(k_meas, k_intp)
                r2 = r2_score(k_meas, k_intp)
                rmse = root_mean_squared_error(k_meas, k_intp)
                smape = SMAPE(k_meas, k_intp)

                r2_homs[i_homs,j] = r2
                smape_homs[i_homs,j] = smape

                labels_homs.append(rf'$\langle \kappa{t} \rangle$ - {method}')

                i_homs+=1

                np.savetxt(stats_out+f'k{t}_{cosmo}_{zs}_{method}_fit_stats.txt',np.column_stack((KS.statistic,mape,r2,rmse,smape)),header=stats_header)

    print('Saving files and plots\n')
    
    # # R2
    # plt.figure(figsize=(8,6))
    # sns.set_theme(style='whitegrid')
    # sns.set_palette('colorblind')
    # plt.title(f'2PCF - {plot_label}')
    # for r in range(len(r2_2pcf)):
    #     plt.plot(zs_values, r2_2pcf[r,:],marker='.')
    # plt.xticks(zs_values)
    # plt.ylabel(r'$R^2$ score')
    # plt.xlabel(r'$z_s$')
    # plt.legend(labels_2pcf,ncols=3,fontsize='small',framealpha=0.3)
    # plt.savefig(plots_out+f'2PCF_{cosmo}_R2.png', dpi=300)
    # plt.clf()
    # plt.close('all')

    # fig, axs = plt.subplots(2,2, figsize=(15,10), sharex=True, sharey=False)
    # plt.suptitle(f'{plot_label}')
    # sns.set_theme(style='whitegrid')
    # sns.set_palette(palette='colorblind')
    # for r in range(len(r2_homs)):
    #     axs[0,0].plot(zs_values, r2_homs[r,:],marker='.')
    #     axs[0,0].set(xlabel=r'$z_s$', ylabel=r'$R^2$ score')
    #     axs[0,0].set_xticks(zs_values)
    #     axs[0,0].legend(labels_homs,ncols=3,fontsize='x-small',framealpha=0.3)
    # for t in [2,3,4]:
    #     if t == 2:
    #         i_plot, j_plot = 0,1
    #     else:
    #         i_plot, j_plot = 1, (t-3)
    #     for r in range(len(r2_homs)):
    #         if rf'$\langle \kappa{t} \rangle$' in labels_homs[r]:
    #             t_index = labels_homs.index(labels_homs[r])
    #             axs[i_plot,j_plot].plot(zs_values, r2_homs[t_index,:],marker='.',label=labels_homs[r])
    #     axs[i_plot,j_plot].set_xticks(zs_values)
    #     axs[i_plot,j_plot].set(xlabel=r'$z_s$',ylabel=r'$R^2$ score')
    #     axs[i_plot,j_plot].legend(fontsize='small',framealpha=0.3)
    # plt.savefig(plots_out+f'HOMs_{cosmo}_R2.png', dpi=300)
    # plt.clf()
    # plt.close('all')

    # # SMAPE
    # plt.figure(figsize=(8,6))
    # sns.set_theme(style='whitegrid')
    # sns.set_palette('colorblind')
    # plt.title(f'2PCF - {plot_label}')
    # for r in range(len(smape_2pcf)):
    #     plt.plot(zs_values, smape_2pcf[r,:],marker='.')
    # plt.xticks(zs_values)
    # plt.ylabel('SMAPE')
    # plt.xlabel(r'$z_s$')
    # plt.legend(labels_2pcf,ncols=3,fontsize='small',framealpha=0.3)
    # plt.savefig(plots_out+f'2PCF_{cosmo}_SMAPE.png', dpi=300)
    # plt.clf()
    # plt.close('all')

    # fig, axs = plt.subplots(2,2, figsize=(15,10), sharex=True, sharey=False)
    # plt.suptitle(f'{plot_label}')
    # sns.set_theme(style='whitegrid')
    # sns.set_palette(palette='colorblind')
    # for r in range(len(smape_homs)):
    #     axs[0,0].plot(zs_values, smape_homs[r,:],marker='.')
    #     axs[0,0].set(xlabel=r'$z_s$', ylabel='SMAPE')
    #     axs[0,0].set_xticks(zs_values)
    #     axs[0,0].legend(labels_homs,ncols=3,fontsize='x-small',framealpha=0.3)
    # for t in [2,3,4]:
    #     if t == 2:
    #         i_plot, j_plot = 0,1
    #     else:
    #         i_plot, j_plot = 1, (t-3)
    #     for r in range(len(smape_homs)):
    #         if rf'$\langle \kappa{t} \rangle$' in labels_homs[r]:
    #             t_index = labels_homs.index(labels_homs[r])
    #             axs[i_plot,j_plot].plot(zs_values, smape_homs[t_index,:],marker='.',label=labels_homs[r])
    #     axs[i_plot,j_plot].set_xticks(zs_values)
    #     axs[i_plot,j_plot].set(xlabel=r'$z_s$',ylabel='SMAPE')
    #     axs[i_plot,j_plot].legend(fontsize='small',framealpha=0.3)
    # plt.savefig(plots_out+f'HOMs_{cosmo}_SMAPE.png', dpi=300)
    # plt.clf()
    # plt.close('all')

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')