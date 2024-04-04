### HIDE ALL WARNINGS (remove during debugging) ###
import warnings
warnings.filterwarnings('ignore')
###################################################


from astropy.io import fits
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import multiprocessing
from multiprocessing import Pool
import os 
import re
import itertools
import time
import sys
import treecorr
import lenstools # warning: requires emcee<=2.2.1
from lenstools import ConvergenceMap
from astropy.convolution import convolve_fft, Gaussian2DKernel, Tophat2DKernel
from lenspack.peaks import find_peaks2d, peaks_histogram
from skimage.measure import label, regionprops


###############
## Functions ##
###############


# CPU-Parallelising Function #
def parallel(func_name, loop_sequence, n_proc):

    # Make CTRL+C command interruption able to kill all parallel processes if needed [Source: https://code-examples.net/en/q/157d64]
    import signal
    
    def initializer():
        """Ignore CTRL+C in the worker process."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Main
    if __name__ == '__main__':
        
        parallel_processes = n_proc #int(multiprocessing.cpu_count()*0.5) # Number of parallel processes 
        pool = Pool(parallel_processes)  # Creates a multiprocessing Pool using parallel_processes threads. One loop iteration per core.
        
        try:
            
            func_result = pool.map(func_name, loop_sequence) # pool.map returns a list of result-tuples. len(pool.map) = loop_sequence.size and len(result-tuple) = number of variables returned by 'func_name'.
                                                             # if 'func_name' does not return anything, parallel() will work correctly formally returning 'func_result=None'. 
        except KeyboardInterrupt:
            
            pool.terminate()
            pool.join()
        
        pool.close() 
        
    return(func_result)


##----------------------------------------------------------------------------##     


# (AUTO-) Convergence-2PCF Function #
def measure_kappa_2pcf_oneLOS(los):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_2pcf'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    os.makedirs(DV_outpath, exist_ok=True)
    os.makedirs(plot_outpath, exist_ok=True)

    # Load kappa map
    # kappa_map_inpath = inpath + f'convergence_maps/{nbody_pipeline}_{cosmo}/{KS_inversion_type}/'
    # kappa_map_name = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}'
    # kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 
    # old paths, testing new paths for DUSTGRAIN f(R)

    # Load kappa map
    # added models different from LCDM through {cosmo}_zs= [...] - MODIFIED
    kappa_map_inpath = inpath + f'{cosmo}_zs={str(z_bin_index)}_1_kappaBapp/DUSTGRAIN-pathfinder/WeakLensing/downscaled/{cosmo}/output_relative/{str(los).zfill(2)}/'
    kappa_map_name = f'1_kappaBApp'
    kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 


    # Define map properties
    npix = kappaE_map.shape[0] # pixel number per side 
    pix_res = map_res # arcmins 

    # Mean-shift map
    kappaE_map -= np.nanmean(kappaE_map) # kappa maps reconstructed with KS/KSP are already set to a null mean
    
    # Define map coordinates
    ra, dec = np.meshgrid(np.arange(len(kappaE_map))*pix_res, np.arange(len(kappaE_map))*pix_res)
    
    # Flattenise map and coordinates        
    k_cat   = kappaE_map.flatten()
    ra_cat  = ra.flatten()
    dec_cat = dec.flatten()
    
    # Remove nan pixels
    sel     = np.where(np.logical_not(np.isnan(k_cat)))
    k_cat   = k_cat[sel]
    ra_cat  = ra_cat[sel]
    dec_cat = dec_cat[sel]
    
    # Compute kappa-2pcf using Paper I settings
    twopt_min  = pix_res
    twopt_max  = pix_res * 512 * np.sqrt(2) # Fix maximum separation to the DUSTGRAIN map diagonal in order to have measurements at the same scales across all cosmologies of DUSTGRAIN and SLICS
    twopt_bins = 10 #25
    
    treecormap = treecorr.Catalog(ra=ra_cat, dec=dec_cat, k=k_cat, ra_units='arcmin', dec_units='arcmin')
    kk = treecorr.KKCorrelation(min_sep=twopt_min, max_sep=twopt_max, nbins=twopt_bins, sep_units='arcmin')
    kk.process(treecormap)
    
    kk_r   = np.exp(kk.meanlogr) # Note: meanlogr represents the mean distance of all the galaxy pairs falling in the given bin (it is an alternative to the simple bin mid-point rnom considered in the mean plots)
    kk_xi  = kk.xi
    kk_sig = np.sqrt(kk.varxi)
    
    # Plot kappa-2pcf using meanlogr (for debugging) for first two realisations
    if los<2:
             
        plt.xscale('log')
        plt.yscale('log', nonposy='clip')
        plt.xlabel(r'$\theta$ (arcmin)', fontsize=18)
        plt.ylabel(r'$\xi$', fontsize=18)
        plt.xlim( [twopt_min, twopt_max] )
        
        if any(kk_xi>0):
            plt.errorbar(kk_r[kk_xi>0], kk_xi[kk_xi>0], yerr=kk_sig[kk_xi>0], color='blue', ls='None', marker='o')
        if any(kk_xi<0):
            plt.errorbar(kk_r[kk_xi<0], -kk_xi[kk_xi<0], yerr=kk_sig[kk_xi<0], color='white', markeredgecolor='blue', ls='None', marker='o')
        
        plt.title(f'LOS_{str(los).zfill(3)}_z-bin {str(z_bin_index).zfill(2)}')   
        plt.savefig(plot_outpath + f'{kappa_map_name}_{statistic}.png', dpi=300)  
        plt.clf()
        plt.close('all')
  
    # Write kappa-2pcf DV to text
    #kk.write(DV_outpath + f'{kappa_map_name}_{statistic}.txt')


##----------------------------------------------------------------------------## 


def write_map_to_fits(map_data, deg_side, fits_outpath, fits_fname):
    
    # Make temporary directory for the fits file if it does not exist 
    os.makedirs(fits_outpath, exist_ok=True)
    
    # Write fits file
    fits.writeto(fits_outpath + fits_fname + '.fits', data=map_data, overwrite=True)
    
    # Add 'ANGLE' keyword header for the 'lenstools' instance
    with fits.open(fits_outpath + fits_fname + '.fits', mode="update") as hdul:
        
        # get the primary header
        header = hdul[0].header
        
        # add the 'ANGLE' keyword 
        header["ANGLE"] = deg_side
        
        # save the changes to the file
        hdul.flush()


##----------------------------------------------------------------------------##     


# Kappa_moments Function # 
def measure_kappa_moments_oneLOS(los):
    
    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_moments'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    os.makedirs(DV_outpath, exist_ok=True)
    os.makedirs(plot_outpath, exist_ok=True)
    
    # Load kappa map
    # kappa_map_inpath = inpath + f'convergence_maps/{nbody_pipeline}_{cosmo}/{KS_inversion_type}/'
    # kappa_map_name = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}'
    # kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 
    # old paths, testing new paths for DUSTGRAIN f(R)

    # Load kappa map
    # added models different from LCDM through {cosmo}_zs= [...] - MODIFIED
    kappa_map_inpath = inpath + f'{cosmo}_zs={str(z_bin_index)}_1_kappaBapp/DUSTGRAIN-pathfinder/WeakLensing/downscaled/{cosmo}/output_relative/{str(los).zfill(2)}/'
    kappa_map_name = f'1_kappaBApp'
    kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 


    # Define filter type  
    kappa_moments_filter = 'tophat'

    # Mean-shift map
    kappaE_map -= np.nanmean(kappaE_map) # kappa maps reconstructed with KS/KSP are already set to a null mean    
    
    # Declare moments
    k2     = np.zeros(len(scales_px))
    k3     = np.zeros(len(scales_px))
    k4     = np.zeros(len(scales_px))
    S3     = np.zeros(len(scales_px))
    S4     = np.zeros(len(scales_px))   

    # Compute kappa_moments for all scales
    for i, (scale, scale_px) in enumerate(zip(scales, scales_px)):

        kappa_moments_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}' + f'_filter_{kappa_moments_filter}_scale_{scale}_{statistic}'
        
        # Smooth map with a Top-hat filter
        kernel = Tophat2DKernel(scale_px) # pass scale in pixels    
        kappaE_map_smoothed = convolve_fft(kappaE_map, kernel, normalize_kernel=True, nan_treatment='interpolate')
        
        # Remove pixels whom smothing radius intersects (or coincides with) the map boundary (aka remove a stripe of width int(np.ceil(scale_px)) from the map border)
        border_width = int(np.ceil(scale_px)) # round up to the nearest integer
        assert(border_width > 0 and kappaE_map_smoothed.shape[0] > 2 * border_width) # check whether the border cutting is applicable to the map
        kappaE_map_smoothed = kappaE_map_smoothed[border_width:-border_width, border_width:-border_width]     

        k2[i]     = np.nanmean(kappaE_map_smoothed**2)
        k3[i]     = np.nanmean(kappaE_map_smoothed**3)
        k4[i]     = np.nanmean(kappaE_map_smoothed**4)
        S3[i]     = k3[i] / (k2[i])**(3./2.)
        S4[i]     = k4[i] / (k2[i])**2. 

    # Write kappa_moments DV to text
    #np.savetxt(DV_outpath + f'{kappa_map_name}_filter_{kappa_moments_filter}_scales_{scales_px}_{statistic}.txt', np.column_stack([scales, k2, k3, k4, S3, S4]))
    np.savetxt(DV_outpath + kappa_moments_fname + '.txt', np.column_stack([scales, k2, k3, k4, S3, S4]))


##----------------------------------------------------------------------------##  


# Kappa-1PDF Function # 
def measure_kappa_1pdf_oneLOS(los):
    
    # Load kappa map
    # kappa_map_inpath = inpath + f'convergence_maps/{nbody_pipeline}_{cosmo}/{KS_inversion_type}/'
    # kappa_map_name = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}'
    # kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 
    # old paths, testing new paths for DUSTGRAIN f(R)

    # Load kappa map
    # added models different from LCDM through {cosmo}_zs= [...] - MODIFIED
    kappa_map_inpath = inpath + f'{cosmo}_zs={str(z_bin_index)}_1_kappaBapp/DUSTGRAIN-pathfinder/WeakLensing/downscaled/{cosmo}/output_relative/{str(los).zfill(2)}/'
    kappa_map_name = f'1_kappaBApp'
    kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 

    # Define filter type  
    kpdf_filter = 'tophat'

    # Mean-shift map
    kappaE_map -= np.nanmean(kappaE_map) # kappa maps reconstructed with KS/KSP are already set to a null mean    

    # Compute 1-pdf of kappa maps for all scales
    for scale, scale_px in zip(scales, scales_px):
        
        # Smooth map with a Top-hat filter
        kernel = Tophat2DKernel(scale_px) # pass scale in pixels    
        kappaE_map_smoothed = convolve_fft(kappaE_map, kernel, normalize_kernel=True, nan_treatment='interpolate')
        
        # Remove pixels whom smothing radius intersects (or coincides with) the map boundary (aka remove a stripe of width int(np.ceil(scale_px)) from the map border)
        border_width = int(np.ceil(scale_px)) # round up to the nearest integer
        assert(border_width > 0 and kappaE_map_smoothed.shape[0] > 2 * border_width) # check whether the border cutting is applicable to the map
        kappaE_map_smoothed = kappaE_map_smoothed[border_width:-border_width, border_width:-border_width]        
        
        # Save kappa map in a temporary fits file to allow 1-pdf computation
        write_map_to_fits(kappaE_map_smoothed, deg_side = deg_side, fits_outpath = outpath + 'tmp_kappa_map/', fits_fname = kappa_map_name + f'_filter_{kpdf_filter}_scale_{scale}')         
        
        # Measure 1-pdf from kappa map
        measure_1pdf_from_kappa(los, map_path = outpath + 'tmp_kappa_map/', map_fname = kappa_map_name + f'_filter_{kpdf_filter}_scale_{scale}', scale=scale)
     

def measure_1pdf_from_kappa(los, map_path, map_fname, scale):
    
    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_1pdf'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    os.makedirs(DV_outpath, exist_ok=True)
    os.makedirs(plot_outpath, exist_ok=True)
    
    # Measure kappa-1pdf using Paper I settings
    ''' Note: conv_map.pdf histograms the kappa values of the map using the given bin edges and returning either the mid-points of each bin and the histogrammed-pdf (ranging in [0,1] whether norm=True or [0,100] whether norm=False) '''  
    conv_map = ConvergenceMap.load(map_path + map_fname + '.fits', masked=False)
    
    # Find bin-edges from number of bins (mid-points) of HOWLS-KP1
    midpoints = np.linspace(-0.1, 0.1, endpoint=True, num=201)
    step = abs(midpoints[1]-midpoints[0])
    bin_edges = np.linspace(-0.1-step/2, 0.1+step/2, endpoint=True, num=202)

    kappa_smoothed_bin_edges = bin_edges # Note: for nbin, one has nbin+1 bin-edges!
    kappa_smoothed_midpoints, pdf = conv_map.pdf(kappa_smoothed_bin_edges, norm=False) 
    
    # Plot kappa 1-PDF (reporting either S/N and \kappa in the x-axis) for the first two realisations
    if los<2:
    
        fig_pdf, ax_pdf = plt.subplots()
        
        ax_pdf.plot(kappa_smoothed_midpoints, pdf, label = f"{map_fname}_{statistic}", color='blue')
        ax_pdf.set_xlabel(r"$\kappa$")
        ax_pdf.set_ylabel(r"$\kappa-1PDF$")
        ax_pdf.legend(loc="upper right", prop={'size': 5})
        
        fig_pdf.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
        fig_pdf.savefig(plot_outpath + f'{map_fname}_{statistic}.png', dpi=300)
        plt.close('all')
    
    # Write kappa 1-PDF DV to text
    np.savetxt(DV_outpath + f'{map_fname}_{statistic}.txt', np.column_stack([kappa_smoothed_midpoints, pdf]))
    
    # Remove the already used temporary fits file
    if os.path.exists(map_path + map_fname + '.fits'):
        # remove the file
        os.remove(map_path + map_fname + '.fits')


##----------------------------------------------------------------------------##     


# Apm-Peaks Function #
def measure_kappa_peaks_oneLOS(los):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_peaks'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    os.makedirs(DV_outpath, exist_ok=True)
    os.makedirs(plot_outpath, exist_ok=True)

    # Define filter type
    peaks_filter = 'gaussian'    

    # Load kappa map
    # kappa_map_inpath = inpath + f'convergence_maps/{nbody_pipeline}_{cosmo}/{KS_inversion_type}/'
    # kappa_map_name = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}'
    # kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention)  
    # old paths, testing new paths for DUSTGRAIN f(R)

    # Load kappa map
    # added models different from LCDM through {cosmo}_zs= [...] - MODIFIED
    kappa_map_inpath = inpath + f'{cosmo}_zs={str(z_bin_index)}_1_kappaBapp/DUSTGRAIN-pathfinder/WeakLensing/downscaled/{cosmo}/output_relative/{str(los).zfill(2)}/'
    kappa_map_name = f'1_kappaBApp'
    kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 

    # Define map properties
    npix = kappaE_map.shape[0] # pixel number per side 
    pix_res = map_res # arcmins

    # Compute peaks of kappa map for all scales
    for scale, scale_px in zip(scales, scales_px):

        kappa_peaks_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}' + f'_filter_{peaks_filter}_scale_{scale}_{statistic}'
        
        # Smooth map with a Gaussian filter
        kernel = Gaussian2DKernel(scale_px) # pass scale in pixels    
        kappaE_map_smoothed = convolve_fft(kappaE_map, kernel, normalize_kernel=True, nan_treatment='interpolate')
       
        # Detect and count peaks on the kappa E-mode with a S/N > min_peak_threshold wrt 8 neighboor pixels using Paper I settings (no peaks at borders)
        kappaE_map_smoothed /= np.std(kappaE_map_smoothed, ddof=0)
        min_peak_threshold = -5.
        max_peak_threshold = 10.
        x,y,h = find_peaks2d(kappaE_map_smoothed, threshold=min_peak_threshold, include_border=False, mask=None)
        x,y,h = x[h < + max_peak_threshold], y[h < + max_peak_threshold], h[h < + max_peak_threshold] # esclude peaks with S/N > +10 as in Paper I
        
        # Compute theoretically midpoints starting from HOWLS KP1 bin_edges
        nbin = 150 # aka number of mid-points
        bin_edges = np.linspace(min_peak_threshold, max_peak_threshold, endpoint=True, num=nbin+1)
        bin_midpoints = 0.5 * ( bin_edges[1:] + bin_edges[:-1] )
        
        # Count peaks with a S/N > min_peak_threshold in 'nbins' bins
        N_peaks, bin_edges = np.histogram(h, bins=bin_edges) # Equivalently: N_peaks, bin_edges = peak_histogram(kappaE_map_smoothed, bins=bin_edges, mask=None)

        # Find fraction of peaks found at borders
        N_border_peaks = (np.count_nonzero(y==0) + np.count_nonzero(x==0) + np.count_nonzero(y==npix) + np.count_nonzero(x==npix)) 

        # Plot kappa apm peaks for each scale for the first two realisations
        if los<2:
        
            fig_peaks, ax_peaks = plt.subplots() 
            
            ax_peaks.plot(bin_midpoints, N_peaks, label= f'{kappa_map_name}' + f'_filter_{peaks_filter}_scale_{scale}_{statistic}' + '\n' +
                               f'Peaks found at map borders {N_border_peaks}')
            ax_peaks.set_xlabel(r"$\nu$")
            ax_peaks.set_ylabel(r"$\kappa$ Peak Counts")
            ax_peaks.legend(loc="upper right", prop={'size': 5})
            
            fig_peaks.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
            fig_peaks.savefig(plot_outpath + kappa_map_name + f'_filter_{peaks_filter}_scale_{scale}_{statistic}.png', dpi=300)
            plt.close('all')
        
        # Write kappa apm peaks DV to text
        # np.savetxt(DV_outpath + kappa_map_name + f'_filter_{peaks_filter}_scale_{scale}_{statistic}.txt', np.column_stack([bin_midpoints, N_peaks]))
        np.savetxt(DV_outpath + kappa_peaks_fname + '.txt', np.column_stack([bin_midpoints, N_peaks]))
        

##----------------------------------------------------------------------------##  


# Kappa-MFs Function #    
def measure_kappa_mfs_oneLOS(los):    
    
    # Define intermediate functions
    def step(x):
        return 1 * (x > 0)
    
    def delta(x, delta_nu):
        return (step(x+delta_nu/2.)-step(x-delta_nu/2.))/delta_nu
    
    def mf1(k, nu):
        return float(np.sum(step(k-nu)))/float(k.size)
    
    def mf2(k, nu, delta_nu, kx, ky):
        return float(np.sum(delta(k-nu, delta_nu)*(np.sqrt(kx**2+ky**2))))/float(4.*k.size)
    
    def mf3(k, nu, delta_nu, kx, ky, kxy, kxx, kyy):
        return float(np.sum(delta(k-nu, delta_nu)*(2.*kx*ky*kxy-kx**2*kyy-ky**2*kxx)/(kx**2+ky**2)))/float(2.*np.pi*k.size)
    
    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_mfs'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    os.makedirs(DV_outpath, exist_ok=True)
    os.makedirs(plot_outpath, exist_ok=True)
    
    # Define filter type
    kmfs_filter = 'gaussian'
    
    # Load kappa map
    # kappa_map_inpath = inpath + f'convergence_maps/{nbody_pipeline}_{cosmo}/{KS_inversion_type}/'
    # kappa_map_name = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}'
    # kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention)  
    # old paths, testing new paths for DUSTGRAIN f(R)

    # Load kappa map
    # added models different from LCDM through {cosmo}_zs= [...] - MODIFIED
    kappa_map_inpath = inpath + f'{cosmo}_zs={str(z_bin_index)}_1_kappaBapp/DUSTGRAIN-pathfinder/WeakLensing/downscaled/{cosmo}/output_relative/{str(los).zfill(2)}/'
    kappa_map_name = f'1_kappaBApp'
    kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 

    # Define map properties
    npix = kappaE_map.shape[0] # pixel number per side 
    pix_res = map_res # arcmins

    # Define thresholds for measurements
    delta_nu =  0.01
    nu_min   = -5
    nu_max   =  5
    nu = np.arange(nu_min, nu_max+delta_nu, delta_nu)   

    # Mean-shift map
    kappaE_map -= np.nanmean(kappaE_map) # kappa maps reconstructed with KS/KSP are already set to a null mean    

    # Compute MFs of kappa map for all scales
    for scale, scale_px in zip(scales, scales_px):

        kappa_mfs_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}' + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}'
        
        # Initialize MFs
        v0, v1, v2 = np.zeros(len(nu)), np.zeros(len(nu)), np.zeros(len(nu))
        
        # Smooth map with a Gaussian filter
        kernel = Gaussian2DKernel(scale_px) # pass scale in pixels    
        kappaE_map_smoothed = convolve_fft(kappaE_map, kernel, normalize_kernel=True, nan_treatment='interpolate')
        
        # Normalize map
        kappaE_map_smoothed /= np.std(kappaE_map_smoothed, ddof=1)    
        
        # Compute map gradients
        kx,  ky  = np.gradient(kappaE_map_smoothed, edge_order=2)
        kxx, kxy = np.gradient(kx, edge_order=2)
        kyx, kyy = np.gradient(ky, edge_order=2)      
        
        # Measure MFs        
        for i in range(len(nu)):
        
            v0[i] = mf1(kappaE_map_smoothed, nu[i])
            v1[i] = mf2(kappaE_map_smoothed, nu[i], delta_nu, kx, ky)
            v2[i] = mf3(kappaE_map_smoothed, nu[i], delta_nu, kx, ky, kxy, kxx, kyy)

        # Plot kappa MFs for each scale for the first two realisations
        if los<2:
        
            fig_mfs, ax_mfs = plt.subplots(1, 3, squeeze=False, figsize=(20, 5)) 
            
            ax_mfs[0,0].plot(nu, v0, label= r'$V_{0}(\nu)$' + '\t' + f'{kappa_map_name}' + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}', color='blue')
            ax_mfs[0,0].set_xlabel(r"$\nu$")
            ax_mfs[0,0].set_ylabel(r"$V_{0}(\nu)$")
            ax_mfs[0,0].legend(loc="upper right", prop={'size': 5})
    
            ax_mfs[0,1].plot(nu, v1, label= r'$V_{1}(\nu)$' + '\t' + f'{kappa_map_name}' + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}', color='darkgreen')
            ax_mfs[0,1].set_xlabel(r"$\nu$")
            ax_mfs[0,1].set_ylabel(r"$V_{1}(\nu)$")
            ax_mfs[0,1].legend(loc="upper right", prop={'size': 5})
    
            ax_mfs[0,2].plot(nu, v2, label= r'$V_{2}(\nu)$' + '\t' + f'{kappa_map_name}' + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}', color='darkorange')
            ax_mfs[0,2].set_xlabel(r"$\nu$")
            ax_mfs[0,2].set_ylabel(r"$V_{2}(\nu)$")
            ax_mfs[0,2].legend(loc="upper right", prop={'size': 5})        
            
            fig_mfs.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
            fig_mfs.savefig(plot_outpath + kappa_map_name + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}.png', dpi=300)
            plt.close('all')
        
        # Write kappa MFs DV to text
        #np.savetxt(DV_outpath + kappa_map_name + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}.txt', np.column_stack([nu, v0, v1, v2]))
        np.savetxt(DV_outpath + kappa_mfs_fname + '.txt', np.column_stack([nu, v0, v1, v2]))


##----------------------------------------------------------------------------##  


# Kappa-BNs Function # 
def measure_kappa_bns_oneLOS(los):

        # Make output folders for the considered statistics if they do not exist
        statistic = 'kappa_bns'
        DV_outpath = outpath + 'DVs/' + statistic + '/'
        plot_outpath = outpath + 'plots/' + statistic + '/'
        os.makedirs(DV_outpath, exist_ok=True)
        os.makedirs(plot_outpath, exist_ok=True)
    
        # Load kappa map
        # added models different from LCDM through {cosmo}_zs= [...] - MODIFIED
        kappa_map_inpath = inpath + f'{cosmo}_zs={str(z_bin_index)}_1_kappaBapp/DUSTGRAIN-pathfinder/WeakLensing/downscaled/{cosmo}/output_relative/{str(los).zfill(2)}/'
        kappa_map_name = f'1_kappaBApp'
        kappaE_map = fits.getdata(kappa_map_inpath + kappa_map_name + '.fits') # (SLICS/DUSTGRAIN: TRANSPOSE kappa maps to match Matricial convention) 

        # Define filter type
        kbns_filter = 'gaussian'

        # Define map properties
        npix = kappaE_map.shape[0] # pixel number per side 
        pix_res = map_res # arcmins
    
        # Define thresholds for measurements
        delta_nu =  0.5#0.01
        nu_min   = -5
        nu_max   =  5
        nu = np.arange(nu_min, nu_max+delta_nu, delta_nu)   
    
        # Mean-shift map
        kappaE_map -= np.nanmean(kappaE_map) # kappa maps reconstructed with KS/KSP are already set to a null mean    
        
        # Compute BNs of kappa map for all scales
        for scale, scale_px in zip(scales, scales_px):
            
            kappa_bns_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}' + f'_filter_{kbns_filter}_scale_{scale}_{statistic}'
          
            # Initialize BNs
            betti0 = np.zeros(len(nu))
            betti1 = np.zeros(len(nu))
            
            # Smooth map with a Gaussian filter
            kernel = Gaussian2DKernel(scale_px) # pass scale in pixels    
            kappaE_map_smoothed = convolve_fft(kappaE_map, kernel, normalize_kernel=True, nan_treatment='interpolate')
            
            # Normalize map
            kappaE_map_smoothed /= np.std(kappaE_map_smoothed, ddof=1)   
            
            for i in range(len(nu)):
            
                sel = np.where(kappaE_map_smoothed>nu[i])
                if len(sel[0])>0:
                    ex_set = np.zeros(kappaE_map_smoothed.shape)
                    ex_set[kappaE_map_smoothed>nu[i]] = 1
                    
                    labels_nu, n_labels_nu = label(ex_set, background=0, connectivity = 2, return_num=True)
                    
                    betti0[i] = n_labels_nu
    
                sel = np.where(kappaE_map_smoothed<=nu[i])
                if len(sel[0])>0:
                    ex_set_holes = np.ones(kappaE_map_smoothed.shape)
                    ex_set_holes[kappaE_map_smoothed>nu[i]] = 0
    
                    labels_nu_holes, n_labels_nu_holes = label(ex_set_holes, background=0, connectivity = 1, return_num=True)
    
                    regions = regionprops(labels_nu_holes)
                    tmp=0
                    for j in range(n_labels_nu_holes):
                        border = list(regions[j].coords.flatten())
                        if npix-1 not in border and 0 not in border:
                            tmp = tmp + 1
                            
                    betti1[i] = tmp

            # Plot kappa BNs for each scale for the first two realisations
            if los<2:
            
                fig_bns, ax_bns = plt.subplots(1, 2, squeeze=False, figsize=(20, 5)) 
                
                ax_bns[0,0].plot(nu, betti0, label= r'$\beta_{0}(\nu)$' + '\t' + f'{kappa_map_name}' + f'_filter_{kbns_filter}_scale_{scale}_{statistic}', color='blue')
                ax_bns[0,0].set_xlabel(r"$\nu$")
                ax_bns[0,0].set_ylabel(r"$\beta_{0}(\nu)$")
                ax_bns[0,0].legend(loc="upper right", prop={'size': 5})
        
                ax_bns[0,1].plot(nu, betti1, label= r'$\beta_{1}(\nu)$' + '\t' + f'{kappa_map_name}' + f'_filter_{kbns_filter}_scale_{scale}_{statistic}', color='darkgreen')
                ax_bns[0,1].set_xlabel(r"$\nu$")
                ax_bns[0,1].set_ylabel(r"$\beta_{1}(\nu)$")
                ax_bns[0,1].legend(loc="upper right", prop={'size': 5})
                    
                fig_bns.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
                fig_bns.savefig(plot_outpath + kappa_bns_fname + '.png', dpi=300)
                plt.close('all')
        
            # Write kappa BNs DV to text
            np.savetxt(DV_outpath + kappa_bns_fname + '.txt', np.column_stack([nu, betti0, betti1]))


##----------------------------------------------------------------------------## 


# (AUTO-) Kappa-2PCF Mean Calculator Function #
def compute_and_plot_mean_kappa_2pcf(los_list):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_2pcf'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    mean_DV_outpath = outpath + 'mean_DVs/' + statistic + '/'
    mean_plot_outpath = outpath + 'mean_plots/' + statistic + '/'
    os.makedirs(mean_DV_outpath, exist_ok=True)
    os.makedirs(mean_plot_outpath, exist_ok=True)

    mean_kk_xi, mean_kk_sig = 0., 0.
    
    # Compute mean auto-kappa-2pcf
    for los in los_list:
        
        kappa_2pcf_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}_{statistic}'
        kk_r, kk_xi, kk_sig = np.genfromtxt(DV_outpath + kappa_2pcf_fname + '.txt', usecols=(0, 3, 4), unpack=True)
        
        mean_kk_xi += kk_xi / len(los_list)
        mean_kk_sig += kk_sig**2 # propagate y-errors on the mean formula
    
    mean_kk_sig = np.sqrt(mean_kk_sig) / len(los_list)
         
    # Define output file name
    mean_kappa_2pcf_fname = f'{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}_{statistic}'
    
    # Plot mean auto-kappa-2pcf using rnom
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=18)
    plt.ylabel(r'$\xi$', fontsize=18)
    plt.xlim( [kk_r.min(), kk_r.max()] )
    
    if any(mean_kk_xi>0):
        plt.errorbar(kk_r[mean_kk_xi>0], mean_kk_xi[mean_kk_xi>0], yerr=mean_kk_sig[mean_kk_xi>0], color='blue', ls='None', marker='o')
    if any(kk_xi<0):
        plt.errorbar(kk_r[mean_kk_xi<0], -mean_kk_xi[mean_kk_xi<0], yerr=mean_kk_sig[mean_kk_xi<0], color='white', markeredgecolor='blue', ls='None', marker='o')
    
    plt.title(f'z-bin {str(z_bin_index).zfill(2)}')
    plt.savefig(mean_plot_outpath + mean_kappa_2pcf_fname + '.png', dpi=300)  
    plt.clf()
    plt.close('all')

    # Write mean auto-kappa-2pcf DVs to text
    np.savetxt(mean_DV_outpath + mean_kappa_2pcf_fname + '.txt', np.column_stack([kk_r, mean_kk_xi, mean_kk_sig]))


##----------------------------------------------------------------------------##  


# Kappa-moments Mean Calculator Function #
def compute_and_plot_mean_kappa_moments(los_list):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_moments'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    mean_DV_outpath = outpath + 'mean_DVs/' + statistic + '/'
    mean_plot_outpath = outpath + 'mean_plots/' + statistic + '/'
    os.makedirs(mean_DV_outpath, exist_ok=True)
    os.makedirs(mean_plot_outpath, exist_ok=True)
    
    kappa_moments_filter = 'tophat'
    
    # Compute mean kappa_moments for all filter scales 
    for los in los_list:
        
        kappa_moments_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}' + f'_filter_{kappa_moments_filter}_scales_{scales_px}_{statistic}'
        moments_scales, k2, k3, k4, S3, S4 = np.genfromtxt(DV_outpath + kappa_moments_fname + '.txt', unpack=True)
        
        # Compute error on the mean
        if los == los_list[0]: k2_mean, k3_mean, k4_mean, S3_mean, S4_mean = k2 * 1, k3 * 1, k4 * 1, S3 * 1, S4 * 1 # Avoid shallow copy
        else:                  k2_mean, k3_mean, k4_mean, S3_mean, S4_mean = np.column_stack([k2_mean, k2]), np.column_stack([k3_mean, k3]), np.column_stack([k4_mean, k4]), np.column_stack([S3_mean, S3]), np.column_stack([S4_mean, S4])
        
    # Compute error on the mean and mean
    k2_sigma, k3_sigma, k4_sigma, S3_sigma, S4_sigma = np.std(k2_mean, axis=1, ddof=1), np.std(k3_mean, axis=1, ddof=1), np.std(k4_mean, axis=1, ddof=1), np.std(S3_mean, axis=1, ddof=1), np.std(S4_mean, axis=1, ddof=1) # Compute sigma over all realisations of the same DV-element
    k2_mean, k3_mean, k4_mean, S3_mean, S4_mean  = np.mean(k2_mean, axis=1), np.mean(k3_mean, axis=1), np.mean(k4_mean, axis=1), np.mean(S3_mean, axis=1), np.mean(S4_mean, axis=1) # Compute mean over all realisations of the same DV-element
    
    # Write mean DVs to text
    mean_kappa_moments_fname = f'{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}' + f'_filter_{kappa_moments_filter}_scales_{scales_px}_{statistic}'
    np.savetxt(mean_DV_outpath + mean_kappa_moments_fname + '.txt', np.column_stack([moments_scales, k2_mean, k2_sigma, k3_mean, k3_sigma, k4_mean, k4_sigma, S3_mean, S3_sigma, S4_mean, S4_sigma]))
    
    # Plot mean kappa_moments
    fig, ax = plt.subplots(1, 5, squeeze=False, figsize=(15,5))
    
    ax[0,0].plot(moments_scales, k2_mean)
    ax[0,0].errorbar(moments_scales, k2_mean, yerr=k2_sigma)
    ax[0,0].set_xlabel(r"$\theta(')$")
    ax[0,0].set_ylabel(r"$\left<\kappa^{2}\right>$")
    
    ax[0,1].plot(moments_scales, k3_mean)
    ax[0,1].errorbar(moments_scales, k3_mean, yerr=k3_sigma)
    ax[0,1].set_xlabel(r"$\theta(')$")
    ax[0,1].set_ylabel(r"$\left<\kappa^{3}\right>$")

    ax[0,2].plot(moments_scales, k4_mean)
    ax[0,2].errorbar(moments_scales, k4_mean, yerr=k4_sigma)
    ax[0,2].set_xlabel(r"$\theta(')$")
    ax[0,2].set_ylabel(r"$\left<\kappa^{4}\right>$")

    ax[0,3].plot(moments_scales, S3_mean)
    ax[0,3].errorbar(moments_scales, S3_mean, yerr=S3_sigma)
    ax[0,3].set_xlabel(r"$\theta(')$")
    ax[0,3].set_ylabel(r"$\left<\kappa^{3}\right> / \left<\kappa^{2}\right>^{3/2}$")

    ax[0,4].plot(moments_scales, S4_mean)
    ax[0,4].errorbar(moments_scales, S4_mean, yerr=S4_sigma)
    ax[0,4].set_xlabel(r"$\theta(')$")
    ax[0,4].set_ylabel(r"$\left<\kappa^{4}\right>  / \left<\kappa^{2}\right>^{2}$")
    
    fig.suptitle(f'{nbody_pipeline}_{cosmo} z-bin {str(z_bin_index).zfill(2)} Filter: {kappa_moments_filter} Scales: {np.round(scales, 2)} arcmin')
    fig.tight_layout()
    fig.savefig(mean_plot_outpath + mean_kappa_moments_fname + '.png', dpi=300)
    plt.close('all')


##----------------------------------------------------------------------------##   


# Kappa-1PDF Mean Calculator Function #
def compute_and_plot_mean_kappa_1pdf(los_list):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_1pdf'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    mean_DV_outpath = outpath + 'mean_DVs/' + statistic + '/'
    mean_plot_outpath = outpath + 'mean_plots/' + statistic + '/'
    os.makedirs(mean_DV_outpath, exist_ok=True)
    os.makedirs(mean_plot_outpath, exist_ok=True)

    # Define filter type    
    kpdf_filter = 'tophat'
    
    # Compute mean kappa-1pdf for all filter scales
    for scale, scale_px in zip(scales, scales_px):
        
        pdf_mean = 0.
        
        for los in los_list:
            
            kappa_1pdf_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}' + f'_filter_{kpdf_filter}_scale_{scale}_{statistic}'
            kappa_smoothed_midpoints, pdf = np.genfromtxt(DV_outpath + kappa_1pdf_fname + '.txt', unpack=True)
            
            # Compute error on the mean
            if los == los_list[0]: data = pdf * 1 # Avoid shallow copy
            else:                  data = np.column_stack([data, pdf])
            
        # Finalize error on the mean
        sigma_pdf = np.std(data, axis=1, ddof=1) # Compute sigma over all realisations of the same DV-element
        pdf_mean  = np.mean(data, axis=1) # Compute mean over all realisations of the same DV-element

        # Define output file name
        mean_kappa_1pdf_fname = f'{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}' + f'_filter_{kpdf_filter}_scale_{scale}_{statistic}'
        
        # Plot mean k-1pdf for the given scale
        fig, ax = plt.subplots()
        
        ax.plot(kappa_smoothed_midpoints, pdf_mean, label = f'{mean_kappa_1pdf_fname}', color='blue')
        ax.errorbar(kappa_smoothed_midpoints, pdf_mean, yerr=sigma_pdf, color='blue')
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\kappa-1PDF$")
        ax.legend(loc="upper right", prop={'size': 5})
        
        fig.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
        fig.savefig(mean_plot_outpath + mean_kappa_1pdf_fname + '.png', dpi=300)
        plt.close('all')

        # Write mean k-1pdf for the given scale to text
        np.savetxt(mean_DV_outpath + mean_kappa_1pdf_fname + '.txt', np.column_stack([kappa_smoothed_midpoints, pdf_mean, sigma_pdf]))


##----------------------------------------------------------------------------##             


# Apm-Peaks Mean Calculator Function #
def compute_and_plot_mean_kappa_peaks(los_list):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_peaks'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    mean_DV_outpath = outpath + 'mean_DVs/' + statistic + '/'
    mean_plot_outpath = outpath + 'mean_plots/' + statistic + '/'
    os.makedirs(mean_DV_outpath, exist_ok=True)
    os.makedirs(mean_plot_outpath, exist_ok=True)
    
    # Define filter type
    peaks_filter = 'gaussian'
    
    # Compute mean kappa_peaks for all filter scales
    for scale, scale_px in zip(scales, scales_px):
        
        N_peaks_mean = 0.
        
        for los in los_list:
            
            kappa_peaks_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}_filter_{peaks_filter}_scale_{scale}_{statistic}'
            nu, N_peaks = np.genfromtxt(DV_outpath + kappa_peaks_fname + '.txt', unpack=True)
            
            # Compute error on the mean
            if los == los_list[0]: data = N_peaks * 1 # Avoid shallow copy
            else:                  data = np.column_stack([data, N_peaks])
            
        # Finalize error on the mean
        sigma_N_peaks = np.std(data, axis=1, ddof=1) # Compute sigma over all realisations of the same DV-element
        N_peaks_mean  = np.mean(data, axis=1) # Compute mean over all realisations of the same DV-element
        
        # Define output file name
        mean_kappa_peaks_fname = f'{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}_filter_{peaks_filter}_scale_{scale}_{statistic}'
        
        # Plot mean kappa_peaks for the given scale
        fig, ax = plt.subplots()
        
        ax.plot(nu, N_peaks_mean, label = f'{mean_kappa_peaks_fname}', color='blue')
        ax.errorbar(nu, N_peaks_mean, yerr=sigma_N_peaks, color='blue')
        ax.set_xlabel(r"$\nu$")
        ax.set_ylabel(r"$M_{ap}(\kappa)$ Peak Counts")
        ax.legend(loc="upper right", prop={'size': 5})
        
        fig.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
        fig.savefig(mean_plot_outpath + mean_kappa_peaks_fname + '.png', dpi=300)
        plt.close('all')

        # Write mean kappa_peaks for the given scale to text
        np.savetxt(mean_DV_outpath + mean_kappa_peaks_fname + '.txt', np.column_stack([nu, N_peaks_mean, sigma_N_peaks]))


##----------------------------------------------------------------------------##


# Kappa-MFs Mean Calculator Function #
def compute_and_plot_mean_kappa_mfs(los_list):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_mfs'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    mean_DV_outpath = outpath + 'mean_DVs/' + statistic + '/'
    mean_plot_outpath = outpath + 'mean_plots/' + statistic + '/'
    os.makedirs(mean_DV_outpath, exist_ok=True)
    os.makedirs(mean_plot_outpath, exist_ok=True)
    
    # Define filter type
    kmfs_filter = 'gaussian'
    
    # Compute mean kappa MFs for all filter scales
    for scale, scale_px in zip(scales, scales_px):
        
        # Initialize mean MFs
        v0_mean, v1_mean, v2_mean = 0., 0., 0.
        
        for los in los_list:
            
            kappa_mfs_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}' + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}'
            nu, v0, v1, v2 = np.genfromtxt(DV_outpath + kappa_mfs_fname + '.txt', unpack=True)
            
            # Compute error on the mean
            if los == los_list[0]: data_v0, data_v1, data_v2 = v0 * 1, v1 * 1, v2 * 1 # Avoid shallow copy
            else:                  data_v0, data_v1, data_v2 = np.column_stack([data_v0, v0]), np.column_stack([data_v1, v1]), np.column_stack([data_v2, v2])
            
        # Finalize error on the mean
        sigma_v0, sigma_v1, sigma_v2 = np.std(data_v0, axis=1, ddof=1), np.std(data_v1, axis=1, ddof=1), np.std(data_v2, axis=1, ddof=1) # Compute sigma over all realisations of the same DV-element
        v0_mean,  v1_mean,  v2_mean  = np.mean(data_v0, axis=1), np.mean(data_v1, axis=1), np.mean(data_v2, axis=1) # Compute mean over all realisations of the same DV-element
        
        # Define output file name
        mean_kappa_mfs_fname = f'{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}' + f'_filter_{kmfs_filter}_scale_{scale}_{statistic}'
        
        # Plot mean kappa MFs for the given scale
        fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(20, 5))
       
        ax[0,0].plot(nu, v0_mean, label = r'$V_{0}(\nu)$' + '\t' + f'{mean_kappa_mfs_fname}', color='blue')
        ax[0,0].errorbar(nu, v0_mean, yerr=sigma_v0, color='blue')
        ax[0,0].set_xlabel(r"$\nu$")
        ax[0,0].set_ylabel(r"$V_{0}(\nu)$")
        ax[0,0].legend(loc="upper right", prop={'size': 5})

        ax[0,1].plot(nu, v1_mean, label = r'$V_{1}(\nu)$' + '\t' + f'{mean_kappa_mfs_fname}', color='darkgreen')
        ax[0,1].errorbar(nu, v1_mean, yerr=sigma_v1, color='darkgreen')
        ax[0,1].set_xlabel(r"$\nu$")
        ax[0,1].set_ylabel(r"$V_{1}(\nu)$")
        ax[0,1].legend(loc="upper right", prop={'size': 5})

        ax[0,2].plot(nu, v2_mean, label = r'$V_{2}(\nu)$' + '\t' + f'{mean_kappa_mfs_fname}', color='darkorange')
        ax[0,2].errorbar(nu, v2_mean, yerr=sigma_v2, color='darkorange')
        ax[0,2].set_xlabel(r"$\nu$")
        ax[0,2].set_ylabel(r"$V_{2}(\nu)$")
        ax[0,2].legend(loc="upper right", prop={'size': 5})
        
        fig.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
        fig.savefig(mean_plot_outpath + mean_kappa_mfs_fname + '.png', dpi=300)
        plt.close('all')

        # Write mean kappa MFs for the given scale to text
        np.savetxt(mean_DV_outpath + mean_kappa_mfs_fname + '.txt', np.column_stack([nu, v0_mean, sigma_v0, v1_mean, sigma_v1, v2_mean, sigma_v2]))
        

##----------------------------------------------------------------------------##


# Kappa-BNs Mean Calculator Function #
def compute_and_plot_mean_kappa_bns(los_list):

    # Make output folders for the considered statistics if they do not exist
    statistic = 'kappa_bns'
    DV_outpath = outpath + 'DVs/' + statistic + '/'
    plot_outpath = outpath + 'plots/' + statistic + '/'
    mean_DV_outpath = outpath + 'mean_DVs/' + statistic + '/'
    mean_plot_outpath = outpath + 'mean_plots/' + statistic + '/'
    os.makedirs(mean_DV_outpath, exist_ok=True)
    os.makedirs(mean_plot_outpath, exist_ok=True)
    
    # Define filter type
    kbns_filter = 'gaussian'
    
    # Compute mean kappa BNs for all filter scales
    for scale, scale_px in zip(scales, scales_px):
        
        # Initialize mean BNs
        betti0_mean, betti1_mean = 0., 0.
        
        for los in los_list:
    
            kappa_bns_fname = f'{str(los).zfill(3)}_{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}' + f'_filter_{kbns_filter}_scale_{scale}_{statistic}'
            nu, betti0, betti1 = np.genfromtxt(DV_outpath + kappa_bns_fname + '.txt', unpack=True)
            
            # Compute error on the mean
            if los == los_list[0]: data_betti0, data_betti1 = betti0 * 1, betti1 * 1 # Avoid shallow copy
            else:                  data_betti0, data_betti1 = np.column_stack([data_betti0, betti0]), np.column_stack([data_betti1, betti1])
            
        # Finalize error on the mean
        sigma_betti0, sigma_betti1 = np.std(data_betti0, axis=1, ddof=1), np.std(data_betti1, axis=1, ddof=1) # Compute sigma over all realisations of the same DV-element
        betti0_mean,  betti1_mean = np.mean(data_betti0, axis=1), np.mean(data_betti1, axis=1) # Compute mean over all realisations of the same DV-element
        
        # Define output file name
        mean_kappa_bns_fname = f'{cosmo}_{nbody_pipeline}_convergence_{KS_inversion_type}_{shear_type}_z_{str(z_bin_index).zfill(2)}' + f'_filter_{kbns_filter}_scale_{scale}_{statistic}'
        
        # Plot mean kappa BNs for the given scale
        fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(20, 5))
       
        ax[0,0].plot(nu, betti0_mean, label = r'$\beta_{0}(\nu)$' + '\t' + f'{mean_kappa_bns_fname}', color='blue')
        ax[0,0].errorbar(nu, betti0_mean, yerr=sigma_betti0, color='blue')
        ax[0,0].set_xlabel(r"$\nu$")
        ax[0,0].set_ylabel(r"$\beta_{0}(\nu)$")
        ax[0,0].legend(loc="upper right", prop={'size': 5})

        ax[0,1].plot(nu, betti1_mean, label = r'$\beta_{1}(\nu)$' + '\t' + f'{mean_kappa_bns_fname}', color='darkgreen')
        ax[0,1].errorbar(nu, betti1_mean, yerr=sigma_betti1, color='darkgreen')
        ax[0,1].set_xlabel(r"$\nu$")
        ax[0,1].set_ylabel(r"$\beta_{1}(\nu)$")
        ax[0,1].legend(loc="upper right", prop={'size': 5})
        
        fig.suptitle(f'z-bin {str(z_bin_index).zfill(2)}')
        fig.savefig(mean_plot_outpath + mean_kappa_bns_fname + '.png', dpi=300)
        plt.close('all')

        # Write mean kappa BNs for the given scale to text
        np.savetxt(mean_DV_outpath + mean_kappa_bns_fname + '.txt', np.column_stack([nu, betti0_mean, sigma_betti0, betti1_mean, sigma_betti1]))

    
##----------------------------------------------------------------------------##


###########
## Setup ##
###########

# Define simulation variables
nbody_pipelines = ['DUSTGRAIN']
DUSTGRAIN_cosmos = ['fR4_0.3eV']#,'fR4','fR5_0.1eV','fR5_0.15eV','fR5','fR6_0.1eV','fR6_0.06eV','fR6','LCDM']#, 'LCDM_Om02', 'LCDM_Om04', 'LCDM_s80707210', 'LCDM_s80976624', 'LCDM_w_-0.84', 'LCDM_w_-1.16']
KS_inversion_types = ['true'] # ['KS', 'KSP', 'KSPP']
shear_types = [''] # ['shear_noise_nomask', 'rshear_noise_mask']

# Define list of DUSTGRAIN los
DUSTGRAIN_nmaps = 256
DUSTGRAIN_los_list = np.arange(DUSTGRAIN_nmaps)

# Redshift bins
z_bin_list = [0.5]#[0.5, 1.0, 2.0, 4.0]

# Define map resolution
map_res = (10 / 1024) * 60 # pixel scale in arcmin (same for DUSTGRAIN and SLICS)

# Define filter and scales to use for the aperture mass map production
scales_px = np.array([pow(2,n) for n in np.arange(2, 5)]) # [4,8,16] pixels
scales = np.around(scales_px * map_res, decimals=2) # in arcmins


# Number of parallel processes (WARNING: set max nproc=4 for kappa-2pcf functions)
nproc = 4

# Buttons
measure_kappa2pcf = 0
measure_kappamoments = 1
measure_pdf = 0
measure_peaks = 1
measure_mfs = 1
measure_bns = 0


##----------------------------------------------------------------------------##


##########
## Main ##
##########

# Initialize whole analysis
t_tot1 = time.time()
print('\n' + f'Pipeline initialized \n')

for nbody_pipeline in nbody_pipelines:
    
    if nbody_pipeline == 'DUSTGRAIN': cosmos = DUSTGRAIN_cosmos;    los_list = DUSTGRAIN_los_list;    deg_side = int(5)
    
    for cosmo in cosmos:
        
        t_cosmo1 = time.time()
        print(f'{nbody_pipeline}_{cosmo} computation has started')
        
        # Define inpath/outpath
        inpath = 'inputs/'
        outpath = f'outputs/'
    
        # Make intermediate directories if needed [python3 3.2+]
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(outpath + 'plots/', exist_ok=True)
        os.makedirs(outpath + 'DVs/', exist_ok=True)
        os.makedirs(outpath + 'mean_DVs/', exist_ok=True)
        os.makedirs(outpath + 'mean_plots/', exist_ok=True)

        for shear_type in shear_types:
        
            for z_bin_index in z_bin_list:  
                           
                for KS_inversion_type in KS_inversion_types:
                    
                    if measure_kappa2pcf:

                        print('Computing 2-PCF')

                        # Produce auto-kappa-2pcf from kappa map
                        parallel(measure_kappa_2pcf_oneLOS, los_list, nproc)
    
                        # Compute mean auto-kappa-2pcf
                        compute_and_plot_mean_kappa_2pcf(los_list)

                                 
                    if measure_kappamoments:

                        print('Computing convergence moments')
                        
                        # Produce kappa_moments
                        parallel(measure_kappa_moments_oneLOS, los_list, nproc)
                        
                        # Compute mean kappa_moments
                        compute_and_plot_mean_kappa_moments(los_list)


                    if measure_pdf:  

                        print('Computing 1-PDF')
                                    
                        # Produce 1-pdf of kappa maps
                        parallel(measure_kappa_1pdf_oneLOS, los_list, nproc)
    
                        # Compute mean 1-pdf
                        compute_and_plot_mean_kappa_1pdf(los_list)
                    
                    if measure_peaks:

                        print('Computing peaks')
                    
                        # Produce peaks of kappa-apm maps
                        parallel(measure_kappa_peaks_oneLOS, los_list, nproc)
    
                        # Compute mean peaks
                        compute_and_plot_mean_kappa_peaks(los_list)
                        
                    if measure_mfs:

                        print('Computing MFs')
                    
                        # Produce MFs of kappa maps
                        parallel(measure_kappa_mfs_oneLOS, los_list, nproc)
                        
                        # Compute mean MFs
                        compute_and_plot_mean_kappa_mfs(los_list)
                        
                    if measure_bns:

                        print('Computing BNs')
                    
                        # Produce BNs of kappa maps
                        #measure_kappa_bns_oneLOS(los=0)
                        #measure_kappa_bns_oneLOS(los=1)
                        parallel(measure_kappa_bns_oneLOS, los_list, nproc)
                        
                        # Compute mean BNs
                        compute_and_plot_mean_kappa_bns(los_list)

        # Remove temporary directory before moving to the next cosmology
        if os.path.exists(outpath + 'tmp_kappa_map/'): os.removedirs(outpath + 'tmp_kappa_map/') 

        t_cosmo2 = time.time()
        print(f'{nbody_pipeline}_{cosmo} computation finished in: {(t_cosmo2 - t_cosmo1)/60} mins \n')
                                   
t_tot2 = time.time()
print(f'Pipeline total computational time = {(t_tot2 - t_tot1)/60} mins \n')