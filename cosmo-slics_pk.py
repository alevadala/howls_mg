import numpy as np
import sys
sys.path.insert(0, '/home/alessandro/code')

from MGCAMB import camb

import os

import csv

from time import time

# Creating necessary folders
outpath = 'Cosmo-SLICS/'
pk_path = outpath+'PKnl/'
print('Creating necessary directories\n')
os.makedirs(outpath, exist_ok=True)
os.makedirs(pk_path, exist_ok=True)

# Cosmo-SLICS fixed parameters
Ob = 0.0473
As = 2.199e-9 # Fiducial, to be changed for different sigma8
ns = 0.969
Onu = 0
mnu = 0 # Sadly no neutrinos

# Cosmo-SLICS points in parameters space (fixed parameters: ns = 0.969, Ob = 0.0473, Onu = 0)
cosmo_sets = {
    'FID': {'Om': 0.2905, 'Oc': 0.2432, 'h': 0.6898, 'w0': -1., 'sig8': 0.8364},
    '00': {'Om': 0.3282, 'Oc': 0.2809, 'h': 0.6766, 'w0': -1.2376, 'sig8': 0.6677},
    '01': {'Om': 0.1019, 'Oc': 0.0546, 'h': 0.7104, 'w0': -1.6154, 'sig8': 1.3428},
    '02': {'Om': 0.2536, 'Oc': 0.2063, 'h': 0.6238, 'w0': -1.7698, 'sig8': 0.6670},
    '03': {'Om': 0.1734, 'Oc': 0.1261, 'h': 0.6584, 'w0': -0.5223, 'sig8': 0.9581},
    '04': {'Om': 0.3759, 'Oc': 0.3286, 'h': 0.6034, 'w0': -0.9741, 'sig8': 0.8028},
    '05': {'Om': 0.4758, 'Oc': 0.4285, 'h': 0.7459, 'w0': -1.3046, 'sig8': 0.6049},
    '06': {'Om': 0.1458, 'Oc': 0.0985, 'h': 0.8031, 'w0': -1.4498, 'sig8': 1.1017},
    '07': {'Om': 0.3099, 'Oc': 0.2626, 'h': 0.6940, 'w0': -1.8784, 'sig8': 0.7734},
    '08': {'Om': 0.4815, 'Oc': 0.4342, 'h': 0.6374, 'w0': -0.7737, 'sig8': 0.5371},
    '09': {'Om': 0.3425, 'Oc': 0.2952, 'h': 0.8006, 'w0': -1.5010, 'sig8': 0.6602},
    '10': {'Om': 0.5482, 'Oc': 0.5009, 'h': 0.7645, 'w0': -1.9127, 'sig8': 0.4716},
    '11': {'Om': 0.2898, 'Oc': 0.2425, 'h': 0.6505, 'w0': -0.6649, 'sig8': 0.7344},
    '12': {'Om': 0.4247, 'Oc': 0.3774, 'h': 0.6819, 'w0': -1.1986, 'sig8': 0.6313},
    '13': {'Om': 0.3979, 'Oc': 0.3506, 'h': 0.7833, 'w0': -1.1088, 'sig8': 0.7360},
    '14': {'Om': 0.1691, 'Oc': 0.1218, 'h': 0.7890, 'w0': -1.6903, 'sig8': 1.1479},
    '15': {'Om': 0.1255, 'Oc': 0.0782, 'h': 0.7567, 'w0': -0.9878, 'sig8': 0.9479},
    '16': {'Om': 0.5148, 'Oc': 0.4675, 'h': 0.6691, 'w0': -1.3812, 'sig8': 0.6243},
    '17': {'Om': 0.1928, 'Oc': 0.1455, 'h': 0.6285, 'w0': -0.8564, 'sig8': 1.1055},
    '18': {'Om': 0.2784, 'Oc': 0.2311, 'h': 0.7151, 'w0': -1.0673, 'sig8': 0.6747},
    '19': {'Om': 0.2106, 'Oc': 0.1633, 'h': 0.7388, 'w0': -0.5667, 'sig8': 1.0454},
    '20': {'Om': 0.4430, 'Oc': 0.3957, 'h': 0.6161, 'w0': -1.7037, 'sig8': 0.6876},
    '21': {'Om': 0.4062, 'Oc': 0.3589, 'h': 0.8129, 'w0': -1.9866, 'sig8': 0.5689},
    '22': {'Om': 0.2294, 'Oc': 0.1821, 'h': 0.7706, 'w0': -0.8602, 'sig8': 0.9407},
    '23': {'Om': 0.5095, 'Oc': 0.4622, 'h': 0.6988, 'w0': -0.7164, 'sig8': 0.5652},
    '24': {'Om': 0.3652, 'Oc': 0.3179, 'h': 0.7271, 'w0': -1.5414, 'sig8': 0.5958},
}

# Define the range for k (just k, not k/h)
k_min = 10**(-4.1) # for lower values CAMB does not compute P(z, k)
k_max = 100 #10**(1.5) # set to higher value

# Define the redshift range
z_range = np.linspace(0, 4, 100)

t1 = time()

for key in cosmo_sets.keys():

    print(f'Starting computation for model {key}\n')

    h = cosmo_sets[key]['h']
    Oc = cosmo_sets[key]['Oc']
    w0 = cosmo_sets[key]['w0']

    print('Computing initial transfer functions\n')
    # Computing initial transfer functions
    pars = camb.CAMBparams(WantTransfer=True, 
                            Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                            NonLinear = 'NonLinear_pk',
                            omnuh2=0,
                            WantTensors=False, WantVectors=False, WantCls=False, WantDerivedParameters=False,
                            want_zdrag=False, want_zstar=False,
                            MG_flag = 0)

    pars.set_cosmology(H0=h*100, ombh2=Ob*h**2, omch2=Oc*h**2, omk=0, mnu=mnu)
    pars.set_initial_power(camb.initialpower.InitialPowerLaw(As=As, ns=ns))
    pars.set_dark_energy(w=w0)

    pars.set_matter_power(redshifts=z_range[::-1], kmax=k_max)

    res = camb.get_results(pars)
    sigma8_init = res.get_sigma8()[-1]

    # Compute target As for desired sigma8
    target_sig8 = cosmo_sets[key]['sig8']

    As_target = As*(target_sig8/sigma8_init)**2

    # Setting CAMB with the desired AS
    pars.set_initial_power(camb.initialpower.InitialPowerLaw(As=As_target, ns=ns))
    pars.set_matter_power(redshifts=z_range[::-1], kmax=k_max)

    res = camb.get_results(pars)
    sigma_8 = res.get_sigma8()[-1]

    # Checking that computed sigma8 matches the desired sigma8
    if (round(sigma_8,4)-target_sig8) == 0:
        print('Sigma 8 value is correct!\n')
    else:
        raise ValueError('Sigma 8 values do not match!')
    
    # Computing non-linear P(z,k) in correct units
    print(f'Computing non-linear P(z,k) for sigma8={sigma_8:.4f}\n')
    kh_camb, z_camb, pk_nonlin = res.get_nonlinear_matter_power_spectrum(hubble_units=False, k_hunit=False)

    # Saving power spectra, k, and z arrays to file
    print('Saving on file\n')
    with open(pk_path+f'logPk_cosmoslics_ID={key}_s8={sigma_8:.4f}.txt','w',newline='\n') as file:
        writer = csv.writer(file)
        writer.writerows(np.log10(pk_nonlin))
    
    with open(pk_path+f'Pk_cosmoslics_ID={key}_s8={sigma_8:.4f}.txt','w',newline='\n') as file:
        writer = csv.writer(file)
        writer.writerows(pk_nonlin)

np.savetxt(outpath+f'k.txt',kh_camb)
np.savetxt(outpath+f'log_k.txt',np.log10(kh_camb))
np.savetxt(outpath+f'z.txt',z_camb)

t2 = time()

print(f'Total time: {int((t2-t1)/60)} min {(t2-t1)%60:.2f} s')