import numpy as np
import sys
sys.path.insert(0, '/home/alessandro/code')

from MGCAMB import camb
import pyreact
import pyhmcode as hmcode

# set cosmology

# to be changed, params change for each cosmology
Omega_c = 0.25
Omega_b = 0.05
h = 0.7
n_s = 0.97
A_s = 2.1e-9
m_nu = 0.0
w0 = -1.0
w_a = 0.0

p = camb.CAMBparams(WantTransfer=True,
                    Want_CMB=False, Want_CMB_lensing=False, DoLensing=False,
                    NonLinear="NonLinear_none",
                    WantTensors=False, WantVectors=False, WantCls=False, WantDerivedParameters=False,
                    want_zdrag=False, want_zstar=False)

p.set_cosmology(H0=h*100, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, omk=0, mnu=m_nu)
p.set_initial_power(camb.initialpower.InitialPowerLaw(As=A_s, ns=n_s))
p.set_dark_energy(w=w0,wa=w_a)

# binning of DUSTGRAIN-pathfinder simulations (arXiv:1806.04681v3)
z_bins = [0.5, 1.0, 2.0, 4.0]

# binning in 10 bins, from z min to z max
#z_bottom = 0.5
#z_top = 4.0
#z_bins = np.linspace(z_bottom,z_top,10)

#React default
#p.set_matter_power(redshifts=np.linspace(0.0, 1.0, 4, endpoint=True)[::-1], # to be changed
#                   kmax=10.0, nonlinear=False)

p.set_matter_power(redshifts=z_bins[::-1],kmax=10.0,nonlinear=False)

r = camb.get_results(p)
sigma_8 = r.get_sigma8()[-1]
k_lin, z_lin, pofk_lin_camb = r.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints=128)

Omega_v = r.omega_de + r.get_Omega("photon") + r.get_Omega("neutrino")
Omega_m = p.omegam

# Model selection and parameter (gr,f(r),dgp,quintessence or cpl)
mymodel = "f(R)"
fR0 = 1e-5 #here it is the absolute value of fR0
Omega_rc = None
massloop = 30

react = pyreact.ReACT()

# Only compute the reaction up to z=2.5
z_lin = np.array(z_lin)
#z_react = z_lin[z_lin < 2.5]
z_react = z_lin[z_lin<4.0] # changed to the last z bin

R, pofk_lin_MG_react,sigma_8_MG = react.compute_reaction(
                                h, n_s, Omega_m, Omega_b, sigma_8, z_react, k_lin, pofk_lin_camb[0], model=mymodel,
                                fR0=fR0, Omega_rc=Omega_rc, w=w0, wa=w_a,
                                is_transfer=False, mass_loop=massloop,
                                verbose=True)

hmc = hmcode.Cosmology()

# Set HMcode internal cosmological parameters
hmc.om_m = Omega_c+Omega_b
hmc.om_b = Omega_b
hmc.om_v = 1.-(Omega_c+Omega_b)
hmc.h = h
hmc.ns = n_s
hmc.sig8 = sigma_8_MG
hmc.m_nu = m_nu

# Set the halo model in HMcode
# Options: HMcode2015, HMcode2016, HMcode2020
hmod = hmcode.Halomodel(hmcode.HMcode2020, verbose=False)
# Power spectrum calculation for f(R)
z_f = 0
hmc.set_linear_power_spectrum(k_lin, np.asarray(z_react), pofk_lin_MG_react)
Pk_hm_fR = hmcode.calculate_nonlinear_power_spectrum(hmc, hmod, verbose=True)


z_f = 1
D_LCDM_z1 = 0.4765851677712304
D_LCDM_z0 = 0.7789813117487887

pofk_lin_MG_react_new = np.zeros(pofk_lin_MG_react.shape)
for j in [1, 2, 3]:
    for i in range(len(pofk_lin_MG_react[1])):
        pofk_lin_MG_react_new[j][i] = pofk_lin_MG_react[j][i]        
for i in range(len(pofk_lin_MG_react[0])):       
    pofk_lin_MG_react_new[0][i] = (D_LCDM_z0/D_LCDM_z1)**2 * pofk_lin_MG_react[-1][i]

    
hmc.set_linear_power_spectrum(k_lin, np.asarray(z_react), pofk_lin_MG_react_new)    
Pk_hm_fR_z1 = hmcode.calculate_nonlinear_power_spectrum(hmc, hmod, verbose=False)  

# Power spectrum calculation for GR
hmc.sig8 = sigma_8
hmc.set_linear_power_spectrum(k_lin, np.asarray(z_react), pofk_lin_camb)
Pk_hm_GR = hmcode.calculate_nonlinear_power_spectrum(hmc, hmod, verbose=False)  
