import numpy as np
from scipy import interpolate
import scipy.special as sp
import matplotlib.pyplot as plt
import pdb

import emcee
import corner

from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u

import argparse
import os, fnmatch

from spectrum_class import Spectrum, Template
from model_class import Model, Line
import velociraptor as vr
import vr_plots


global c
c=3.e5

"""
Wrapper for running Velociraptor. 

Example command line call for an object named 12345 in the COSMOS field to be run in background:  

python wrapper.py 12345 COSMOS -p & 
    
Field name is only used for file organizational purposes. 
    
This wrapper:
    
    -Finds all reduced spectra files corresponding to a given object ID in "reduction_path"
    -Creates Spectrum objects from these files
    -Runs Velociraptor in "Single-Mode" on the individual spectra, in order to initialize walkers for hierarchical mode
    -Runs Velociraptor in Hierarchical Mode
    -If run with "-p" flag, will save acceptance fraction plot and trace plots for all model parameters.  

"""

def run_vr_single(spec, spec_reg, star_temp, aband_temp, npoly, nwalkers, nsteps, nburn, plot=False, \
return_last_step=False, save_chain=False, return_chain=False, return_sample=False):
    """
    Measure the velocity from a single observation. 
    """

    spec_theta=vr.initialize_continuum(spec, star_temp, aband_temp, spec_reg, npoly)
    ndim=len(spec_theta)
    nreg=len(spec_reg)
    starting_guesses=np.zeros((nwalkers, ndim))
    starting_guesses[:, 0:2]=np.random.normal(spec_theta[0:2], 3.*np.ones(2), (nwalkers, 2))
    starting_guesses[:, 2:2+nreg]=np.random.normal(spec_theta[2:2+nreg], 0.1, (nwalkers, nreg))
    #Make sure c_h parameter is not out of range
    bad=(starting_guesses[:,2:2+nreg]<=-1.)
    starting_guesses[:,2:2+nreg][bad]=-0.9
    starting_guesses[:, 2+nreg::2]=np.random.normal(spec_theta[2+nreg::2], 0.1*np.abs(spec_theta[2+nreg::2]), (nwalkers, nreg))
    starting_guesses[:, 3+nreg::2]=np.random.normal(spec_theta[3+nreg::2], 10.*np.ones(nreg), (nwalkers, nreg))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, vr.ln_post_single, args=[spec, star_temp, aband_temp, spec_reg])
    sampler.run_mcmc(starting_guesses, nsteps)
    sample = sampler.chain
    sample=sampler.chain[:, nburn:, :].reshape(-1, ndim)
    if save_chain:
        np.save('chain.npy')
    if return_sample:
        return sample
    if plot:
        vr_plots.single_mode_diagnostics('.', sampler, spec, spec_reg, star_temp, aband_temp, nburn, ndim, nwalkers)
    if return_last_step:
        return sampler.chain[:, -1, :].reshape(nwalkers, ndim)
    if return_chain:
        return sampler.chain
    else:
        return

def init_sigma(nwalkers):
    """
    Initialize walkers for hyper-parameter sigma^2 by sampling from the prior distribution, using Monte Carlo accept/reject.
    """
    samples=np.random.random(10000)*30.
    u=np.random.rand(10000)*0.1
    pdfs=igam_pdf(samples, 7., 9.*7.+9.)
    keep=(u<pdfs)
    inds=np.random.random_integers(0, high=len(samples[keep])-1, size=nwalkers)
    return samples[keep][inds]

def igam_pdf(x, a, b):
    """
    Evaluate the probability density function for the Inverse-Gamma distribution.
    """
    return b**a/sp.gamma(a)*x**(-a-1)*np.exp(-b/x)

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
    
"""
Beginning of wrapper routine, to be run at the command line.       
"""

parser = argparse.ArgumentParser()
parser.add_argument("obj_id", help="ID number of the object of interest (string)", type=str)
parser.add_argument("field", help="Field Name", type=str)
#parser.add_argument("slit_id", help="Slit number of the object of interest (string)", type=str)
parser.add_argument("-p", "--plot", action="store_true", help="save trace, corner and model plots")
args = parser.parse_args()

"""
Defining template file and spectral regions.
"""

temp_file='/net/phizo/data/d/deimos_apr12/HD105546/spec1d.LVMslits.001.slit0.8.fits.gz'

star_temp=Template(temp_file, 37.9)
aband_temp=Template(temp_file, 0.0)
star_temp.flux=star_temp.flux/np.median(star_temp.flux)
aband_temp.flux=aband_temp.flux/np.median(aband_temp.flux)
h_alpha={'Name':'H-alpha', 'range':[6500, 6650], 'Telluric':False}
bband={'Name':'B-Band', 'range':[6800, 6950], 'Telluric':True}
aband={'Name':'A-Band', 'range':[7500, 7750], 'Telluric':True}
cat={'Name':'Calcium Triplet', 'range':[8450, 8700], 'Telluric':False}
spec_reg=[h_alpha, aband, cat]
#Order of polynomial for continuum fit over a single spectral region:
npoly=1

"""
Define where reduced spectra are stored, and find all files belonging to a given OBJ-ID.
"""

reduction_path='/net/phizo/data/d/halo7d_reductions/'
search_path=reduction_path+args.field+'/'
pattern='spec1d.*.'+args.obj_id+'.fits.gz'
spec_files=find(pattern, search_path)


#Create directory to store output
save_path='/net/phizo/data/d/velociraptor/'+args.field+'/'+args.obj_id
if not os.path.exists(save_path):
    os.makedirs(save_path)

#Read in each of the spec1d files and store in list of Spectrum objects.

spec_list=[]
for i in range(len(spec_files)):
    spec_tmp=Spectrum(spec_files[i])
    h_alpha=(spec_tmp.lam>6500.)&(spec_tmp.lam<6650.)
    snr=np.median(spec_tmp.flux[h_alpha]*(spec_tmp.ivar[h_alpha])**(1./2.))
    mask=vr.get_spec_mask(spec_tmp, spec_reg)
    spec_tmp.mask_spec(mask)
    mask_chip=vr.get_chip_gap_mask(spec_tmp)
    spec_tmp.mask_spec(mask_chip)
    if snr>0.:
        spec_list.append(spec_tmp)


nspec=len(spec_list)

#Initialize parameters for single-mode.

v_init=np.zeros(nspec)
aband_init=np.zeros(nspec)
v_helio=np.zeros(nspec)
spec_theta=np.array([])
for i in range(len(spec_list)):
    spec_theta_tmp=vr.initialize_continuum(spec_list[i], star_temp, aband_temp, spec_reg, npoly)
    v_helio[i]=spec_list[i].vhelio
    v_init[i]=spec_theta_tmp[0]
    aband_init[i]=spec_theta_tmp[1]
    spec_theta=np.append(spec_theta, spec_theta_tmp)
spec_theta=spec_theta.reshape(nspec, -1)
#Initialize Parameters: Run in single mode first
ndim=len(spec_theta[0])
nwalkers=800
nburn=500
nsteps=700
#print(ndim)

nreg=len(spec_reg)
v_corr_array=np.zeros((nwalkers*(nsteps-nburn), nspec))
starting_vals=np.array([])
starting_scale=np.array([])
last_step=np.zeros((nspec, nwalkers, ndim))
v_corr_array=np.zeros((nwalkers*(nsteps-nburn), nspec))

#Run VR in single mode to get intialize walkers for hierarchical mode. 
for i in range(nspec):
    chain=run_vr_single(spec_list[i], spec_reg, star_temp, aband_temp, npoly, nwalkers, \
                            nsteps, nburn, plot=False, return_chain=True)
    sample=chain[:,nburn:, :].reshape(-1, ndim)
    last_step[i] = chain[:, -1, :].reshape(nwalkers, ndim)
    v_corr=sample[:,0]-sample[:,1]-v_helio[i]
    v_corr_array[:,i]=v_corr

np.save(save_path+'/v_corr.npy', v_corr_array)


#Now run in hierarchical mode

ndim=(nspec)*11+2
nwalkers=800
nburn=4800
nsteps=5000

starting_guesses=np.zeros((nwalkers, ndim))
starting_guesses[:,0]=np.random.normal(np.mean(v_corr_array), 7., nwalkers)
##Generate random samples from prior 
starting_guesses[:,1]=init_sigma(nwalkers)

for i in range(0, nspec):
    starting_guesses[:, 2+11*i:2+11*(i+1)]=last_step[i, :, :]
    
#This is for bookkeeping purposes; for keeping track of which indicies in the parameter array correspond to which parameters.
param_dict = vr.get_inds_dict(starting_guesses[0,:], nspec, len(spec_reg))
velocities=np.concatenate(([0], param_dict['v_raw'], param_dict['v_aband']))
coeff_mask=np.ma.make_mask(np.ones(len(starting_guesses[0,:])))
coeff_mask[1]=False
coeff_mask[velocities]=False
coeff_mask[param_dict['c_h']]=False
inds_coeff=np.where(coeff_mask)[0]
c_0_inds=inds_coeff[::2]
c_1_inds=inds_coeff[1::2]


#Intialize sampler and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, vr.ln_post, args=[spec_list, star_temp, aband_temp, spec_reg, param_dict], a=1.3)

sampler.run_mcmc(starting_guesses, nsteps) 
sample=sampler.chain[:, nburn:, :].reshape(-1, ndim)
chain=sampler.chain

np.save(save_path+'/chain.npy', sampler.chain)

#Save acceptance fraction figure and trace plots. 

if (args.plot):

    f_acc=plt.figure(1)
    plt.plot(sampler.acceptance_fraction)
    plt.savefig(save_path+'/acceptance_fraction.png')
    plt.clf()

    f, ax = plt.subplots(nspec+1,2, figsize=(15, 10), sharex=True)
    steps=np.arange(nsteps)
    ind=np.random.randint(0, nwalkers, 30)
    ax[0, 0].plot(chain[ind,:,0].T)
    ax[0,0].set_ylabel('v_mean')
    ax[0,1].plot(chain[ind,:,1].T)
    ax[0,1].set_ylabel('sigma')
    for i in range(0, nspec):
        ax[i+1, 0].plot(chain[ind,:,param_dict['v_raw'][i]].T)
        ax[i+1, 0].set_ylabel('v'+str(i))
        ax[i+1, 1].plot(chain[ind,:,param_dict['v_aband'][i]].T)
        ax[i+1, 1].set_ylabel('v_aband'+str(i))
    plt.savefig(save_path+'/velocity_traces.png')
    plt.clf()

    f, ax = plt.subplots((nspec),3, figsize=(15, 15))
    for i in range(0, nspec):
        ax[i, 0].plot(chain[ind,:,param_dict['c_h'][i]].T)
        ax[i, 0].set_ylabel('c_h'+str(i))
        ax[i, 1].plot(chain[ind,:,param_dict['c_h'][i+(nspec-1)]].T)
        ax[i, 1].set_ylabel('c_h_aband'+str(i))
        ax[i, 2].plot(chain[ind,:,param_dict['c_h'][i+2*(nspec-1)]].T)
        ax[i, 2].set_ylabel('c_h_cat'+str(i))
    plt.savefig(save_path+'/c_h_traces.png')
    plt.clf()

    f, ax = plt.subplots(nspec,3, figsize=(15, 15))
    for i in range(len(c_0_inds)/3):
        ax[i, 0].plot(chain[ind,:,c_0_inds[3*i]].T)
        ax[i, 0].set_ylabel('c_0_h'+str(i))
        ax[i, 1].plot(chain[ind,:,c_0_inds[3*i+1]].T)
        ax[i, 1].set_ylabel('c_0_aband'+str(i))
        ax[i, 2].plot(chain[ind,:,c_0_inds[3*i+2]].T)
        ax[i, 2].set_ylabel('c_0_cat'+str(i))
    plt.savefig(save_path+'/c_0_traces.png')
    plt.clf()

    f, ax = plt.subplots(nspec,3, figsize=(15, 15))
    for i in range(len(c_1_inds)/3):
        ax[i, 0].plot(chain[ind,:6300,c_1_inds[3*i]].T)
        ax[i, 0].set_ylabel('c_1_h'+str(i))
        ax[i, 1].plot(chain[ind,:6300,c_1_inds[3*i+1]].T)
        ax[i, 1].set_ylabel('c_1_aband'+str(i))
        ax[i, 2].plot(chain[ind,:6300,c_1_inds[3*i+2]].T)
        ax[i, 2].set_ylabel('c_1_cat'+str(i))
    plt.savefig(save_path+'/c_1_traces.png')
    plt.clf()

print(args.obj_id, 'Done!')
    


