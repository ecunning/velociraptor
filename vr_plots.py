import numpy as np
import matplotlib.pyplot as plt
import corner
from model_class import Model
import pdb
#import os

"""
Creates diagnostic plots for velociraptor run in single mode.
--Corners
--Traces
--Spectra with models overplotted 
"""

def single_mode_diagnostics(dir_name, sampler, spec, spec_reg, star_temp, aband_temp, nburn, ndim, nwalkers):
#    ndim=len(sampler.chain[0,0,:])
#    if not os.path.exists(dir_name):
#        os.makedirs(dir_name)
    nrow=int(ndim/2)+ndim%2
#    pdb.set_trace()
    f, axarr=plt.subplots(nrow, 2, figsize=(15, 15))
    ind=np.random.randint(0, nwalkers, 50)
    for i in range(nrow):
        axarr[i, 0].plot(sampler.chain[ind,:,2*i].T)
        if i<(nrow-1):
            axarr[i, 1].plot(sampler.chain[ind,:,2*i+1].T)
        elif ndim%2==0:
            axarr[i, 1].plot(sampler.chain[ind,:,2*i+1].T)
    sample=sampler.chain[:, nburn:, :].reshape(-1, ndim) 
    plt.savefig(dir_name+'/traces.png')
    plt.clf()       
    f2=corner.corner(sample)
    plt.savefig(dir_name+'/corner.png')
    plt.clf()
    plot_ind=np.random.randint(0, len(sample[:,0]), 20)
    f3, (ax1, ax2, ax3)=plt.subplots(3, 1, figsize=(15, 10))
    ax1.plot(spec.lam, spec.flux)
    ax1.set_xlim(6500, 6650)
    ax2.plot(spec.lam, spec.flux)
    ax2.set_xlim(7500, 7750)
    ax3.plot(spec.lam, spec.flux)
    ax3.set_xlim(8450, 8700)
    for i in range(len(plot_ind)):
#for i in range(1):
        theta_spec=sample[plot_ind[i],:]
        model = Model(theta_spec, spec_reg, spec, star_temp, aband_temp)
        ax1.plot(model.lam, model.flux,color='red', alpha=0.3)
        ax2.plot(model.lam, model.flux,color='red', alpha=0.3)
        ax3.plot(model.lam, model.flux,color='red', alpha=0.3)
    plt.savefig(dir_name+'/spec_models.png')
    plt.clf()         