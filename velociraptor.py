import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.stats as st
import scipy.special as sp

from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from spectrum_class import Template, Spectrum
from model_class import Model, Line

global c
c=3.e5

"""
Contains the key routines of Velociraptor, called in wrapper.py. 
-->Parameter Initialization
-->Likelihood, Prior and Posterior Functions
"""

"""
Parameter Intialization
"""        
def init_ch(spectrum, temp, spec_reg):
    #Initializes guess for the absorption line strength parameter "c_h" over a given spectral region "spec_reg"
    mask_spec=(spectrum.lam>spec_reg['range'][0])&((spectrum.lam<spec_reg['range'][1]))
    mask_temp=(temp.lam>spec_reg['range'][0])&((temp.lam<spec_reg['range'][1]))
    temp_med=np.median(temp.flux[mask_temp])
    temp_min=np.min(temp.flux[mask_temp])
    star_med=np.median(spectrum.flux[mask_spec])
    star_min=np.min(spectrum.flux[mask_spec])
    c_h=((star_med/star_min)*temp_min - temp_med)/(1.-star_med/star_min)
    if c_h<-1.: 
        return 0.0
    else:
        return c_h

def initialize_continuum(spec, star_temp, aband_temp, spec_reg, npoly):
    #For each spec_reg, find initial guesses for c_h, b_i, v, and v_aband
    v_arr=np.arange(-500., 500., 1.)
    aband_arr=np.arange(-70., 70., 1.)
    b_i=np.array([])
    c_h=np.zeros(len(spec_reg))
    like_v=np.zeros(len(v_arr))
    like_aband=np.zeros(len(aband_arr))
    for i in range(len(spec_reg)):
        if spec_reg[i]['Telluric']:
            c_h[i]=init_ch(spec, aband_temp, spec_reg[i])
            b_i_reg=fit_cont_reg(spec, c_h[i], spec_reg[i], aband_temp, npoly)
            for j in range(len(aband_arr)):
                line=Line(spec_reg[i], aband_arr[j], c_h[i], b_i_reg, spec.lam, aband_temp)
                like_aband[j]=like_aband[j]+line.likelihood(spec)
            b_i=np.append(b_i, b_i_reg)
        else:
            c_h[i]=init_ch(spec, star_temp, spec_reg[i])
            b_i_reg=fit_cont_reg(spec, c_h[i], spec_reg[i], star_temp, npoly)
            b_i=np.append(b_i, b_i_reg)
            for j in range(len(v_arr)):
                line=Line(spec_reg[i], v_arr[j], c_h[i], b_i_reg, spec.lam, star_temp)
                like_v[j]=like_v[j]+line.likelihood(spec)
    v_init=v_arr[like_v == np.max(like_v)]
    aband_init=aband_arr[like_aband==np.max(like_aband)]
#    b_i=b_i.reshape((len(spec_reg), -1))
    init_spec_theta=np.concatenate((v_init, aband_init, c_h, b_i))
    return init_spec_theta      

def fit_cont_reg(spectrum, c_h, spec_reg, temp, npoly):
    #Provide initial fit to spectrum continuum region (divided by template)
    #Pick out "continuum regions" that vary from median by less than 5% (to exclude absorption lines)
    mask=(spectrum.lam>spec_reg['range'][0])&((spectrum.lam<spec_reg['range'][1]))
    thresh=0.05
    lam_fit=spectrum.lam[mask]
    flux_fit=(temp.flux+c_h)/(1.+c_h)
    f_interp=interpolate.interp1d(temp.lam, flux_fit)
    spec_fit=spectrum.flux[mask]/(f_interp(spectrum.lam[mask]))
    cont_mask=(np.abs((spec_fit-np.median(spec_fit))/spec_fit) < thresh)
    #If no entries in cont_mask, use full spectral region to estimate continuum level
    if (np.sum(cont_mask)==0):
        cont_mask=np.ones(len(spec_fit), dtype=bool)
    #Shift wavelength to scale of -1 to 1 for fit
    lam_eval=2.*(lam_fit-np.min(lam_fit))/np.max(lam_fit-np.min(lam_fit))-1.
    b_i=np.polynomial.legendre.legfit(lam_eval[cont_mask], spec_fit[cont_mask], npoly)
    return b_i
    
def get_inds_dict(theta, nspec, nreg):
    #Given a theta array, return dictionary that contains indicies of velocities, abands, and c_h
    param_ind_dict={}
    nspec_params=len(theta[2:])/nspec
    param_ind_dict['v_raw']=np.arange(2, len(theta), nspec_params, dtype=np.int8)
    param_ind_dict['v_aband']=np.arange(3, len(theta), nspec_params, dtype=np.int8)
    c_h_arr=np.array([], dtype=np.int8)
    for i in range(0, nreg):
        inds_c=np.arange(4+i, len(theta), nspec_params, dtype=np.int8)
        c_h_arr=np.append(c_h_arr, inds_c)
    param_ind_dict['c_h']=c_h_arr
    return param_ind_dict
    
def get_spec_mask(spectrum, spec_reg):
    mask_list=[]
    for i in range(len(spec_reg)):
        mask_line=(spectrum.lam>spec_reg[i]['range'][0])&((spectrum.lam<spec_reg[i]['range'][1]))
#        mask_tmp=np.ma.make_mask(mask_line)
        mask_list.append(mask_line)
    mask=np.any(np.column_stack(mask_list), axis=1)
    return mask
    
def get_chip_gap_mask(spectrum):
    #Determines if the chip gap is present in the masked spectrum (if so, would be near A-band).
    #Removes chip gap and and the 3 pixels on either side from analysis.
    bad_pix=(spectrum.ivar==0)
    mask = np.ones(len(spectrum.lam), np.bool)
    if np.sum(bad_pix)>0:
        bad_pix_min=np.where(bad_pix)[0][0]-3
        bad_pix_max=np.where(bad_pix)[0][-1]+3
        mask[bad_pix_min:bad_pix_max] = 0
    return mask
         
"""
Prior and Posterior Functions for a Single Spectrum velocity measurement
"""
        
def ln_post_single(theta_spec, spectrum, star_temp, aband_temp, spec_reg):
    nspec_reg=len(spec_reg)
    prior=ln_prior_single(theta_spec, spectrum, nspec_reg)
    if np.isneginf(prior):
        return prior
    else:
        like=ln_likelihood(theta_spec, spectrum, star_temp, aband_temp, spec_reg)
        return np.sum(like)+prior   

def ln_prior_single(theta_spec, spectrum, n_spec_reg):
    c_h_arr=theta_spec[2:2+n_spec_reg]
    #Gamma prior on absorption line strength
    if any(c_h_arr < -1.):
        return -np.inf
    #Place limits on velocities
    elif (np.abs(theta_spec[0])>600.):
        return -np.inf
    elif(np.abs(theta_spec[1])>100.):
        return -np.inf
    else:
#        return -np.sum(np.log(c_h_arr+1))
#        return -0.5*np.sum(np.log(sig_arr)) 
        return np.sum(st.gamma.logpdf(c_h_arr+1., 2.0, loc=0., scale=0.5))   
         
"""
Likelihood, Prior and Posterior Functions for "Hierarchical Mode"
"""    
    
def ln_post(theta, spec_list, star_temp, aband_temp, spec_reg, params_ind_dict):
    #Pull out hyperparameters
    v, sig_sys2 = theta[0:2]
    nspec=len(spec_list)
    #Reshape array into nspec x nspec_params array 
    theta_spec=np.reshape(theta[2:], (nspec, -1))
    prior=ln_prior(theta, spec_reg, nspec, params_ind_dict)
    if np.isneginf(prior):
        return prior
    else:
        #Evaluate likelihood and compute v_corr for each spectrum. 
        like=np.zeros(nspec)
        v_corr=np.zeros(nspec)
        for i in range(nspec):
            like[i]=ln_likelihood(theta_spec[i], spec_list[i], star_temp, aband_temp, spec_reg)
            v_corr[i]=theta_spec[i, 0]-theta_spec[i, 1]-spec_list[i].vhelio
        return np.sum(like)+prior+like_hyperparams(v, sig_sys2, v_corr)
        
def like_hyperparams(v_mean, sig_sys2, v_corr):
    #Determines likelihood of hyperparameters given corrected velocities 
    ln_like=st.norm.logpdf(v_corr, loc=v_mean, scale=(sig_sys2**(1./2.)))
    return np.sum(ln_like)
    
def ln_likelihood(theta_spec, spectrum, star_temp, aband_temp, spec_reg):
    #Creates model spectrum and evaluates likelihood of model parameters given 
    model = Model(theta_spec, spec_reg, spectrum, star_temp, aband_temp)
    return model.likelihood(spectrum)

def ln_prior(theta, spec_reg, nspec, params_ind_dict):
    """"
    Evaluates prior on all model parameters.
    """
    sig2=theta[1]
    nspec_reg=len(spec_reg)
    c_h_arr=theta[params_ind_dict['c_h']]
    if sig2 <= 0:
        return -np.inf
    elif (np.abs(theta[0])>600.):
        return -np.inf
    elif any(np.abs(theta[params_ind_dict['v_raw']])>600.):
        return -np.inf
    elif any(np.abs(theta[params_ind_dict['v_aband']])>100.):
        return -np.inf
    elif any(c_h_arr < -1.):
        return -np.inf
    else:
        return np.sum(st.gamma.logpdf(c_h_arr+1., 2.0, loc=0., scale=0.5))+logigam_pdf(sig2, 7., 9.*7.+9.)

def logigam_pdf(x, a, b):
    #Compute logarithm of the PDF of an Inverse-Gamma distribution with parameters a,b
    return a*np.log(b)-sp.gammaln(a)+(-a-1.)*np.log(x)-b/x
             
