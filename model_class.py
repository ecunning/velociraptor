import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm
import pdb

global c
c=3.e5

"""
Contains the routines for creating model spectra given parameters. 
"""

class Line(object):
    """
    Model for a single region of the spectrum (e.g., H-alpha, A-band,CaT). Used only in initialization routines.
    
    Input spectral region dictionary and params.
    """
    def __init__(self, spec_reg, v, c_h, b_i, lam0, temp):
        self.name=spec_reg['Name']
        self.v=v
        self.c_h=c_h
        self.b_i=b_i
        mask=(lam0>spec_reg['range'][0])&((lam0<spec_reg['range'][1]))
        self.lam = lam0[mask]
        lam_temp=temp.lam*(1.+(self.v-temp.v)/c)
        temp_flux=(temp.flux+self.c_h)/(1.+self.c_h)
        f_new=interpolate.interp1d(lam_temp, temp_flux)
        lam_eval=2.*(self.lam-np.min(self.lam))/np.max(self.lam-np.min(self.lam))-1.
        self.flux=f_new(lam0[mask])*np.polynomial.legendre.legval(lam_eval, self.b_i)
        
    def likelihood(self, spectrum):        
        mask=(spectrum.lam>=np.min(self.lam))&((spectrum.lam<=np.max(self.lam)))
        noise_fit=(1./spectrum.ivar[mask])**(1./2.)
        ln_like=norm.logpdf(spectrum.flux[mask], loc=self.flux, scale=noise_fit)
        return np.sum(ln_like)
    
class Model(object):
    """
    Creates model spectrum (list of n_spec_reg Lines). 
    
    Likelihood method evaluates the likelihood of science Spectrum given model parameters.
    """
    def __init__(self, theta_spec, spec_reg, Spectrum, star_temp, aband_temp):
        #Takes v_raw, v_aband, spec_reg_params1...spec_reg_paramsN
        #Determine number of spectral regions and params per region
        n_spec_reg=len(spec_reg)
        n_reg_params=len(theta_spec[2:])/n_spec_reg
        #Unpack Theta Array
        v_raw, v_aband=theta_spec[0:2]
        c_h_arr=theta_spec[2:2+n_spec_reg]
        b_i=theta_spec[2+n_spec_reg:].reshape(n_spec_reg, n_reg_params-1)
        #Create interpolation objects 
        lam_new_star=star_temp.lam*(1.+(v_raw-star_temp.v)/c)
        star_interp=interpolate.interp1d(lam_new_star, star_temp.flux)
        lam_new_aband=aband_temp.lam*(1.+v_aband/c)
        aband_interp=interpolate.interp1d(lam_new_aband, aband_temp.flux)
        #Initialize Lam and Flux Arrays
        lam=np.array([])
        flux=np.array([])
        for i in range(len(spec_reg)):
            if spec_reg[i]['Telluric']:
                lam_tmp, flux_tmp=make_line(spec_reg[i], c_h_arr[i], b_i[i], Spectrum.lam, aband_interp)
                lam=np.append(lam, lam_tmp)
                flux=np.append(flux, flux_tmp)
            else:
                lam_tmp, flux_tmp=make_line(spec_reg[i], c_h_arr[i], b_i[i], Spectrum.lam, star_interp)      
                lam=np.append(lam, lam_tmp)
                flux=np.append(flux, flux_tmp)
        self.lam=lam
        self.flux=flux 
        
    def likelihood(self, spectrum):
        noise_fit=((1./spectrum.ivar))**(1./2.)
        ln_like=norm.logpdf(spectrum.flux, loc=self.flux, scale=noise_fit)
        return np.sum(ln_like)
             
def make_line(spec_reg, c_h, b_i, lam0, interp_obj):
    mask=(lam0>spec_reg['range'][0])&((lam0<spec_reg['range'][1]))
    lam = lam0[mask]
    temp_flux=(interp_obj(lam)+c_h)/(1.+c_h)
    lam_eval=2.*(lam-np.min(lam))/np.max(lam-np.min(lam))-1.
    flux=temp_flux*np.polynomial.legendre.legval(lam_eval, b_i)
    return lam, flux 