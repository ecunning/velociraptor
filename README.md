# velociraptor
Hierarchical modeling of the velocity of an star from multiple spectroscopic observations

Velociraptor implements a Bayesian hierarchical model in order to estimate the velocity of a star from multiple spectroscopic observations. This model is discussed in detail in [Cunningham et al. (2018a)](https://ui.adsabs.harvard.edu//#abs/2018arXiv180904082C/abstract). 

As written, Velociraptor is designed to read in spec1d files as produced by the DEIMOS spec2d pipeline. If you are not working with DEIMOS data, please modify the "Spectrum" object in spectrum_class.py.
