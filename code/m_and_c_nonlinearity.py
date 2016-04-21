#!/usr/bin/python

import numpy as np
import os
import sys
import math
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf  import PdfPages
import matplotlib.font_manager as fm

## 6-1-15
## Simple code to explore NL as a function of beta, by using interleaving method


import logging
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("tests_hsm_interleaving")


from galsim.cdmodel import *
from sim2 import *   ## where all the BF stuff is
from scipy import optimize


from measurement_function import *

import galsim.wfirst as wfirst
filters = wfirst.getBandpasses (AB_zeropoint=True)

#Those plots use mock galaxies in addition to PSF; the galaxies are Gaussian light profiles and the PSFs are Airy with obscuration. I apply a shear to the galaxies, convolve with the PSF, draw onto an image, apply IPC ( NL in your case ) and measure the intrinsic shape of the galaxy using galsim.hsm.EstimateShear. By applying the same shear over many galaxies of different ellipticities, the average of the recovered ellipticities will give you an unbiased estimate of shear. I am comparing this estimated shear against the applied shear and fitting the data to a straight line gives m,c.


### Parameters
k=1000
base_size=1*k + 1 ## ??  # galsim.Image(base_size, base_size)
n=8
m_zero=20  # 24
#m_gal=20
gal_sigma=0.1
print "gal_sigma", gal_sigma
pixel_scale=0.11
noise=20

type='nl'    # 'nl' or 'bf'
x_var='beta'   #'magnitude' or 'beta' or 'mag_and_beta'
profile_type='wfirst'   # 'gaussian', 'optical', or 'wfirst'
label_type='lambda'  # 'lambda' or 'ellipticity'


#lam = 1380. #  NB: don't use lambda - that's a reserved word.
tel_diam = 2.4
obscuration_optical=0.3
beta0=3.566e-7


from collections import OrderedDict

#Define wavelengths, ellipticities, and magnitudes
wavelength_dict=OrderedDict([('Z087',0.869), ('Y106',1.060)  , ('J129',1.293) , ('W149',1.485), ('H158',1.577), ('F184',1.842)])  # in microns
#wavelength_dict=OrderedDict([('H158',1.577), ('F184',1.842)])
flux_dict={'Z087':8.57192e4,'Y106':8.68883e4 , 'J129':8.76046e4, 'W149':2.68738e4, 'H158':8.81631e4, 'F184':6.08258e4}
#e_vec=[ (0., 0.), (0.05, 0.), (0.05, 0.05) ] #, (0., 0.075), (0.075, 0.), (0.075, 0.075)] #, 0.05, 0.06, 0.07, 0.08]
new_params = galsim.hsm.HSMParams(max_amoment=6000000, max_mom2_iter=1000000,  max_moment_nsig2=10000)
big_fft_params = galsim.GSParams(maximum_fft_size=4096)



gal_flux = 3.e2    # counts
gal_r0 = 1.0       # arcsec
g1 = 0.2           #
g2 = 0.1           #


gal = galsim.Gaussian(flux=gal_flux, scale_radius=gal_r0)
gal = gal.shear(g1=g1, g2=g2)


psf=wfirst.getPSF(SCAs=7,approximate_struts=True, wavelength=filters[lam])[7].shear(galsim.Shear(e1=e[0],e2=e[1])).withFlux(gal_flux)



final = galsim.Convolve([gal, psf])

image_epsf = psf.drawImage(scale=pixel_scale/n)  # Oversampling
image_epsf*=n*n   ###  need to adjust flux per pixel
image_epsf.applyNonlinearity(f,beta)









