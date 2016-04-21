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


import logging
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("tests")


from galsim.cdmodel import *
#from sim2 import *   ## where all the BF stuff is
from scipy import optimize


#from measurement_function import *

import galsim.wfirst as wfirst
filters = wfirst.getBandpasses (AB_zeropoint=True)


def make2DGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)



def secondmoments ( stamp, fwhm=3):
    # calculates second moments and returns list [ q11 q22 q12 ]
    # this is less elegant than it could be
    # stamp needs to be a background-subtracted image
    moments = [ 0, 0, 0, 0, 0, 0 ] # q11 q22 q12 norm x y
    centx = (stamp.xmin+stamp.xmax+1)/2.
    centy = (stamp.ymin+stamp.ymax+1)/2.
    
    #create gaussian stamp
    assert (stamp.xmin+stamp.xmax+1 == stamp.ymin+stamp.ymax+1)
    gauss_stamp=make2DGaussian (stamp.xmin+stamp.xmax+1, fwhm, center=[centx,centy])
    
    for x in range(stamp.xmin,stamp.xmax+1):
        for y in range(stamp.ymin,stamp.xmax+1):
            moments[0] += stamp.at(x,y)*(x-centx)*(x-centx).gauss_stamp.at(x,y)         # q11
            moments[1] += stamp.at(x,y)*(y-centy)*(y-centy).gauss_stamp.at(x,y)         # q22
            moments[2] += stamp.at(x,y)*(x-centx)*(y-centy).gauss_stamp.at(x,y)         # q12
            moments[3] += stamp.at(x,y).gauss_stamp.at(x,y)
            moments[4] += stamp.at(x,y)*(x-centx).gauss_stamp.at(x,y)
            moments[5] += stamp.at(x,y)*(y-centy).gauss_stamp.at(x,y)
    moments[0] /= moments[3]
    moments[1] /= moments[3]
    moments[2] /= moments[3]
    moments[4] /= moments[3]
    moments[5] /= moments[3]

    return moments



def secondmoments_gauss ( stamp ):
    # calculates second moments and returns list [ q11 q22 q12 ]
    # this is less elegant than it could be
    # stamp needs to be a background-subtracted image
    moments = [ 0, 0, 0, 0, 0, 0 ] # q11 q22 q12 norm x y
    centx = (stamp.xmin+stamp.xmax+1)/2.
    centy = (stamp.ymin+stamp.ymax+1)/2.
    for x in range(stamp.xmin,stamp.xmax+1):
        for y in range(stamp.ymin,stamp.xmax+1):
            moments[0] += stamp.at(x,y)*(x-centx)*(x-centx)         # q11
            moments[1] += stamp.at(x,y)*(y-centy)*(y-centy)         # q22
            moments[2] += stamp.at(x,y)*(x-centx)*(y-centy)         # q12
            moments[3] += stamp.at(x,y)
            moments[4] += stamp.at(x,y)*(x-centx)
            moments[5] += stamp.at(x,y)*(y-centy)
    moments[0] /= moments[3]
    moments[1] /= moments[3]
    moments[2] /= moments[3]
    moments[4] /= moments[3]
    moments[5] /= moments[3]

    return moments




# Purpose: Loop over all WFIRST PSFs and measure their ellipticities, to find the maximum ellipticity in the focal plane.
# Date: 10-20-15
# Paralelized using multiprocessing Nov-4-2015


k=64
base_size=1.5*k ## ??  # galsim.Image(base_size, base_size)
n=3

m_zero=18.3  # 24
mag=18.3
pixel_scale=wfirst.pixel_scale

from collections import OrderedDict

#Define wavelengths, ellipticities, and magnitudes
#wavelength_dict=OrderedDict([('Z087',0.869), ('Y106',1.060)  , ('J129',1.293) , ('W149',1.485), ('H158',1.577), ('F184',1.842)])  # in microns
#flux_dict={'Z087':8.57192e4,'Y106':8.68883e4 , 'J129':8.76046e4, 'W149':2.68738e4, 'H158':8.81631e4, 'F184':6.08258e4}

#wavelength_dict=OrderedDict([('J129',1.292), ('Y106',1.060), ('H158',1.577), ('F184',1.837)]) 
#flux_dict={'Y106':9.39189e4 , 'J129':9.5477e4, 'H158':9.5178e4, 'F184':7.1792e4}


wavelength_dict=OrderedDict([('J129',1.292) , ('Y106',1.060), ('H158',1.577), ('F184',1.837)])  
flux_dict={}
for bandpass in ['Y106', 'J129', 'H158', 'F184']:
    f=filters[bandpass]
    b=galsim.sed.SED(f)
    c=b.withMagnitude(m_zero, f)
    flux_dict[bandpass]=c.calculateFlux(f)

print flux_dict

new_params = galsim.hsm.HSMParams(max_amoment=6000000, max_mom2_iter=1000000,  max_moment_nsig2=10000)
big_fft_params = galsim.GSParams(maximum_fft_size=4096)


d={}
for k in range(18*4):
    d[k]=[0.,0.]
i=0
for lam in wavelength_dict:
    for sca in range (1,19):
        d[i][0], d[i][1]=lam, sca
        i+=1

import moments

def wfirst_psf_function (index):  #takes only one argument
    lam, sca = d[index][0], d[index][1]
    gal_flux=flux_dict[lam]*10**(0.4*(m_zero-mag))
    gal=galsim.Convolve(wfirst.getPSF(SCAs=sca,approximate_struts=False, wavelength=filters[lam])[sca].withFlux(gal_flux), galsim.Pixel(pixel_scale))
        
    image=gal.drawImage(image=galsim.Image(base_size, base_size), scale=pixel_scale/n, method='no_pixel')
    #image.write('wf_psf.fits')
    #sys.exit(1)
    
    #hsm
    results=image.FindAdaptiveMom(hsmparams=new_params)
    ref_e1=results.observed_shape.e1
    ref_e2=results.observed_shape.e2
    ref_e=math.sqrt (ref_e1**2 + ref_e2**2)
    ref_s=results.moments_sigma

    #unweighted moments
    #moments= secondmoments_gauss (image)
    #q11, q22, q12= moments[0],moments[1],moments[2]
    #denom=q11 + q22 + 2.*math.sqrt(q11*q22-q12*q12)
    #ref_e1=(q11-q22)/denom
    #ref_e2= (2*q12)/denom
    #ref_e=math.sqrt (ref_e1**2 + ref_e2**2)
    #ref_s=1.
    
    #Chaz's moments
    #ref_e1, ref_e2, ref_s, xc, yc, dxc, dyc = moments.measure_e1e2R2(image.array, skysub=False, sigma=128)
    #ref_e=math.sqrt (ref_e1**2 + ref_e2**2)
    #print "lam, sca, ref_e1, ref_e2, ref_e, ref_s",   lam, sca, ref_e1, ref_e2, ref_e, ref_s

    return lam, sca, ref_e1, ref_e2, ref_e, ref_s


#for lam in wavelength_dict:
#    for sca in range (1,19):
#        ref_e1, ref_e2, ref_e, ref_s = wfirst_psf_function (lam, sca)
#        print "lam, SCA, e1, e2, e, size: ", sca, lam, ref_e1, ref_e2, ref_e, ref_s



## Try multiprocessing
from multiprocessing import cpu_count
from multiprocessing import Pool

processes=cpu_count()
use=6
print "I have ", processes, "cores here. Using: %g" %use
pool = Pool(processes=6)

print "starting: "
results=pool.map (wfirst_psf_function, range(18*4))
for line in results:
    print "lam, SCA, e1, e2, e, size: ", line



















