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

import galsim
import galsim.wfirst as wfirst
filters = wfirst.getBandpasses (AB_zeropoint=True)

import logging
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("tests_hsm_interleaving")



f=lambda x,beta : x - beta*x*x

gal_sigma=0.1
pixel_scale=0.11


def measurement_function (profile, e1_inter_vec=[], e2_inter_vec=[], size_inter_vec=[], noise=None, string='', type='nl'):
    beta0=3.566e-7
    
    ### First, measure moments *without* NL applied, at given flux
    
    image=profile.drawImage(image=galsim.Image(base_size, base_size), scale=pixel_scale/n, method='no_pixel')
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise/(n**2))
        image.addNoise(read_noise)
    results=image.FindAdaptiveMom(hsmparams=new_params)
    ref_e1=results.observed_shape.e1
    ref_e2=results.observed_shape.e2
    ref_s=results.moments_sigma

    ## Interleave the profiles
    im_list=[]
    offsets_list=[]
    #create list of images to interleave-no effect
    for j in xrange(n):
        for i in xrange(n):
            im=galsim.Image(base_size, base_size)
            offset=galsim.PositionD(-(i+0.5)/n+0.5, -(j+0.5)/n+0.5)
            offsets_list.append(offset)
            print "Offset: ", offset
            profile.drawImage(image=im, scale=pixel_scale, offset=offset, method='no_pixel')
            if type == 'bf':
                #cd = PowerLawCD(5, 5.e-7, 5.e-7, 1.5e-7, 1.5e-7, 2.5e-7, 2.5e-7, 1.3)
                cd = BaseCDModel (aL,aR,aB,aT)
                im=cd.applyForward(im)
            elif type == 'nl':
                im.applyNonlinearity(f,beta0)
            else:
                print "wrong type: 'bf' or 'nl' "
                sys.exit(1)
            im_list.append(im)

    image=galsim.utilities.interleaveImages(im_list=im_list, N=(n,n), offsets=offsets_list)
    print "Image shape, after interleave: ", image.array.shape
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise)
        image.addNoise(read_noise)
    results=image.FindAdaptiveMom(hsmparams=new_params)
    e1_inter_vec.append  (results.observed_shape.e1 - ref_e1)
    e2_inter_vec.append  (results.observed_shape.e2 - ref_e2)
    size_inter_vec.append(  (results.moments_sigma - ref_s) / ref_s)


pp=PdfPages ('test.pdf')



def ratio (sigma,  beta, flux_vec):
    sigma/=pixel_scale   ### Convert to pixels?
    return  ( (8*math.pi - beta*flux_vec/(sigma**2) ) / (8*math.pi - 2*beta*flux_vec/(sigma**2)  ) )   - 1



def flux (mag_vec):
	return 8.81631e1*2.521**(20-mag_vec)


mag_vec= np.array([18,19,20,21,22,23,24])

flux_vec = flux (mag_vec)

beta=3.566e-7

ratio_vec= ratio (gal_sigma, beta, flux_vec)
assert (len(ratio_vec) == len (mag_vec))


##### Now measure the same thing in GalSim



### Parameters

base_size=1024 ## ??
n=3
m_zero=20  # 24
#m_gal=20
print "gal_sigma", gal_sigma
noise=20
e=0.0

type='nl'



#lam = 1380. #  NB: don't use lambda - that's a reserved word.
tel_diam = 2.4
obscuration_optical=0.4
#lam_over_diam = lam * 1.e-9 / tel_diam * galsim.radians
#lam_over_diam = lam_over_diam / galsim.arcsec
#print "lam_over_diam", lam_over_diam
beta0=3.566e-7


#Define wavelengths, ellipticities, and magnitudes
new_params = galsim.hsm.HSMParams(max_amoment=6.0e7, max_mom2_iter=10000000)
big_fft_params = galsim.GSParams(maximum_fft_size=4096)
mag_gal_vec= [18,19,20,21,22,23,24]

e1_vec, e2_vec, r_measured_vec = [],[],[]

print "mag_vec, ratio_vec", mag_vec, ratio_vec

for mag in mag_gal_vec:
    gal_flux=6e4*2.521**(m_zero-mag)
    gal= galsim.Convolve (galsim.Gaussian(flux=gal_flux, sigma=gal_sigma).shear(galsim.Shear(e1=0.0,e2=0.0)),galsim.Pixel(pixel_scale), gsparams=big_fft_params )
    measurement_function (gal, e1_inter_vec=e1_vec, e2_inter_vec=e2_vec, size_inter_vec=r_measured_vec, noise=None, string='Gausian, no noise', type=type)

prop = fm.FontProperties(size=7)
alpha=0.6
fig = plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(mag_vec, ratio_vec, yerr=None, ecolor = 'b', label='theory', fmt='bo-', markersize=11, alpha=alpha)
ax.errorbar(mag_vec, r_measured_vec, yerr=None, ecolor = 'b', label='Galsim. Pixel scale: %g arcsec/pixel' %pixel_scale, fmt='ro-', markersize=11, alpha=alpha)
ax.legend(loc='upper right' , fancybox=True, ncol=1, numpoints=1, prop = prop)
plt.xlabel('mag_object')
plt.ylabel(r"$\frac{\Delta R}{R}$")
pp.savefig()
pp.close()





