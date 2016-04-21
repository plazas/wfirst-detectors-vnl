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
## Simple code to test hsm and interleave method
import galsim
import galsim.wfirst as wfirst
filters = wfirst.getBandpasses (AB_zeropoint=True)

import logging
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("tests_hsm_interleaving")

def measurement_function (profile, true_e1=0, true_e2=0, true_s=0, e1_vec=[], e2_vec=[], size_vec=[], e1_inter_vec=[], e2_inter_vec=[], size_inter_vec=[], noise=None, string=''):

    true_size=true_s
    print "    "
    print "n: ", n
    logger.info (string)
    im=galsim.Image(base_size, base_size)
    gal=profile
    image=gal.drawImage(image=im, scale=pixel_scale/n, method='no_pixel')
    logger.info("Image shape, before interleave: (%g, %g)", image.array.shape[0], image.array.shape[1] )
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise/(n**2))
        image.addNoise(read_noise)
    results=image.FindAdaptiveMom(hsmparams=new_params)

    e1_vec.append ( (results.observed_shape.e1 - true_e1))
    e2_vec.append ( (results.observed_shape.e2 - true_e2))
    size_vec.append(results.moments_sigma - true_size)
    
    
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
            gal.drawImage(image=im, scale=pixel_scale, offset=offset, method='no_pixel')
            im_list.append(im)

    image=galsim.utilities.interleaveImages(im_list=im_list, N=(n,n), offsets=offsets_list)
    print "Image shape, after interleave: ", image.array.shape
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise)
        image.addNoise(read_noise)
    results=image.FindAdaptiveMom(hsmparams=new_params)
    e1_inter_vec.append  (np.abs(results.observed_shape.e1 - e))
    e2_inter_vec.append  (np.abs(results.observed_shape.e2 - e))
    size_inter_vec.append(results.moments_sigma - gal_sigma)




base_size = 2048 ## ??
n=4
gal_flux=1.e7
gal_sigma=1
pixel_scale=0.11
noise=20


lam = 1490. #float(filters['W149'])              # nm    NB: don't use lambda - that's a reserved word.
tel_diam = 2.4
lam_over_diam = lam * 1.e-9 / tel_diam * galsim.radians
lam_over_diam = lam_over_diam / galsim.arcsec
print "lam_over_diam", lam_over_diam


e_vec=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07] #, 0.05, 0.06, 0.07, 0.08]

#gauss
g_e1, g_e2, g_s, gi_e1, gi_e2, gi_s = [],[],[],[],[],[]
ng_e1, ng_e2, ng_s, ngi_e1, ngi_e2, ngi_s = [],[],[],[],[],[]
# Optical
o_e1, o_e2, o_s, oi_e1, oi_e2, oi_s = [],[],[],[],[],[]
no_e1, no_e2, no_s, noi_e1, noi_e2, noi_s = [],[],[],[],[],[]
# WFIRST
w_e1, w_e2, w_s, wi_e1, wi_e2, wi_s = [],[],[],[],[],[]
nw_e1, nw_e2, nw_s, nwi_e1, nwi_e2, nwi_s = [],[],[],[],[],[]


new_params = galsim.hsm.HSMParams(max_amoment=5.0e7, max_mom2_iter=100000)




for e in e_vec:
    
        logger.info("   ")
        logger.info("ellipticity: %g", e)
        # Gaussian
        # no noise
        #logger.info("First loop: gaussian, no noise")

        #gal= galsim.Convolve (galsim.Gaussian(flux=gal_flux, sigma=gal_sigma).shear(galsim.Shear(e1=e,e2=e)) , galsim.Pixel(pixel_scale) )
        #measurement_function (gal, true_e1=e, true_e2=e, true_s=gal_sigma, e1_vec=g_e1, e2_vec=g_e2, size_vec=g_s, e1_inter_vec=gi_e1, e2_inter_vec=gi_e2, size_inter_vec=gi_s, noise=None, string='Gaussian, no noise')
        # noise
        #measurement_function (gal, true_e1=e, true_e2=e, true_s=gal_sigma, e1_vec=ng_e1, e2_vec=ng_e2, size_vec=ng_s, e1_inter_vec=ngi_e1, e2_inter_vec=ngi_e2, size_inter_vec=ngi_s, noise=noise, string='Gaussian, noise')
    
        #Optical
        gal=galsim.Convolve (galsim.OpticalPSF(lam_over_diam, obscuration = 0.4, flux=gal_flux).shear(galsim.Shear(e1=e,e2=e)), galsim.Pixel(pixel_scale) )
        measurement_function (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=o_e1, e2_vec=o_e2, size_vec=o_s, e1_inter_vec=oi_e1, e2_inter_vec=oi_e2, size_inter_vec=oi_s, noise=None, string='Optical, no noise')
        # noise
        measurement_function (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=no_e1, e2_vec=no_e2, size_vec=no_s, e1_inter_vec=noi_e1, e2_inter_vec=noi_e2, size_inter_vec=noi_s, noise=noise, string='Optical, noise')
 
        #WFIRST
        #gal=wfirst.getPSF(SCAs=7,approximate_struts=True, wavelength=filters['W149'])[7].shear(galsim.Shear(e1=e, e2=e))
        #measurement_function (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=w_e1, e2_vec=w_e2, size_vec=w_s, e1_inter_vec=wi_e1, e2_inter_vec=wi_e2, size_inter_vec=wi_s, noise=None, string='WFIRST, no noise')
        # noise
        #measurement_function (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=nw_e1, e2_vec=nw_e2, size_vec=nw_s, e1_inter_vec=nwi_e1, e2_inter_vec=nwi_e2, size_inter_vec=nwi_s, noise=noise, string='WFIRST, noise')



pp=PdfPages("test_bias.pdf")
#### PLOTS
#### Do the plotting here
plt.minorticks_on()
#plt.tight_layout()

### We do not have matplotlib 1.1, with the 'style' package. Modify the matplotlibrc file parameters instead
import matplotlib as mpl
mpl.rc('lines', linewidth=1, color='black', linestyle='-')
mpl.rc('font', family='serif',weight='normal', size=10.0 )
mpl.rc('text',  color='black', usetex=False)
mpl.rc('axes',  edgecolor='black', linewidth=1, grid=False, titlesize='x-large', labelsize='x-large', labelweight='normal',labelcolor='black')
mpl.rc('axes.formatter', limits=[-4,4])
mpl.rcParams['xtick.major.size']=7
mpl.rcParams['xtick.minor.size']=4
mpl.rcParams['xtick.major.pad']=8
mpl.rcParams['xtick.minor.pad']=8
mpl.rcParams['xtick.labelsize']= 'x-large'
mpl.rcParams['xtick.minor.width']= 1.0
mpl.rcParams['xtick.major.width']= 1.0
mpl.rcParams['ytick.major.size']=7
mpl.rcParams['ytick.minor.size']=4
mpl.rcParams['ytick.major.pad']=8
mpl.rcParams['ytick.minor.pad']=8
mpl.rcParams['ytick.labelsize']= 'x-large'
mpl.rcParams['ytick.minor.width']= 1.0
mpl.rcParams['ytick.major.width']= 1.0
mpl.rc ('legend', numpoints=1, fontsize='x-large', shadow=False, frameon=False)

## Plot parameters
plt.subplots_adjust(hspace=0.01, wspace=0.01)
prop = fm.FontProperties(size=10)
marker_size=11
loc_label = "upper left"
visible_x, visible_y = True, True
grid=True
ymin, ymax = -0.0001, 0.0001
m_req=1e-3
c_req=1e-4


def plot_function (x_vec, y1_vec, y2_vec, y3_vec, y4_vec, string=''):
    fig = plt.figure()
    ax = fig.add_subplot (111)
    ax.errorbar( x_vec, y1_vec, yerr=None, ecolor = 'r', label='$\Delta$ e1. Sample=0.11/%g'%n, fmt='r.-', markersize=marker_size)
    ax.errorbar( x_vec, y2_vec, yerr=None, ecolor = 'g', label='$\Delta$ e2', fmt='g.-', markersize=marker_size)
    #ax.errorbar( e_vec, size_vec, yerr=None, ecolor = 'b', label='$\Delta$ size', fmt='b.-', markersize=marker_size)
    ax.errorbar( x_vec, y3_vec, yerr=None, ecolor = 'r', label='$\Delta$ e1 inter. N=(%g,%g)'%(n,n), fmt='rx--', markersize=marker_size)
    ax.errorbar( x_vec, y4_vec, yerr=None, ecolor = 'g', label='$\Delta$ e2 inter', fmt='gx--', markersize=marker_size)
    #ax.errorbar( g_vec, size_inter_vec, yerr=None, ecolor = 'b', label='$\Delta$ size inter' , fmt='bx--', markersize=marker_size)
    #ax.errorbar( flux_vec, delta_size_vec, yerr=None, ecolor = 'b', label='$\Delta$ r, vnl', fmt='b.-', markersize=marker_size)
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
    lx=ax.set_xlabel(r"e", visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
    ly=ax.set_ylabel(r"$\Delta$e", visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('linear')
    #plt.ylim ([ymin, ymax])
    plt.title(string)
    ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
    plt.grid(grid, which='both', ls='-')
    pp.savefig()

#plot_function (e_vec, g_e1, g_e2, gi_e1, gi_e2, string="Gaussian * Pixel, no noise. Sample: 0.11" )
#plot_function (e_vec, ng_e1, ng_e2, ngi_e1, ngi_e2, string="Gaussian * Pixel, noise" )
plot_function (e_vec, o_e1, o_e2, oi_e1, oi_e2, string="Optical * Pixel, no noise" )
plot_function (e_vec, no_e1, no_e2, noi_e1, noi_e2, string="Optical * Pixel, noise" )
#plot_function (e_vec, w_e1, w_e2, wi_e1, wi_e2, string="WFIRST, no noise" )
#plot_function (e_vec, nw_e1, nw_e2, nwi_e1, nwi_e2, string="WFIRST, noise" )




pp.close()





