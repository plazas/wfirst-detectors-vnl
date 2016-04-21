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

## 7-20-15
## Simple code to explore NL as a function of beta, by using interleaving method
import galsim
import galsim.wfirst as wfirst
filters = wfirst.getBandpasses (AB_zeropoint=True)

import logging
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("tests_hsm_interleaving")

f=lambda x,beta : x - beta*x*x

from galsim.cdmodel import *
from sim2 import *   ## where all the BF stuff is
from scipy import optimize

def measurement_function_NL (profile, e1_inter_vec=[], e2_inter_vec=[], size_inter_vec=[], noise=None, beta=3.566e-7, string='', type='nl'):
    
    print "    "
    print "n: ", n
    logger.info (string)
    print "beta: ", beta
    
    #profile_before=profile.withScaledFlux(n*n)
    
    
    """
    #### Calculate moments without effect
    print "FLUX: ", profile.getFlux()
    image=profile.drawImage(image=galsim.Image(base_size, base_size), scale=pixel_scale/n, method='no_pixel')
    print  np.amax(image.array), np.sum(image.array)
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise/(n**2))
        image.addNoise(read_noise)
    results=image.FindAdaptiveMom(hsmparams=new_params)
    ref_e1=results.observed_shape.e1
    ref_e2=results.observed_shape.e2
    ref_s=results.moments_sigma
    print "Image shape, before interleave: ", image.array.shape
    ##Create fits image in disk
    file_name="object_before_interleaving.fits"
    image.write(file_name)
    print "ref_e1, ref_e2, ref_s", ref_e1, ref_e2, ref_s

    #### Calculate moments with the effect

    image=profile.drawImage(image=galsim.Image(base_size, base_size), scale=pixel_scale/n, method='no_pixel')
    print  np.amax(image.array), np.sum(image.array)
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise/(n**2))
        image.addNoise(read_noise)
    image*=n*n
    image.applyNonlinearity(f,beta)
    results=image.FindAdaptiveMom(hsmparams=new_params)
    print "results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma ", results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma
    print "Differences: ", results.observed_shape.e1 - ref_e1, results.observed_shape.e2 - ref_e2, (results.moments_sigma - ref_s) / ref_s

    """

    ## Interleave the profiles with NO EFFECT
    im_list=[]
    offsets_list=[]
    #create list of images to interleave-no effect
    
    
    for j in xrange(n):
        for i in xrange(n):
            im=galsim.Image(base_size, base_size)
            offset=galsim.PositionD(-(i+0.5)/n+0.5, -(j+0.5)/n+0.5)
            offsets_list.append(offset)
            #print "Offset: ", offset
            profile.drawImage(image=im, scale=pixel_scale, offset=offset, method='no_pixel')
            im_list.append(im)

    image=galsim.utilities.interleaveImages(im_list=im_list, N=(n,n), offsets=offsets_list, add_flux=True)
    print "Image shape, after interleave, no effect: ", image.array.shape
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise)
        image.addNoise(read_noise)
    results=image.FindAdaptiveMom(hsmparams=new_params)
    ref_e1=results.observed_shape.e1
    ref_e2=results.observed_shape.e2
    ref_s=results.moments_sigma
    print "ref_e1, ref_e2, ref_s", ref_e1, ref_e2, ref_s
    file_name="object_after_interleaving_n_%g_no_effect.fits" %n
    image.write(file_name)



    ## Interleave the profiles with the effect
    im_list=[]
    offsets_list=[]
    #create list of images to interleave-no effect
    for j in xrange(n):
        for i in xrange(n):
            im=galsim.Image(base_size, base_size)
            offset=galsim.PositionD(-(i+0.5)/n+0.5, -(j+0.5)/n+0.5)
            offsets_list.append(offset)
            #print "Offset: ", offset
            profile.drawImage(image=im, scale=pixel_scale, offset=offset, method='no_pixel')
            if type == 'bf':
                #cd = PowerLawCD(5, 5.e-7, 5.e-7, 1.5e-7, 1.5e-7, 2.5e-7, 2.5e-7, 1.3)
                cd = BaseCDModel (aL,aR,aB,aT)
                im=cd.applyForward(im)
            elif type == 'nl':
                #print "Skipping NL"
                im.applyNonlinearity(f,beta)
            else:
                print "wrong type: 'bf' or 'nl' "
                sys.exit(1)
            im_list.append(im)


    image2=galsim.utilities.interleaveImages(im_list=im_list, N=(n,n), offsets=offsets_list, add_flux=True)
    print "Image shape, after interleave: ", image2.array.shape
    if not noise == None:
        read_noise = galsim.GaussianNoise(sigma=noise)
        image2.addNoise(read_noise)
    results=image2.FindAdaptiveMom(hsmparams=new_params)
    e1_inter_vec.append  (results.observed_shape.e1 - ref_e1)
    e2_inter_vec.append  (results.observed_shape.e2 - ref_e2)
    size_inter_vec.append ( (results.moments_sigma - ref_s) / ref_s)
    print "results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma ", results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma
    print "Differences: ", results.observed_shape.e1 - ref_e1, results.observed_shape.e2 - ref_e2, (results.moments_sigma - ref_s) / ref_s
    file_name="object_after_interleaving_n_%g_effect.fits" %n
    image2.write(file_name)

    ##Take the difference of the images
    diff=image-image2
    print "diff: ", diff
    file_name="diff_object_interleaving_n_%g.fits" %n
    diff.write(file_name)



### Parameters

k=1024
base_size=1*k + 1## ??
n=7
m_zero=20  # 24
#m_gal=20
#gal_flux=6e4*2.521**(m_zero-m_gal)
gal_sigma=0.1
print "gal_sigma", gal_sigma
pixel_scale=0.11
noise=20
#e=0.0

type='nl'


#if type == 'bf':
#    (aL,aR,aB,aT) = readmeanmatrices()


#lam = 1380. #  NB: don't use lambda - that's a reserved word.
tel_diam = 2.4
obscuration_optical=0.3
#lam_over_diam = lam * 1.e-9 / tel_diam * galsim.radians
#lam_over_diam = lam_over_diam / galsim.arcsec
#print "lam_over_diam", lam_over_diam



#Define wavelengths, ellipticities, and magnitudes
wavelength_dict={'Z087':0.869,'Y106':1.060, 'J129':1.293, 'W149':1.485, 'H158':1.577, 'F184':1.842}  # in microns
#wavelength_dict={'Y106':1.060, 'H158':1.577}
flux_dict={'Z087':8.57192e4,'Y106':8.68883e4, 'J129':8.76046e4, 'W149':2.68738e4, 'H158':8.81631e4, 'F184':6.08258e4}
#e_vec=[ (0., 0.), (0.05, 0.),  (0., 0.05), (0.05, 0.05) ]#, (0., 0.075), (0.075, 0.), (0.075, 0.075)] #, 0.05, 0.06, 0.07, 0.08]
#e_vec= [(-0.05, 0.), (-0.025, 0.), (0.0, 0.0), (0.05, 0.), (0.025, 0.), (0.0, -0.05), (0.0, -0.025), (0.0, 0.025), (0.0, 0.05)]

#e_vec= [ (-0.05, 0.), (-0.025, 0.), (0.0, 0.0), (0.025, 0.), (0.05, 0.) ]

e_vec=[(0.0, 0.0)]

new_params = galsim.hsm.HSMParams(max_amoment=60000000, max_mom2_iter=10000000,  max_moment_nsig2=10000)
big_fft_params = galsim.GSParams(maximum_fft_size=4096)
m_gal_vec= [18, 19, 20, 21, 22,23,24]


beta0=3.566e-7
beta_vec=[ 1.5*beta0, 0.5*beta0, beta0, 1.5*beta0, 2*beta0, 5*beta0]

#print beta_vec

#sys.exit()

#vectors that will hold the output to plot
gauss_no_noise={}
optical_no_noise={}
gauss_noise={}
optical_noise={}

gauss_no_noise={}
optical_no_noise={}
gauss_noise={}
optical_noise={}
for lam in wavelength_dict:
    gauss_no_noise[lam]={}   #\Delta e1, \Delta e2, \Delta R/R
    optical_no_noise[lam]={}
    gauss_noise[lam]={}
    optical_noise[lam]={}
    #for e in e_vec:
    #    gauss_no_noise[lam][e]=[[],[],[]]   #\Delta e1, \Delta e2, \Delta R/R
    #    optical_no_noise[lam][e]=[[],[],[]]
    #    gauss_noise[lam][e]=[[],[],[]]
    #    optical_noise[lam][e]=[[],[],[]]

    for m_gal in m_gal_vec:
        gauss_no_noise[lam][m_gal]=[[],[],[]]   #\Delta e1, \Delta e2, \Delta R/R
        optical_no_noise[lam][m_gal]=[[],[],[]]
        gauss_noise[lam][m_gal]=[[],[],[]]
        optical_noise[lam][m_gal]=[[],[],[]]



#for e in e_vec:
#    gauss_no_noise[e]=[[],[],[]]
#    optical_no_noise[e]=[[],[],[]]
#    gauss_noise[e]=[[],[],[]]
#    optical_noise[e]=[[],[],[]]

#for m_gal in m_gal_vec:
#    gauss_no_noise[m_gal]=[[],[],[]]
#    optical_no_noise[m_gal]=[[],[],[]]
#    gauss_noise[m_gal]=[[],[],[]]
#    optical_noise[m_gal]=[[],[],[]]


#for e in [e1_true]:   ### Just one value of e1=0.01. Not really a nested loop.


for lam in wavelength_dict:
    lam_over_diam = wavelength_dict[lam] * 1.e-6 / tel_diam * galsim.radians
    lam_over_diam = lam_over_diam / galsim.arcsec
    for e in e_vec:
        for m_gal in m_gal_vec:
          for beta in beta_vec:
            logger.info("   ")
            logger.info("ellipticity: (%g, %g)", e[0], e[1] )
            logger.info("lambda: %s microns",  wavelength_dict[lam])
            logger.info("beta: %g", beta)
            logger.info("magnitude: %g", m_gal)

            # Gaussian
            # no noise
            #logger.info("First loop: gaussian, no noise")
        
            gal_flux=flux_dict[lam]*2.512**(m_zero-m_gal)
            #gal= galsim.Convolve (galsim.Gaussian(flux=gal_flux, sigma=gal_sigma).shear(galsim.Shear(e1=e[0],e2=e[1])) , galsim.Pixel(pixel_scale), gsparams=big_fft_params )
            #gal.shift (0.5, 0.5)
            #measurement_function_NL (gal,  e1_inter_vec=gauss_no_noise[lam][m_gal][0], e2_inter_vec=gauss_no_noise[lam][m_gal][1], size_inter_vec=gauss_no_noise[lam][m_gal][2], noise=None, beta=beta, string='Gausian, no noise')
            #sys.exit(1)
            ###### noise
            #measurement_function_NL (gal,  e1_inter_vec=gauss_noise[m_gal][0], e2_inter_vec=gauss_noise[m_gal][1], size_inter_vec=gauss_noise[m_gal][2], noise=noise, beta=beta, string='Gaussian, noise')

            #######################Optical
        
            #logger.info("Third loop: Optical, no noise")
            gal=galsim.Convolve (galsim.OpticalPSF(lam_over_diam, obscuration=obscuration_optical, flux=gal_flux).shear(galsim.Shear(e1=e[0],e2=e[1])), galsim.Pixel(pixel_scale), gsparams=big_fft_params  )
            #gal.shift (0.5, 0.5)
            measurement_function_NL (gal, e1_inter_vec=optical_no_noise[lam][m_gal][0], e2_inter_vec=optical_no_noise[lam][m_gal][1], size_inter_vec=optical_no_noise[lam][m_gal][2], noise=None, beta=beta, string='Optical, no noise')

            sys.exit(1)
            ###### noise
            #measurement_function_NL (gal,  e1_inter_vec=optical_noise[m_gal][0], e2_inter_vec=optical_noise[m_gal][1], size_inter_vec=optical_noise[m_gal][2], noise=noise, beta=beta, string='Optical, noise')

            #########################WFIRST
            #gal=wfirst.getPSF(SCAs=7,approximate_struts=True, wavelength=filters['W149'])[7].shear(galsim.Shear(e1=e, e2=e))
            #measurement_function_NL (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=w_e1, e2_vec=w_e2, size_vec=w_s, e1_inter_vec=wi_e1, e2_inter_vec=wi_e2, size_inter_vec=wi_s, noise=None, string='WFIRST, no noise')
            # noise
            #measurement_function_NL (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=nw_e1, e2_vec=nw_e2, size_vec=nw_s, e1_inter_vec=nwi_e1, e2_inter_vec=nwi_e2, size_inter_vec=nwi_s, noise=noise, string='WFIRST, noise')



#factor_vec=xrange(1,11)
#for e in [e_vec[1]]:
#    for factor in factor_vec:



pp=PdfPages("test_bias_NL_vs_beta.pdf")
print "Name of output PDF file: test_bias_NL_vs_beta.pdf"
#### PLOTS
#### Do the plotting here
plt.minorticks_on()
#plt.tight_layout()

### We do not have matplotlib 1.1, with the 'style' package. Modify the matplotlibrc file parameters instead
import matplotlib as mpl
mpl.rc('lines', linewidth=1, color='black', linestyle='-')
mpl.rc('font', family='serif',weight='normal', size=10.0 )
mpl.rc('text',  color='black', usetex=False)
mpl.rc('axes',  edgecolor='black', linewidth=1, grid=False, titlesize=9, labelsize=11, labelweight='normal',labelcolor='black')
mpl.rc('axes.formatter', limits=[-4,4])
mpl.rcParams['xtick.major.size']=7
mpl.rcParams['xtick.minor.size']=4
mpl.rcParams['xtick.major.pad']=8
mpl.rcParams['xtick.minor.pad']=8
mpl.rcParams['xtick.labelsize']= '12'
mpl.rcParams['xtick.minor.width']= 1.0
mpl.rcParams['xtick.major.width']= 1.0
mpl.rcParams['ytick.major.size']=7
mpl.rcParams['ytick.minor.size']=4
mpl.rcParams['ytick.major.pad']=8
mpl.rcParams['ytick.minor.pad']=8
mpl.rcParams['ytick.labelsize']= '12'
mpl.rcParams['ytick.minor.width']= 1.0
mpl.rcParams['ytick.major.width']= 1.0
mpl.rc ('legend', numpoints=1, fontsize='12', shadow=False, frameon=False)

## Plot parameters
plt.subplots_adjust(hspace=0.01, wspace=0.01)
prop = fm.FontProperties(size=11)
marker_size=7
loc_label = "upper right"
visible_x, visible_y = True, True
grid=False
ymin, ymax = -0.0001, 0.0001
m_req=1e-3
c_req=1e-4


color_vec=['r', 'y', 'g', 'c', 'b', 'm', 'k']
#color_dict={0.0:'r', 0.025:'k', 0.05:'b', 0.075:'m', 0.08:'c', 0.1:'g'}

color_dict_e={}
for i,e in enumerate(e_vec):
    color_dict_e[e]=color_vec[i%len(color_vec)]

color_dict_m={}
for i,m_gal in enumerate(m_gal_vec):
    color_dict_m[m_gal]=color_vec[i%len(color_vec)]

color_vec_lam=['m','b', 'c', 'g', 'y', 'r']
color_dict_lam={}
for i,lam in enumerate(wavelength_dict):
    color_dict_lam[lam]=color_vec_lam[i%len(color_vec_lam)]

alpha=1.0
plot_positions_six={'Z087':321,'Y106':322, 'J129':323, 'W149':324, 'H158':325, 'F184':326}


## Theory for Delta R / R
#def theory_size_gauss (sigma,  beta, flux_vec):
#    sigma/=(pixel_scale)   ### Convert to pixels?
#    return  ( (8*math.pi - beta*flux_vec/(sigma**2) ) / (8*math.pi - 2*beta*flux_vec/(sigma**2)  ) )   - 1
#flux_vec=flux_dict['H158']*2.512**( m_zero - np.array(mag_gal_vec) )
#ratio_vec= theory_size_gauss (gal_sigma, beta_vec, flux_vec  )




def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def plot_function_e_and_r (fig, x1_vec, y1_vec, x2_vec, y2_vec, xlabel1='', xlabel2='', ylabel1=r"$\Delta$e", ylabel2=r"$\Delta$R/R", lam_key='', e_key=(0.0, 0.0), m_key='', label_bool=False):
    color_fmt=color_dict_lam[lam_key]
    #plot_pos=plot_positions_six[lam_key]
    #label='e=(%g, %g)' %(e_key[0], e_key[1])
    label=lam_key
    
    
    #print "x1_vec, y1_vec, x2_vec, y2_vec", x1_vec, y1_vec, x2_vec, y2_vec
    
    #fig = plt.figure()
    ax = fig.add_subplot (211)
    ax.errorbar( x1_vec, y1_vec, yerr=None, ecolor = color_fmt, label=label, fmt=color_fmt+'s-', markersize=marker_size, alpha=alpha)
    #ax.errorbar( x_vec, y2_vec, yerr=None, ecolor = color_fmt, label='e2=%g'%e_key[1], fmt=color_fmt+'x-', markersize=marker_size, alpha=alpha)
    plt.axhline(y=0.,color='k',ls='solid')
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
    lx=ax.set_xlabel(xlabel1, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
    ly=ax.set_ylabel(ylabel1, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('linear')
    #plt.ylim ([ymin, ymax])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
    #plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=11)
    #if plot_pos== 321:
    if label_bool:
        ax.legend(loc=loc_label , fancybox=True, ncol=2, numpoints=1, prop = prop)
    
    #plt.grid(grid, which='both', ls='-', alpha=0.5)
    plt.grid(grid)
    
    
    ax = fig.add_subplot (212)
    ax.errorbar( x2_vec, y2_vec, yerr=None, ecolor = color_fmt, label=label, fmt=color_fmt+'o-', markersize=marker_size, alpha=alpha)
    #ax.errorbar( x_vec, theory_delta_r_gauss, yerr=None, ecolor = 'k', label='theory Gauss', fmt='r-', markersize=marker_size, alpha=1.)
    plt.axhline(y=0.,color='k',ls='solid')
    #if label_bool:
        #plt.axhline(y=1e-4, color='r',ls='-', label='1e-4') # requirement
    #ax.errorbar(x_vec, ratio_vec, yerr=None, ecolor = 'b', label='Theory', fmt='bo-', markersize=marker_size, alpha=alpha)
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
    lx=ax.set_xlabel(xlabel2, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
    ly=ax.set_ylabel(ylabel2, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('log')
    plt.ylim ([10, 2e5])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
    #if type=='o':
    #plt.ylim ([0., 0.026])
    #plt.ylim([0., 0.18e-4])
    #plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=11)
    #if plot_pos== 324:
    if label_bool:
        ax.legend(loc=loc_label , fancybox=True, ncol=2, numpoints=1, prop = prop)
    #Inset with zoom
    #subpos = [0.35, 0.30, 0.475, 0.35]
    #subax1 = add_subplot_axes(ax,subpos)
    #subax1.plot (x_vec, y3_vec, color_fmt+'o-', markersize=marker_size, alpha=alpha)
    #subax1.plot (x_vec, ratio_vec,'bo-', markersize=marker_size, alpha=alpha)
    #subax1.axhline(y=1e-4, color='r',ls='--')
    #plt.ylim([-1e-4, 3e-4])
    #if type == 'o':
    #    plt.xlim ([22, 24.5])
    #else:
    #    plt.xlim ([21.8, 24.2])
    #    subax1.set_yticklabels(subax1.get_yticks(), size=5, visible=True)
    #    subax1.set_xticklabels(subax1.get_xticks(), size=5, visible=True)







def plot_function_e (fig, x_vec, y1_vec, y2_vec, string='', xlabel='', y1label=r"$\Delta$e", label_string='', lam_key='', e_key=(0.0,0.0), m_key=''):
    color_fmt=color_dict_mag[m_key]
    plot_pos=plot_positions_six[lam_key]
    label='e1=%g, m=%g'%(e_key,m_key)
    #label='e1=%g'%e_key
    label2='e2=%g, m=%g'%(e_key,m_key)
    #label2='e2=%g'%e_key
    
    ax = fig.add_subplot (plot_pos)
    ax.errorbar( x_vec, y1_vec, yerr=None, ecolor = color_fmt, label=label, fmt=color_fmt+'s-', markersize=marker_size, alpha=alpha)
    ax.errorbar( x_vec, y2_vec, yerr=None, ecolor = color_fmt, label=label2, fmt=color_fmt+'x-', markersize=marker_size, alpha=alpha)
    plt.axhline(y=0.,color='k',ls='solid')
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
    lx=ax.set_xlabel(xlabel, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
    ly=ax.set_ylabel(y1label, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('linear')
    #plt.ylim ([ymin, ymax])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin-0.03*delta, xmax + 0.03*delta])
    plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=10)
    if plot_pos== 321:
        ax.legend(loc=loc_label , fancybox=True, ncol=2, numpoints=1, prop = prop)
    #plt.grid(grid, which='both', ls='-', alpha=0.5)
    plt.grid(grid)



def plot_function_r (fig, x_vec, y3_vec, xlabel='', y2label=r"$\Delta$R/R", lam_key='',m_key='', e_key=0.0, type='o'):
    color_fmt=color_dict_mag [m_key]
    plot_pos=plot_positions_six[lam_key]
    ax = fig.add_subplot (plot_pos)
    label='m=%g'%(m_key)
    #label='e1=e2=%g'%(e_key)
    
    ax.errorbar( x_vec, y3_vec, yerr=None, ecolor = color_fmt, label=label, fmt=color_fmt+'o-', markersize=marker_size, alpha=alpha)



    plt.axhline(y=0.,color='k',ls='solid')
    plt.axhline(y=1e-4, color='r',ls='--') # requirement
    plt.axvline(x=beta0, color='b',ls='--') # nominal beta
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
    lx=ax.set_xlabel(xlabel, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
    ly=ax.set_ylabel(y2label, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('linear')
    #plt.ylim ([ymin, ymax])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin-0.03*delta, xmax + 0.06*delta])
    if type=='o':
        plt.ylim ([0., 0.009])
    plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=10)
    if plot_pos== 324:
        ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)

    #plt.grid(grid, which='both', ls='-', alpha=0.5)
    
    #Inset with zoom
    subpos = [1-0.275, 0.15, 0.29, 0.375]
    if type == 'o':
        subpos = [1-0.275, 0.25, 0.29, 0.375]
    subax1 = add_subplot_axes(ax,subpos)
    subax1.plot ( x_vec, y3_vec, color_fmt+'o-', label=label, markersize=marker_size, alpha=alpha)
    subax1.axhline(y=1e-4, color='r',ls='--')
    plt.ylim([0., 3e-4])
    if type == 'o':
        plt.xlim ([7e-9, 3e-8])
    else:
        plt.xlim ([1e-8, 6e-8])

    subax1.set_yticklabels(subax1.get_yticks(), size=3.9, visible=True)
    subax1.set_xticklabels(subax1.get_xticks(), size=3.9, visible=True)
    if plot_pos in [322,324,326]:
        subax1.yaxis.set_label_position("right")
        subax1.yaxis.tick_right()
        subax1.set_yticklabels(subax1.get_yticks(), size=4, visible=True)
        subax1.set_xticklabels(subax1.get_xticks(), size=3.9, visible=True)

    #pp.savefig()


plot_positions_six={'Z087':321,'Y106':322, 'J129':323, 'W149':324, 'H158':325, 'F184':326}


if type == 'bf':
    string_g= "BF: BaseCDModel" + "\n" + "Gaussian ($\sigma$=%g'')* Pixel (0.11 arcsec/pix), no noise." %(gal_sigma)
    string_o= "BF: BaseCDModel" + "\n" + "Optical (tel_diam=%g m, obscuration=%g) * Pixel (0.11 arcsec/pix), no noise. "%(tel_diam, obscuration_optical)
elif type == 'nl':
    string_g= r"Non-linearity: $f=x-\beta x^{2}$ " + "\n" + "Gaussian ($\sigma$=%g'') * Pixel (0.11 arcsec/pix), no noise." %(gal_sigma)
    string_o= r"Non-linearity: $f=x-\beta x^{2}$ " + "\n" + "Optical (tel_diam=%g m, obscuration=%g) * Pixel (0.11 arcsec/pix), no noise. "%(tel_diam, obscuration_optical)
else:
    print "invalid type (nor 'bf' nor 'nl')"
    sys.exit(1)

# + r"($\beta$=%g)" %(beta0)


def get_slope (x, y):
    fitfunc = lambda p, x: p[0]*x
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    p0 = [1.]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(x,y))
    print 'pendiente:', p1[0]
    return p1[0]


dic_list=[optical_no_noise]
#e_vec_temp=[]
#for var in e_vec:
#    e_vec_temp.append(var[0])

for dictionary in dic_list:

    beta_vec=np.array(beta_vec)
    slope_dict={}
    for lam in wavelength_dict:
        slope_dict[lam] =[[],[]]   #slope_e1, slope_r
    ## Gaussian no noise, Delta_e, one filter
    fig = plt.figure()
    for lam in wavelength_dict:
        for m_gal in m_gal_vec:
            slope_e1= get_slope ( beta_vec, dictionary[lam][m_gal][0])  #delta_e1
            slope_dict[lam][0].append(slope_e1)
        
            slope_r= get_slope (beta_vec, dictionary[lam][m_gal][2])    #delta_r
            slope_dict[lam][1].append(slope_r)



    for lam in wavelength_dict:
        print "lam", lam
        plot_function_e_and_r (fig, m_gal_vec, slope_dict[lam][0] , m_gal_vec, slope_dict[lam][1], xlabel1='$m$', xlabel2='$m$', ylabel1=r"$\Delta e_1$/$\beta$", ylabel2=r"$\Delta R/R/\beta$", lam_key=lam, m_key=m_gal, label_bool=True)
    plt.suptitle(string_o, fontsize=13)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    pp.savefig(fig)
    plt.close()




"""

## Gaussian no noise, Delta_e, all filters
fig = plt.figure()
for lam in wavelength_dict:
    for e in e_vec:  # One single value
        for m_gal in m_gal_vec:
            plot_function_e (fig, beta_vec , gauss_no_noise[lam][m_gal][0],gauss_no_noise[lam][m_gal][1], xlabel=r"$\beta$", lam_key=lam, e_key=e, m_key=m_gal)
string="Gaussian($\sigma$=%g'')*Pixel, no noise. " %(gal_sigma) +r"$f=x-\beta x^{2}$"+"\n Object flux: gal_flux=6e4*2.521**(%g-%g)" %( m_zero, m_gal)
plt.suptitle(string, fontsize=11)
fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig(fig)


## Gaussian no noise, Delta_R/R, all filters
fig = plt.figure()
for lam in wavelength_dict:
    for e in e_vec:  # One single value
        for m_gal in m_gal_vec:
            plot_function_r (fig, beta_vec , gauss_no_noise[lam][m_gal][2], xlabel=r"$\beta$", lam_key=lam, e_key=e, m_key=m_gal)

string="Gaussian($\sigma$=%g'')*Pixel, no noise. " %(gal_sigma) +r"$f=x-\beta x^{2}$"+"\n Object flux: gal_flux=6e4*2.521**(%g-%g)" %( m_zero, m_gal)
plt.suptitle(string, fontsize=11)
fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig(fig)


## Optical no noise, Delta_e, all filters
fig = plt.figure()
for lam in wavelength_dict:
    for e in e_vec:  # One single value
        for m_gal in m_gal_vec:
            plot_function_e (fig, beta_vec, optical_no_noise[lam][m_gal][0], optical_no_noise[lam][m_gal][1], xlabel=r"$\beta$", lam_key=lam, e_key=e, m_key=m_gal)

string="Optical(tel_diam=%g m)*Pixel, no noise. "%(tel_diam) + r"$f=x-\beta x^{2}$," +"\n Object flux: gal_flux=6e4*2.521**(%g-%g)" %(m_zero, m_gal)
plt.suptitle(string, fontsize=11)
fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig(fig)


## Optical no noise, Delta_R/R, all filters
fig = plt.figure()
for lam in wavelength_dict:
    for e in e_vec:  # One single value
        for m_gal in m_gal_vec:
            plot_function_r (fig, beta_vec , optical_no_noise[lam][m_gal][2], xlabel=r"$\beta$", lam_key=lam, e_key=e, m_key=m_gal, type='o')

string="Optical(tel_diam=%g m)*Pixel, no noise. "%(tel_diam) + r"$f=x-\beta x^{2}$," + "\n Object flux: gal_flux=6e4*2.521**(%g-%g)" %(m_zero, m_gal)
plt.suptitle(string, fontsize=11)
fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig(fig)

"""


"""
fig=plt.figure()
for e in e_vec:
    for m_gal in m_gal_vec:
        plot_function (fig,beta_vec, gauss_noise[m_gal][0],gauss_noise[m_gal][1],gauss_noise[m_gal][2], string="Gaussian*Pixel, noise. " +r"$f=x-\beta x^{2}$", xlabel=r"$\beta$", e_key=e, m_key=m_gal)
pp.savefig(fig)



fig=plt.figure()
for e in e_vec:
    for m_gal in m_gal_vec:
        plot_function (fig, beta_vec, optical_no_noise[m_gal][0], optical_no_noise[m_gal][1], optical_no_noise[m_gal][2], string="Optical($\lambda$=%g nm, tel_diam=%g m)*Pixel, no noise. "%(lam,tel_diam) +r"$f=x-\beta x^{2}$" , xlabel=r"$\beta$", e_key=e, m_key=m_gal)
pp.savefig(fig)


fig=plt.figure()
for e in e_vec:
    for m_gal in m_gal_vec:
        plot_function (fig, beta_vec,  optical_noise[m_gal][0],optical_noise[m_gal][1],optical_noise[m_gal][2], string="Optical*Pixel, noise. " +r"$f=x-\beta x^{2}$" , xlabel=r"$\beta$", e_key=e, m_key=m_gal)
pp.savefig(fig)
"""

pp.close()







