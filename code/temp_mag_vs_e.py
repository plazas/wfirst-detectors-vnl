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



#### This code is just to plot  /Delta e / beta/ e  vs magnitude for NL paper. 

### Parameters
k=1000
base_size=1*k + 1 ## ??
n=8
m_zero=20  # 24
#m_gal=20
gal_sigma=0.1
print "gal_sigma", gal_sigma
pixel_scale=0.11
noise=20

type='nl'    # 'nl' or 'bf'
x_var='mag_and_ellipticity'   #'magnitude' or 'beta' or 'mag_and_beta'
profile_type='optical'   # 'gaussian' or 'optical'
label_type='lambda'  # 'lambda' or 'ellipticity'


#lam = 1380. #  NB: don't use lambda - that's a reserved word.
tel_diam = 2.4
obscuration_optical=0.3
beta0=3.566e-7

from collections import OrderedDict

e_vec=[ (-0.05, 0.), (-0.025, 0.), (0.,0.), (0.025, 0.), (0.05, 0.), (0., -0.05), (0., -0.025), (0., 0.025), (0., 0.05)]
mag_gal_vec=[18, 19, 20, 21, 22]
wavelength_dict=OrderedDict([('Z087',0.869), ('Y106',1.060), ('J129',1.293), ('W149',1.485), ('H158',1.577), ('F184',1.842)])  # in microns

alpha=0.6
"""
lam Z087
mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1] [18, 19, 20, 21, 22] [75742.870427579968, 28184.622313400596, 10930.675099636968, 4306.7322714014599, 1707.8251766452668] [18, 19, 20, 21, 22] [46734.377184175639, 17024.21550656106, 6550.3354381991485, 2573.0948113737395, 1018.6509485882863]
lam Y106
mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1] [18, 19, 20, 21, 22] [34010.730391860387, 12391.134553428625, 4769.7505454131397, 1873.769383607281, 742.35982843838917] [18, 19, 20, 21, 22] [16792.337753104806, 5842.1631550362963, 2210.2871739367233, 862.43111930308351, 340.69945229407904]
lam J129
mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1] [18, 19, 20, 21, 22] [2081.6495970849842, 610.11067968176383, 214.10345620322943, 80.857394587802432, 31.572887410428091] [18, 19, 20, 21, 22] [-4207.5081001661547, -1755.5098475711957, -707.1455227381914, -282.54122661853677, -112.62130721933471]
lam W149
mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1] [18, 19, 20, 21, 22] [-1592.3563997057356, -629.75642983168837, -249.94456341030767, -99.291328474215405, -39.518342722253216] [18, 19, 20, 21, 22] [-2290.9654828275197, -903.30454718955991, -358.36928529396317, -142.08097811688063, -56.513617906103555]
lam H158
mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1] [18, 19, 20, 21, 22] [-6150.5703896285277, -2378.3714611299974, -935.60571825874524, -370.61280809129659, -147.20284187141385] [18, 19, 20, 21, 22] [-7708.1960180413962, -2965.5113576244357, -1164.6449589701208, -461.15279359937051, -182.91858877565352]
lam F184
mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1] [18, 19, 20, 21, 22] [-3699.0422391040042, -1440.0794503965601, -568.07319186230689, -225.35603566550293, -89.752006470152082] [18, 19, 20, 21, 22] [-4015.0815292432935, -1561.8699682539068, -616.18170001934766, -244.46452736574233, -97.058721196402871]
"""

pp=PdfPages("test_bias_NL_vs_flux.pdf")
print "Output PDF: test_bias_NL_vs_flux.pdf"
#### PLOTS
#### Do the plotting here
plt.minorticks_on()
#plt.tight_layout()

### We do not have matplotlib 1.1, with the 'style' package. Modify the matplotlibrc file parameters instead
import matplotlib as mpl
mpl.rc('lines', linewidth=1, color='black', linestyle='-')
mpl.rc('font', family='serif',weight='normal', size=10.0 )
mpl.rc('text',  color='black', usetex=False)
mpl.rc('axes',  edgecolor='black', linewidth=1, grid=False, titlesize=11, labelsize=11, labelweight='normal',labelcolor='black')
mpl.rc('axes.formatter', limits=[-4,4])
mpl.rcParams['xtick.major.size']=7
mpl.rcParams['xtick.minor.size']=4
mpl.rcParams['xtick.major.pad']=8
mpl.rcParams['xtick.minor.pad']=8
mpl.rcParams['xtick.labelsize']= '11'
mpl.rcParams['xtick.minor.width']= 1.0
mpl.rcParams['xtick.major.width']= 1.0
mpl.rcParams['ytick.major.size']=7
mpl.rcParams['ytick.minor.size']=4
mpl.rcParams['ytick.major.pad']=8
mpl.rcParams['ytick.minor.pad']=8
mpl.rcParams['ytick.labelsize']= '11'
mpl.rcParams['ytick.minor.width']= 1.0
mpl.rcParams['ytick.major.width']= 1.0
mpl.rc ('legend', numpoints=1, fontsize='11', shadow=False, frameon=False)

## Plot parameters
plt.subplots_adjust(hspace=0.01, wspace=0.01)
prop = fm.FontProperties(size=9)
marker_size=7
loc_label = "upper right"
visible_x, visible_y = True, True
grid=False
ymin, ymax = -0.0001, 0.0001
m_req=1e-3
c_req=1e-4

color_vec=['r', 'y', 'g', 'c', 'b', 'm', 'k']
#color_dict={0.0:'r', 0.025:'k', 0.05:'b', 0.075:'m', 0.08:'c', 0.1:'g'}
color_vec_lam=['m','b', 'c', 'g', 'y', 'r']



color_dict_e={}
for i,e in enumerate(e_vec):
    color_dict_e[e]=color_vec[i%len(color_vec)]

color_dict_mag={}
for i,m_gal in enumerate(mag_gal_vec):
    color_dict_mag[m_gal]=color_vec[i%len(color_vec)]


color_dict_lam={}
for i,lam in enumerate(wavelength_dict):
    color_dict_lam[lam]=color_vec_lam[i%len(color_vec_lam)]


x_vec=mag_gal_vec
x_label=r"mag_object"


string= r"Non-linearity: $f=x-\beta x^{2}$ " + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g/%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale, n)

def plot_function_e_and_r (fig, x_vec, y1_vec, y2_vec, y3_vec, x1label='', x2label='', y1label=r"$\Delta$e", y2label=r"$\Delta$R/R", lam_key='H158', e_key=(0.0, 0.0)):
    
    if len(x2label) == 0:
        x2label=x1label
    
    
    if label_type == 'lambda':
        color_fmt=color_dict_lam[lam_key]
        label_e1=''
        label_e2=''
        label='%s'%(lam_key)
    elif label_type == 'ellipticity':
        color_fmt=color_dict_lam[lam_key]
        label_e1='%g' %e_key[0]
        label_e2='%g' %e_key[1]
        label='(e1,e2)=(%g,%g)'%(e_key[0], e_key[1])
    else:
        print "wrong label type."
        sys.exit(1)
    
    ax = fig.add_subplot (211)
    ax.errorbar( x_vec, y1_vec, yerr=None, ecolor = color_fmt, label=label_e1, fmt=color_fmt+'s-', markersize=marker_size, alpha=alpha)
    #ax.errorbar( x_vec, y2_vec, yerr=None, ecolor = color_fmt, label=label_e2, fmt=color_fmt+'x-', markersize=marker_size, alpha=alpha)
    plt.axhline(y=0.,color='k',ls='solid')
    if  e_key[0] == 0.0 and e_key[1] == 0.0 and label_type == 'ellipticity':
        plt.axhline(y=1e-5, color='r',ls='-', label='1e-5') # requirement
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
    lx=ax.set_xlabel(x1label, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
    ly=ax.set_ylabel(y1label, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('linear')
    #plt.ylim ([-1e-4, 8e4])
    #plt.ylim ([ymin, ymax])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin - 0.03*delta, xmax + 0.03*delta])
    if label_type == 'ellipticity':
        plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=11)
    #if plot_pos== 321:
    ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
    
    #plt.grid(grid, which='both', ls='-', alpha=0.5)
    plt.grid(grid)


    ax = fig.add_subplot (212)
    if  e_key[0] == 0.0 and e_key[1] == 0.0:
        ax.errorbar( x_vec, y3_vec, yerr=None, ecolor = color_fmt, label=label, fmt=color_fmt+'o-', markersize=marker_size, alpha=alpha)
    #ax.errorbar( x_vec, theory_delta_r_gauss, yerr=None, ecolor = 'k', label='theory Gauss', fmt='r-', markersize=marker_size, alpha=1.)
    plt.axhline(y=0.,color='k',ls='solid')
    if  e_key[0] == 0.0 and e_key[1] == 0.0 and lam_key == 'H158' and label_type == 'ellipticity':
        plt.axhline(y=1e-4, color='r',ls='-', label='1e-4') # requirement
    if x_var == 'magnitude' and profile_type == 'gaussian':
            ax.errorbar(x_vec, ratio_vec, yerr=None, ecolor = 'b', label='Theory', fmt='bo-', markersize=marker_size, alpha=alpha)
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
    lx=ax.set_xlabel(x2label, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
    ly=ax.set_ylabel(y2label, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('linear')
    #plt.ylim ([-1e-4, 8e4])
    #plt.ylim ([ymin, ymax])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin - 0.03*delta, xmax + 0.03*delta])
    #if profile_type=='optical':
    #    plt.ylim ([0., 0.040])
    #    plt.xlim ([17.5, 24.5])
    #plt.ylim([0., 0.18e-4])
    if label_type == 'ellipticity':
        plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=11)
    #if plot_pos== 324:
    ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
    """
        #Inset with zoom
        subpos = [0.45, 0.45, 0.475, 0.35]
        subax1 = add_subplot_axes(ax,subpos)
        if  e_key[0] == 0.0 and e_key[1] == 0.0 and not x_var == 'mag_and_beta':  # does not depend on e, just plot once
        subax1.plot (x_vec, y3_vec, color_fmt+'o-', markersize=marker_size, alpha=alpha)
        if profile_type == 'gaussian':
        subax1.plot (x_vec, ratio_vec,'bo-', markersize=marker_size, alpha=alpha)
        subax1.axhline(y=1e-4, color='r',ls='-')
        subax1.axhline(y=0.,color='k',ls='solid')
        subax1.set_yticklabels(subax1.get_yticks(), size=9, visible=True)
        subax1.set_xticklabels(subax1.get_xticks(), size=9, visible=True)
        plt.ylim([-1e-4, 3e-4])
        if profile_type == 'optical':
        plt.xlim ([21, 24.5])
        #else:
        #    plt.xlim ([21.8, 24.2])
        #    subax1.set_yticklabels(subax1.get_yticks(), size=5, visible=True)
        #    subax1.set_xticklabels(subax1.get_xticks(), size=5, visible=True)
    """




a=[75742.870427579968, 28184.622313400596, 10930.675099636968, 4306.7322714014599, 1707.8251766452668]
b=[46734.377184175639, 17024.21550656106, 6550.3354381991485, 2573.0948113737395, 1018.6509485882863]

c=[34010.730391860387, 12391.134553428625, 4769.7505454131397, 1873.769383607281, 742.35982843838917]
d=[16792.337753104806, 5842.1631550362963, 2210.2871739367233, 862.43111930308351, 340.69945229407904]

e=[2081.6495970849842, 610.11067968176383, 214.10345620322943, 80.857394587802432, 31.572887410428091]
f=[-4207.5081001661547, -1755.5098475711957, -707.1455227381914, -282.54122661853677, -112.62130721933471]

g=[-1592.3563997057356, -629.75642983168837, -249.94456341030767, -99.291328474215405, -39.518342722253216]
h=[-2290.9654828275197, -903.30454718955991, -358.36928529396317, -142.08097811688063, -56.513617906103555]

i=[-6150.5703896285277, -2378.3714611299974, -935.60571825874524, -370.61280809129659, -147.20284187141385]
j=[-7708.1960180413962, -2965.5113576244357, -1164.6449589701208, -461.15279359937051, -182.91858877565352]

k=[-3699.0422391040042, -1440.0794503965601, -568.07319186230689, -225.35603566550293, -89.752006470152082]
l=[-4015.0815292432935, -1561.8699682539068, -616.18170001934766, -244.46452736574233, -97.058721196402871]




slope_dict2={'Z087':[a,b], 'Y106':[c,d], 'J129':[e,f], 'W149':[g,h], 'H158':[i,j], 'F184':[k,l]}

fig = plt.figure()
for lam in slope_dict2:
    plot_function_e_and_r (fig, mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1], x1label=x_label, y1label=r"$\Delta e_1$/$\beta$/$e_1$", y2label=r"$\Delta e_2$/$\beta$/$e_2$", lam_key=lam)
plt.suptitle(string, fontsize=13)
fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig(fig)
plt.close()
pp.close()
