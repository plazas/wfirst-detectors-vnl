#!/usr/bin/python
# sim1: simulate effect of B/F on shape measurement if it's not accounted for

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



try:
    import galsim
    from galsim.cdmodel import *
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
    from galsim.cdmodel import *


def secondmoments ( stamp ):
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


def m(delta_Ixx, delta_Iyy, r_gal):
    return - (delta_Ixx + delta_Iyy)/(r_gal**2)

def c1(delta_Ixx, delta_Iyy, r_gal):
    return (delta_Ixx - delta_Iyy)/(2*r_gal**2)

def c2(delta_Ixy, r_gal):
    return - (delta_Ixy)/(r_gal**2)



flux_vec=[1000, 3000, 5000, 10000, 15000] #,30000, 50000, 100000]
e1, e2 = float(sys.argv[1]), float(sys.argv[2])
m_nl_vec,c1_nl_vec,c2_nl_vec=[],[],[]
m_cd_vec,c1_cd_vec,c2_cd_vec=[],[],[]
f=lambda x: x - 3.566e-7*x*x

for flux in flux_vec:
    
    ## Gaussian PSF
    #create the object at given flux
    psf_flux=flux
    psf_e1, psf_e2= e1, e2
    psf_sigma=0.5/0.27/2.35
    psf=galsim.Gaussian (flux=flux, sigma=psf_sigma)
    psf = psf.shear(galsim.Shear(g1=psf_e1, g2=psf_e2))
    psf= psf.shift (0.5, 0.5)
    psf_image_original=psf.drawImage (scale=0.27)
    psf_image=psf_image_original
    psf_moments= secondmoments (psf_image_original)
    Ixx_psf=psf_moments [0]
    Iyy_psf=psf_moments [1]
    Ixy_psf=psf_moments [2]
    
    # Apply non-linearity
    psf_image.applyNonlinearity(f)
    nl_psf_moments= secondmoments (psf_image)
    Ixx_nl_psf=nl_psf_moments [0]
    Iyy_nl_psf=nl_psf_moments [1]
    Ixy_nl_psf=nl_psf_moments [2]

    m_var=m(Ixx_psf - Ixx_nl_psf,Iyy_psf - Iyy_nl_psf, psf_sigma)
    c1_var=c1(Ixx_psf - Ixx_nl_psf,Iyy_psf - Iyy_nl_psf, psf_sigma)
    c2_var=c2(Ixy_psf - Ixy_nl_psf, psf_sigma)

    m_nl_vec.append(m_var)
    c1_nl_vec.append(c1_var)
    c2_nl_vec.append(c2_var)
 
    # Apply BF from CCDs
    cd = PowerLawCD(5, 5.e-7, 5.e-7, 1.5e-7, 1.5e-7, 2.5e-7, 2.5e-7, 1.3)
    # a symmetric version similar to DECam or slightly stronger
    # note that this is not a particularly physical model, but similar enough for testing
    # for real simulations, use the actual fitted a coefficient with the base class

    # apply charge deflection
    psf_image_cd = cd.applyForward(psf_image_original)
    cd_psf_moments = secondmoments (psf_image_cd)
    Ixx_cd_psf=cd_psf_moments [0]
    Iyy_cd_psf=cd_psf_moments [1]
    Ixy_cd_psf=cd_psf_moments [2]

    m_var=m(Ixx_psf - Ixx_cd_psf,Iyy_psf - Iyy_cd_psf, psf_sigma)
    c1_var=c1(Ixx_psf - Ixx_cd_psf,Iyy_psf - Iyy_cd_psf, psf_sigma)
    c2_var=c2(Ixy_psf - Ixy_cd_psf, psf_sigma)

    m_cd_vec.append(m_var)
    c1_cd_vec.append(c1_var)
    c2_cd_vec.append(c2_var)

pp=PdfPages("mc.pdf")    
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

print flux_vec
print "NL: "
print "m: ", m_nl_vec
print "c1: ", c1_nl_vec
print "c2: ", c2_nl_vec
print "BF: "
print "m: ", m_cd_vec
print "c1: ", c1_cd_vec
print "c2: ", c2_cd_vec


#### Change PSF profile: optical? 

m_nl_opt_vec,c1_nl_opt_vec, c2_nl_opt_vec=[],[],[]
m_cd_opt_vec,c1_cd_opt_vec, c2_cd_opt_vec=[],[],[]

opt_defocus=0. #0.53       # wavelengths
opt_a1=0. #-0.29           # wavelengths
opt_a2=0. #0.12            # wavelengths
opt_c1=0. #0.64            # wavelengths
opt_c2=0. #-0.33           # wavelengths
opt_obscuration=0.4    # linear scale size of secondary mirror obscuration
lam = 1000              # nm    NB: don't use lambda - that's a reserved word.
tel_diam = 2.4          # meters
pixel_scale = 0.23     # arcsec / pixel

lam_over_diam = lam * 1.e-9 / tel_diam * galsim.radians
lam_over_diam = lam_over_diam / galsim.arcsec


for flux in flux_vec:
    ### Optical PSF 
    psf_flux=flux
    psf_e1, psf_e2= e1, e2
    psf_flux=flux
    psf_e1, psf_e2= e1, e2
    psf=galsim.OpticalPSF(lam_over_diam,
                               defocus = opt_defocus,
                               coma1 = opt_c1, coma2 = opt_c2,
                               astig1 = opt_a1, astig2 = opt_a2,
                               obscuration = opt_obscuration,
                               flux=psf_flux)
    psf = psf.shear(galsim.Shear(g1=psf_e1, g2=psf_e2))
    psf= psf.shift (0.5, 0.5)
    psf_image_original=psf.drawImage (scale=0.27)
    psf_image=psf_image_original
    psf_moments= secondmoments (psf_image_original)
    Ixx_psf=psf_moments [0]
    Iyy_psf=psf_moments [1]
    Ixy_psf=psf_moments [2]
    
    # Apply non-linearity
    psf_image.applyNonlinearity(f)
    nl_psf_moments= secondmoments (psf_image)
    Ixx_nl_psf=nl_psf_moments [0]
    Iyy_nl_psf=nl_psf_moments [1]
    Ixy_nl_psf=nl_psf_moments [2]

    m_var=m(Ixx_psf - Ixx_nl_psf,Iyy_psf - Iyy_nl_psf, psf_sigma)
    c1_var=c1(Ixx_psf - Ixx_nl_psf,Iyy_psf - Iyy_nl_psf, psf_sigma)
    c2_var=c2(Ixy_psf - Ixy_nl_psf, psf_sigma)

    m_nl_opt_vec.append(m_var)
    c1_nl_opt_vec.append(c1_var)
    c2_nl_opt_vec.append(c2_var)
 
    # Apply BF from CCDs
    cd = PowerLawCD(5, 5.e-7, 5.e-7, 1.5e-7, 1.5e-7, 2.5e-7, 2.5e-7, 1.3)
    # a symmetric version similar to DECam or slightly stronger
    # note that this is not a particularly physical model, but similar enough for testing
    # for real simulations, use the actual fitted a coefficient with the base class

    # apply charge deflection
    psf_image_cd = cd.applyForward(psf_image_original)
    cd_psf_moments = secondmoments (psf_image_cd)
    Ixx_cd_psf=cd_psf_moments [0]
    Iyy_cd_psf=cd_psf_moments [1]
    Ixy_cd_psf=cd_psf_moments [2]

    m_var=m(Ixx_psf - Ixx_cd_psf,Iyy_psf - Iyy_cd_psf, psf_sigma)
    c1_var=c1(Ixx_psf - Ixx_cd_psf,Iyy_psf - Iyy_cd_psf, psf_sigma)
    c2_var=c2(Ixy_psf - Ixy_cd_psf, psf_sigma)

    m_cd_opt_vec.append(m_var)
    c1_cd_opt_vec.append(c1_var)
    c2_cd_opt_vec.append(c2_var)

    
print "     "
print "     "
print "Optical PSF"
print flux_vec
print "NL: "
print "m: ", m_nl_opt_vec
print "c1: ", c1_nl_opt_vec
print "c2: ", c2_nl_opt_vec
print "BF: "
print "m: ", m_cd_opt_vec
print "c1: ", c1_cd_opt_vec
print "c2: ", c2_cd_opt_vec


###### PLOTS
## 1: m
## Plot parameters
plt.subplots_adjust(hspace=0.01, wspace=0.01)
prop = fm.FontProperties(size=10)
marker_size=11
loc_label = "upper left"
visible_x, visible_y = True, True
grid=True
ymin, ymax = -0.02, 0.02
m_req=1e-3
c_req=1e-4

fig = plt.figure()
ax = fig.add_subplot (111)
ax.errorbar( flux_vec, m_nl_vec, yerr=None, ecolor = 'r', label='Gaussian, vnl', fmt='r.-', markersize=marker_size)
ax.errorbar( flux_vec, m_cd_vec, yerr=None, ecolor = 'r', label='Gaussian, BF' , fmt='rx--', markersize=marker_size)
ax.errorbar( flux_vec, m_nl_opt_vec, yerr=None, ecolor = 'b', label='Optical, vnl', fmt='b.-', markersize=marker_size)
ax.errorbar( flux_vec, m_cd_opt_vec, yerr=None , ecolor = 'b', label='Optical, BF' , fmt='bx--', markersize=marker_size)
plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
lx=ax.set_xlabel(r"Flux", visible=visible_x)
#lx.set_fontsize(font_size)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ly=ax.set_ylabel(r"m", visible=visible_y)
#ly.set_fontsize(font_size)
ax.set_yscale('linear')
#plt.ylim ([ymin, ymax])
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
plt.grid(grid)
pp.savefig()
## 2: c1

fig = plt.figure()
ax = fig.add_subplot (111)
ax.errorbar( flux_vec, c1_nl_vec, yerr=None, ecolor = 'r', label='Gaussian, vnl', fmt='r.-', markersize=marker_size)
ax.errorbar( flux_vec, c1_cd_vec, yerr=None, ecolor = 'r', label='Gaussian, BF' , fmt='rx--', markersize=marker_size)
ax.errorbar( flux_vec, c1_nl_opt_vec, yerr=None, ecolor = 'b', label='Optical, vnl', fmt='b.-', markersize=marker_size)
ax.errorbar( flux_vec, c1_cd_opt_vec, yerr=None , ecolor = 'b', label='Optical, BF' , fmt='bx--', markersize=marker_size)
plt.axhspan(-c_req, c_req, facecolor='0.5', alpha=0.3)
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
lx=ax.set_xlabel(r"Flux", visible=visible_x)
#lx.set_fontsize(font_size)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ly=ax.set_ylabel(r"c1", visible=visible_y)
#ly.set_fontsize(font_size)
ax.set_yscale('linear')
#plt.ylim ([ymin, ymax])
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
plt.grid(grid)
pp.savefig()

## 3: c2


fig = plt.figure()
ax = fig.add_subplot (111)
ax.errorbar( flux_vec, c2_nl_vec, yerr=None, ecolor = 'r', label='Gaussian, vnl', fmt='r.-', markersize=marker_size)
ax.errorbar( flux_vec, c2_cd_vec, yerr=None, ecolor = 'r', label='Gaussian, BF' , fmt='rx--', markersize=marker_size)
ax.errorbar( flux_vec, c2_nl_opt_vec, yerr=None, ecolor = 'b', label='Optical, vnl', fmt='b.-', markersize=marker_size)
ax.errorbar( flux_vec, c2_cd_opt_vec, yerr=None , ecolor = 'b', label='Optical, BF' , fmt='bx--', markersize=marker_size)
plt.axhspan(-c_req, c_req, facecolor='0.5', alpha=0.3)
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
lx=ax.set_xlabel(r"Flux", visible=visible_x)
#lx.set_fontsize(font_size)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ly=ax.set_ylabel(r"c2", visible=visible_y)
#ly.set_fontsize(font_size)
ax.set_yscale('linear')
#plt.ylim ([ymin, ymax])
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
plt.grid(grid)
pp.savefig()

pp.close()



