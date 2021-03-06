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


from multiprocessing import cpu_count
from multiprocessing import Pool



#def measurement_function (profile, e1_inter_vec=[], e2_inter_vec=[], size_inter_vec=[], noise=None, string='', type='nl', beta=beta0, ):
    #beta0=3.566e-7


### Parameters
k=64
base_size=1.5*k  ## ??  # galsim.Image(base_size, base_size)
n=3
m_zero=18.3  # 24
#m_gal=20
gal_sigma=0.1
#print "gal_sigma", gal_sigma
pixel_scale=0.11
#noise=20

type='nl'    # 'nl' or 'bf'
x_var='mag_and_beta'   #'magnitude' or 'beta' or 'mag_and_beta'
profile_type='wfirst'   # 'gaussian', 'optical', or 'wfirst'
label_type='lambda'  # 'lambda' or 'ellipticity'


dispersion=True  # To make Fig2, dispersion delta vs dispersion beta distribution


#lam = 1380. #  NB: don't use lambda - that's a reserved word.
tel_diam = 2.4
obscuration_optical=0.3
#beta0=-3.566e-7
beta0=5e-7
#beta0=1e-6
#gamma0=1e-10
#delta0=-2.0e-15
#beta0=0.5e-6
#gamma0=-0.25e-10
#delta0=1.5e-15
#all_params=(beta0, gamma0, delta0)
#all_params = (beta0, 0., 0.)

### Set one of the following to 1, and the other to whatever you want
if dispersion:
    n_realizations=100  ### parameter to control if betas per pixel are drawn from Gaussians. Set to 1 for 'no'.
else:
    n_realizations=1

n_offsets=1   ## How many realizatons over centroid shift. At least 1.


sqrt_n=math.sqrt(n_realizations)
sqrt_offsets=math.sqrt(n_offsets)


multi=False #multiprocessing
if multi:
    from multiprocessing import cpu_count
    processes=cpu_count()
    print "I have ", processes, "cores here"





if x_var == 'magnitude':# for Delta R/R vs m at fixed beta
    #beta0=0.5e-6
    #gamma0=-0.25e-10
    #delta0=1.5e-15
    #beta_means=np.array([ (beta0, gamma0,delta0) ] )
    #beta_means=(beta0, gamma0,delta0)
    #beta_sigma=0.12*beta_means
    beta_vec=beta_means
    #mag_gal_vec= [20, 21, 22,23] #,24]
    mag_gal_vec= [18.3, 19, 20, 21, 22] #,24]

    
    e_vec=[ (0., 0.)]#, (0.05, 0.), (0.05, 0.05) ]
elif x_var == 'beta':
    #beta0 = delta0
    beta_means= np.array([ 1e-8*beta0, 0.5*beta0, beta0, 2*beta0, 2.5*beta0] )

    if dispersion:
        beta_means = [1e-8*beta0, 0.1*beta0, 0.5*beta0 , beta0, 2*beta0]  # actually, beta_sigma, not beta
    
    
    #beta_means= np.linspace (-0.5e-6, 1.5e-6, 6)   #beta
    #beta_means= np.linspace (-1.0e-10, 0.5e-10, 6)  #gamma
    #beta_means= np.linspace (-1.0e-15, 2.0e-15, 6)   #delta
    beta_means= np.array([ beta0 ])

    #beta_sigma=5.9e-7   # beta
    #beta_sigma=5.625e-11  #gamma
    #beta_sigma=1.35e-15 # delta

    beta_vec=beta_means
    mag_gal_vec= [m_zero]# , 20] #, 22, 24]
    e_vec=[ (0., 0.)]  #    , (0.05, 0.), (0.05, 0.05) ]
elif x_var == 'mag_and_beta':   # when normalizing deltas by beta, and reporting slope (/delta R/ R/ beta) vs magnitude
    #beta0=0.5e-6  # midpoint from ramge in hilbert 2014
    #gamma0=-0.25e-10
    #delta0=1.5e-15
    #Each pixel has its onw beta, drawn from a Gaussian distribution
    beta_means= np.array([ (1-0.01)*beta0, beta0, beta0*(1 + 0.01)])#   , 2*beta0, 5*beta0] )
    #beta_means= np.linspace (gamma0 - 0.01e-10, gamma0 + 0.01e-10, 5)
    #beta_means= np.linspace (delta0 - 0.01e-15, delta0 + 0.01e-15, 5)

    if dispersion:
        beta_means = [1e-8*beta0, 0.5*beta0 , beta0]  # actually, beta_sigma, not beta> point to calculate slope



    #beta_sigma=0.12*beta_means
    beta_vec=[]
    beta_vec=beta_means
    mag_gal_vec= [18.3, 19, 20, 21, 22]
    #if dispersion:
        #mag_gal_vec = [ 1e-8*beta0, 0.5*beta0, beta0, 2*beta0]  # actually beta


    e_vec=[ (0., 0.)]

elif x_var == 'ellipticity':
    beta_vec=[ 0., beta0, 1.5*beta0, 5*beta0]
    mag_gal_vec= [20]#, 20, 21, 22,23,24]
    e_vec=[ (-0.05, 0.), (-0.025, 0.), (0.,0.), (0.025, 0.), (0.05, 0.), (0., -0.05), (0., -0.025), (0., 0.025), (0., 0.05)]
    e1_vec=[ (-0.05, 0.), (-0.025, 0.), (0.,0.), (0.025, 0.), (0.05, 0.)]
    e2_vec= [(0., -0.05), (0., -0.025), (0.,0.), (0., 0.025), (0., 0.05)]



elif x_var == 'mag_and_ellipticity':  # for delta e/ beta/ e vs mag
    beta_vec=[ 0., 0.5*beta0, beta0, 1.5*beta0, 2*beta0, 5*beta0]
    mag_gal_vec= [20, 21, 22]#, 23,24]
    #e_vec=[ (-0.05, 0.), (-0.025, 0.), (0.,0.), (0.025, 0.), (0.05, 0.), (0., -0.05), (0., -0.025), (0., 0.025), (0., 0.05)]
    #e_vec=[ (-0.05, 0.), (-0.025, 0.)]#, (0.,0.), (0.025, 0.), (0.05, 0.)]
    e_vec=[ (-0.05, 0.), (-0.025, 0.), (0.,0.), (0.025, 0.), (0.05, 0.), (0., -0.05), (0., -0.025), (0., 0.025), (0., 0.05)]
    e1_vec=[ (-0.05, 0.), (-0.025, 0.), (0.,0.), (0.025, 0.), (0.05, 0.)]
    e2_vec= [(0., -0.05), (0., -0.025), (0.,0.), (0., 0.025), (0., 0.05)]

else:
    print "ERROR in x_var: magnitude, or beta, or mag_and_beta"
    sys.exit(1)




from collections import OrderedDict


## Effective wavelengths
#  ('Y106', 18.3, 276212.53389330785, 100237.70417666467, 100237.70417666467)
# ('J129', 18.3, 282676.7892946346, 89742.235017183702, 89742.235017183702)
#  ('F184', 18.3, 183466.88869417104, 38654.372703392037, 38654.372703392037)
#  ('H158', 18.3, 279228.07239874348, 71890.380293702794, 71890.380293702794)

#Define wavelengths, ellipticities, and magnitudes
#wavelength_dict=OrderedDict([('Z087',0.869), ('Y106',1.060)  , ('J129',1.293) , ('W149',1.485), ('H158',1.577), ('F184',1.842)])  # in microns
#wavelength_dict=OrderedDict([('J129',1.292), ('Y106',0.873), ('J129',1.292), ('H158',1.577), ('F184',1.837)])  # in microns
wavelength_dict=OrderedDict([('J129',1.292) , ('Y106',1.060), ('H158',1.577), ('F184',1.837)])  # in microns

#wavelength_dict=OrderedDict([('J129',1.292)])

#wavelength_dict=OrderedDict([('Y106',1.060)])


#flux_dict={'Y106':3.4470e4 , 'J129':9.5477e4, 'H158':9.5178e4, 'F184':7.1792e4} # at 20, ETC

# use galsim to get fluxes with WFIRST properties and AB_zeropint
flux_dict={}
for bandpass in ['Y106', 'J129', 'H158', 'F184']:
    f=filters[bandpass]
    b=galsim.sed.SED(f)
    c=b.withMagnitude(m_zero, f)
    flux_dict[bandpass]=c.calculateFlux(f) #30% of largest value will give about 85% of 110k fw.

print "flux_dict: ", flux_dict

#e_vec=[ (0., 0.), (0.05, 0.), (0.05, 0.05) ] #, (0., 0.075), (0.075, 0.), (0.075, 0.075)] #, 0.05, 0.06, 0.07, 0.08]
new_params = galsim.hsm.HSMParams(max_amoment=60000000, max_mom2_iter=1000000000,  max_moment_nsig2=25)
big_fft_params = galsim.GSParams(maximum_fft_size=int(512*k))


## Initialize things
gauss_no_noise=OrderedDict()
optical_no_noise=OrderedDict()
gauss_noise=OrderedDict()
optical_noise=OrderedDict()

wfirst_no_noise=OrderedDict()
wfirst_noise=OrderedDict()


if x_var == 'magnitude':    # Simgle beta, single e, multiple lam, multiple mag. Will plot deltas vs mag
    for c in beta_means:
        gauss_no_noise[c]=OrderedDict()   #\Delta e1, \Delta e2, \Delta R/R
        optical_no_noise[c]=OrderedDict()
        gauss_noise[c]=OrderedDict()
        optical_noise[c]=OrderedDict()
        wfirst_no_noise[c]=OrderedDict()
        wfirst_noise[c]=OrderedDict()
    
        for lam in wavelength_dict:
            gauss_no_noise[c][lam]={}
            optical_no_noise[c][lam]={}
            gauss_noise[c][lam]={}
            optical_noise[c][lam]={}
            wfirst_no_noise[c][lam]={}
            wfirst_noise[c][lam]={}
        
            for e in e_vec:
                gauss_no_noise[c][lam][e]=[ ([],[])  , ([],[]) , ([],[]) ]   #\Delta e1, \Delta e2, \Delta R/R (second entry is error)
                optical_no_noise[c][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                gauss_noise[c][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                optical_noise[c][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                wfirst_no_noise[c][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                wfirst_noise[c][lam][e]=[([],[])  , ([],[]) , ([],[]) ]

else:

    for m in mag_gal_vec:
        gauss_no_noise[m]=OrderedDict()   #\Delta e1, \Delta e2, \Delta R/R
        optical_no_noise[m]=OrderedDict()
        gauss_noise[m]=OrderedDict()
        optical_noise[m]=OrderedDict()
        wfirst_no_noise[m]=OrderedDict()
        wfirst_noise[m]=OrderedDict()
        
        for lam in wavelength_dict:
            gauss_no_noise[m][lam]={}
            optical_no_noise[m][lam]={}
            gauss_noise[m][lam]={}
            optical_noise[m][lam]={}
            wfirst_no_noise[m][lam]={}
            wfirst_noise[m][lam]={}
            
            for e in e_vec:
                gauss_no_noise[m][lam][e]=[ ([],[])  , ([],[]) , ([],[]) ]   #\Delta e1, \Delta e2, \Delta R/R (second entry is error)
                optical_no_noise[m][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                gauss_noise[m][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                optical_noise[m][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                wfirst_no_noise[m][lam][e]=[([],[])  , ([],[]) , ([],[]) ]
                wfirst_noise[m][lam][e]=[([],[])  , ([],[]) , ([],[]) ]






## DO the actual measurements

multi=False
if multi:
    from multiprocessing import Pool
    #Trick to pass one argument to multiprocessing?
    from functools import partial
    p = Pool()
    mymap = p.map
else:
    mymap = map



#### TEMPORAL: histogram of dofferences at different offsets
def plot_histogram (vec1, vec2, vec3, outfile='histogram.pdf'):
    pp1=PdfPages (outfile)
    vec1, vec2, vec3 = np.array(vec1), np.array(vec2), np.array(vec3)

    print vec1
    print vec2
    print vec3

    nbins= 50
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(vec1*10**5, nbins, normed=True, facecolor='blue', alpha=0.65, label='delta e1')
    plt.xlabel('delta e1')
    plt.ylabel('normalized histogram')
    #plt.yscale('log',nonposy='clip')
    plt.legend(loc='upper right')
    pp1.savefig()
    plt.cla()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(vec2*10**4, nbins, normed=True, facecolor='red', alpha=0.65, label='delta e2')
    plt.xlabel('delta e2')
    plt.ylabel('normalized histogram')
    #plt.yscale('log',nonposy='clip')
    plt.legend(loc='upper right')
    pp1.savefig()
    plt.cla()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(vec3*10**3, nbins, normed=True, facecolor='green', alpha=0.65, label='delta R/R')
    plt.xlabel('delta R/R')
    plt.ylabel('normalized histogram')
    #plt.yscale('log',nonposy='clip')
    plt.legend(loc='upper right')
    pp1.savefig()
    pp1.close()


pepe=45

for mag in mag_gal_vec:
    for lam in wavelength_dict:
        lam_over_diam = wavelength_dict[lam] * 1.e-6 / tel_diam * galsim.radians
        lam_over_diam = lam_over_diam / galsim.arcsec
        print "lam_over_diam", lam_over_diam
        for e in e_vec:
                for beta in beta_vec:

                    if dispersion and x_var == 'beta':
                        beta_sigma = beta
                        beta = beta0
                    if dispersion and x_var == 'mag_and_beta':
                        beta_sigma = beta
                        beta = beta0
                    
                    #all_params = (beta, 0., 0.)
                    #all_params = (0., beta, 0.)
                    #all_params = (0., 0., beta)



                    logger.info("PARAMETERS: ")
                    logger.info("lambda: %s microns",  wavelength_dict[lam])
                    logger.info("ellipticity 1: %g", e[0])
                    logger.info("ellipticity 2: %g", e[1])
                    logger.info("mag: %g", mag)
                    print "N: ", n
                    #logger.info("beta: %g", beta)
                    print "beta: ", beta
                    print " "
                    if dispersion:
                        print "Dispersion: (%g, %g)" %(beta, beta_sigma)
                    
                    #### Gaussian
                    ## no noise
                    #logger.info("First loop: gaussian, no noise")
                    gal_flux=flux_dict[lam]*10**(0.4*(m_zero-mag))

                    if profile_type == 'gaussian':
                        gal= galsim.Convolve (galsim.Gaussian(flux=gal_flux, sigma=gal_sigma).shear(galsim.Shear(e1=e[0],e2=e[1])) , galsim.Pixel(pixel_scale), gsparams=big_fft_params )
                        e1_inter_vec=gauss_no_noise[lam][mag][e][0][0]
                        e2_inter_vec=gauss_no_noise[lam][mag][e][1][0]
                        size_inter_vec=gauss_no_noise[lam][mag][e][2][0]

                        e1_inter_vec_err=gauss_no_noise[lam][mag][e][0][1]
                        e2_inter_vec_err=gauss_no_noise[lam][mag][e][1][1]
                        size_inter_vec_err=gauss_no_noise[lam][mag][e][2][1]


                    elif profile_type == 'optical':
                        gal=galsim.Convolve (galsim.OpticalPSF(lam_over_diam, obscuration = obscuration_optical, flux=gal_flux).shear(galsim.Shear(e1=e[0],e2=e[1])), galsim.Pixel(pixel_scale), gsparams=big_fft_params )
                        e1_inter_vec=optical_no_noise[lam][mag][e][0][0]
                        e2_inter_vec=optical_no_noise[lam][mag][e][1][0]
                        size_inter_vec=optical_no_noise[lam][mag][e][2][0]

                        e1_inter_vec_err=optical_no_noise[lam][mag][e][0][1]
                        e2_inter_vec_err=optical_no_noise[lam][mag][e][1][1]
                        size_inter_vec_err=optical_no_noise[lam][mag][e][2][1]
                    
                    elif profile_type == 'wfirst':
                        if x_var == 'magnitude':
                        
                            gal=galsim.Convolve( wfirst.getPSF(SCAs=18,approximate_struts=False, wavelength=filters[lam])[18].shear(galsim.Shear(e1=e[0],e2=e[1])).withFlux(gal_flux), galsim.Pixel(pixel_scale), gsparams=big_fft_params)
                            e1_inter_vec=wfirst_no_noise[c][lam][e][0][0]
                            e2_inter_vec=wfirst_no_noise[c][lam][e][1][0]
                            size_inter_vec=wfirst_no_noise[c][lam][e][2][0]
                    
                            e1_inter_vec_err=wfirst_no_noise[c][lam][e][0][1]
                            e2_inter_vec_err=wfirst_no_noise[c][lam][e][1][1]
                            size_inter_vec_err=wfirst_no_noise[c][lam][e][2][1]
                        else:
                            gal=galsim.Convolve( wfirst.getPSF(SCAs=18,approximate_struts=False, wavelength=filters[lam])[18].shear(galsim.Shear(e1=e[0],e2=e[1])).withFlux(gal_flux), galsim.Pixel(pixel_scale), gsparams=big_fft_params)
                            e1_inter_vec=wfirst_no_noise[mag][lam][e][0][0]
                            e2_inter_vec=wfirst_no_noise[mag][lam][e][1][0]
                            size_inter_vec=wfirst_no_noise[mag][lam][e][2][0]
                            
                            e1_inter_vec_err=wfirst_no_noise[mag][lam][e][0][1]
                            e2_inter_vec_err=wfirst_no_noise[mag][lam][e][1][1]
                            size_inter_vec_err=wfirst_no_noise[mag][lam][e][2][1]
                    
                    
                        
                    else:
                        print "Wrong 'profile_type'. "
                        sys.exit(1)


                    # beta from Normal distribution?
                    if n_realizations > 1:
                        """
                        def helper_function (index):
                            seed=index
                            temp=np.random.RandomState(seed=seed).normal(beta, beta_sigma , base_size*base_size)
                            beta_matrix=np.reshape(temp, (base_size, base_size))
                            e1_out, e2_out, size_out=measurement_function (gal, type=type, beta=beta_matrix, base_size=base_size, n=n, pixel_scale=pixel_scale)
                            return e1_out, e2_out, size_out

                        processes=cpu_count()
                        use=5
                        print "I have ", processes, "cores here. Using: %g" %use
                        pool = Pool(processes=6)

                        print "starting: "
                        results=pool.map (helper_function, range(n_realizations))
                        temp_e1, temp_e2, temp_size=[],[],[]
                        for line in results:
                            temp_e1.append(line[0])
                            temp_e2.append(line[1])
                            temp_size.append(line[2])
                        """

                        temp_e1, temp_e2, temp_size=[],[],[]
                        for seed in range(1, n_realizations+1):
                            print "REALIZATION number (also seed): ", seed
                            temp=np.random.RandomState(seed=seed).normal(beta, beta_sigma , base_size*base_size)
                            beta_matrix=np.reshape(temp, (base_size, base_size))
                            if multi:
                                ##multiprocessing
                                worker_funct=partial (measurement_function, type=type, beta=beta_matrix, base_size=base_size, n=n, pixel_scale=pixel_scale)
                                output=np.array(mymap(worker_funct, gal ))
                                e1_out, e2_out, size_out=(output[:, 0], output[:, 1], output[:, 2])
                            else:
                                e1_out, e2_out, size_out=measurement_function (gal, type=type, beta=beta_matrix, base_size=base_size, n=n, pixel_scale=pixel_scale)
                            
                                ##Measurement with fixed beta, to take the difference
                                #e1_out_fixed, e2_out_fixed, size_out_fixed=measurement_function (gal, type=type, beta=beta, base_size=base_size, n=n, pixel_scale=pixel_scale)

                            temp_e1.append(e1_out)
                            temp_e2.append(e2_out)
                            temp_size.append(size_out)

                            #temp_e1.append(e1_out)
                            #temp_e2.append(e2_out)
                            #temp_size.append(size_out)
                        

                        mean_e1, mean_e2, mean_size= np.mean(temp_e1), np.mean(temp_e2), np.mean(temp_size)
                        
                        error_e1, error_e2, error_size = np.std(temp_e1), np.std(temp_e2), np.std(temp_size)
                        print "mean_e1, mean_e2, mean_size: ", mean_e1, mean_e2, mean_size
                            
                        


                    else:
                        if multi:
                            ##multiprocessing
                            worker_funct=partial (measurement_function, type=type, beta=beta, base_size=base_size, n=n, pixel_scale=pixel_scale)
                            output=np.array(mymap(worker_funct, gal ))
                            e1_out, e2_out, size_out=(output[:, 0], output[:, 1], output[:, 2])
                        else:
                            ## Do loop here over spixel shifts. temporal. To average over beta, delete loop and no means. (This is for a single beta)
                            temp_e1, temp_e2, temp_size=[],[],[]
                            of=[]
                            for ind in range(n_offsets):
                                if ind == 0:
                                    offset= (0.0,0.0)
                                ud=galsim.UniformDeviate(ind)
                                offset= (ud(), ud())
                                offset_highres = (offset[0]*n % 1, offset[1]*n % 1)
                                of.append(offset_highres)
                                #print "offset_highres: ", offset_highres
                                e1_out, e2_out, size_out= measurement_function (gal, type=type, beta=beta, base_size=base_size, n=n, pixel_scale=pixel_scale, offset=offset_highres)
                                #print "INSIDE LOOP: ", e1_out, e2_out, size_out

                                temp_e1.append(e1_out)  #temp
                                temp_e2.append(e2_out)   #temp
                                temp_size.append(size_out)  #temp

                        #mean_e1, mean_e2, mean_size=e1_out, e2_out, size_out
                        #error_e1, error_e2, error_size=0.,0.,0.
                            #Temporal : plot histogram
                            if pepe == 0:
                                plot_histogram (temp_e1, temp_e2, temp_size, outfile='histogram_%s_%g_offsets_pepe.pdf' %(lam, n_offsets))
                                pepe+=1
                            #print "of: ", of
                            mean_e1, mean_e2, mean_size= np.mean(temp_e1), np.mean(temp_e2), np.mean(temp_size)
                            error_e1, error_e2, error_size = np.std(temp_e1), np.std(temp_e2), np.std(temp_size)
                            #print "mean_e1, mean_e2, mean_size: ", mean_e1, mean_e2, mean_size
                            #print "error_e1, error_e2, error_size: ", error_e1, error_e2, error_size
                            #sys.exit(1)

                    if not dispersion:

                        e1_inter_vec.append(mean_e1)
                        e1_inter_vec_err.append(error_e1)

                        e2_inter_vec.append(mean_e2)
                        e2_inter_vec_err.append(error_e2)

                        size_inter_vec.append (mean_size)
                        size_inter_vec_err.append(error_size)

                    else:
                        
                        e1_inter_vec.append(error_e1)
                        e1_inter_vec_err.append(0.)
                        
                        e2_inter_vec.append(error_e2)
                        e2_inter_vec_err.append(0.)
                        
                        size_inter_vec.append (error_size)
                        size_inter_vec_err.append(0.)




                    #WFIRST
                    #gal=wfirst.getPSF(SCAs=7,approximate_struts=True, wavelength=filters['W149'])[7].shear(galsim.Shear(e1=e, e2=e))
                    #measurement_function_NL (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=w_e1, e2_vec=w_e2, size_vec=w_s, e1_inter_vec=wi_e1, e2_inter_vec=wi_e2, size_inter_vec=wi_s, noise=None, string='WFIRST, no noise')
                    #noise
                    #measurement_function_NL (gal, true_e1=e, true_e2=e, true_s=0., e1_vec=nw_e1, e2_vec=nw_e2, size_vec=nw_s, e1_inter_vec=nwi_e1, e2_inter_vec=nwi_e2, size_inter_vec=nwi_s, noise=noise, string='WFIRST, noise')

                    #sys.exit(1)


#sys.exit(1)

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
prop = fm.FontProperties(size=8)
marker_size=7.0
alpha=0.7
loc_label = "upper right"
visible_x, visible_y = True, True
grid=False
ymin, ymax = -0.0001, 0.0001
m_req=1e-3
c_req=1e-4


color_vec=['r', 'y', 'g', 'c', 'b', 'm', 'k']
#color_dict={0.0:'r', 0.025:'k', 0.05:'b', 0.075:'m', 0.08:'c', 0.1:'g'}
color_vec_lam=['b--o', 'g-s', 'y-.x', 'r:+', 'm', 'b']



color_dict_e={}
for i,e in enumerate(e_vec):
    color_dict_e[e]=color_vec[i%len(color_vec)]

color_dict_mag={}
for i,m_gal in enumerate(mag_gal_vec):
    color_dict_mag[m_gal]=color_vec[i%len(color_vec)]


color_dict_lam={}
for i,lam in enumerate(wavelength_dict):
    color_dict_lam[lam]=color_vec_lam[i%len(color_vec_lam)]






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


## Theory for Delta R / R
#def theory_size_gauss (sigma,  beta, flux_vec):
#    sigma/=(pixel_scale)   ### Convert to pixels?
#    return  ( (8*math.pi - beta*flux_vec/(sigma**2) ) / (8*math.pi - 2*beta*flux_vec/(sigma**2)  ) )   - 1
#flux_vec=flux_dict['H158']*2.512**( m_zero - np.array(mag_gal_vec) )
#ratio_vec= theory_size_gauss (gal_sigma, beta0, flux_vec  )

#flux_18 = flux_dict['F184']*2.512**(m_zero - 18)
#ratio_vec_fixed_flux = theory_size_gauss (gal_sigma, beta_means, flux_18)
#print  "GAUSSIAN THEORY: ratio_vec_fixed_flux", ratio_vec_fixed_flux


def plot_function_e_and_r (fig, x_vec, y1_vec, y2_vec, y3_vec, y1_vec_err=None, y2_vec_err=None, y3_vec_err=None, x1label='', x2label='', y1label=r"$\Delta$e", y2label=r"$\Delta$R/R", lam_key='H158', e_key=(0.0, 0.0), position=111, title=''):
    
    if y1_vec_err == None: y1_vec_err=0.
    if y2_vec_err == None: y2_vec_err=0.
    if y3_vec_err == None: y3_vec_err=0.
    x_vec, y1_vec, y2_vec, y3_vec, y1_vec_err, y2_vec_err, y3_vec_err = np.array(x_vec), np.array(y1_vec), np.array(y2_vec), np.array(y3_vec), np.array(y1_vec_err), np.array(y2_vec_err), np.array(y3_vec_err)
    x_vec, y1_vec, y2_vec, y3_vec = np.array(x_vec), np.array(y1_vec), np.array(y2_vec), np.array(y3_vec)
    y1_vec_err, y2_vec_err, y3_vec_err=np.array(y1_vec_err),np.array(y2_vec_err), np.array(y3_vec_err)


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




    if x_var == 'beta':
        #x1label=r"$\gamma$ ($\times 10^{10}$)"
        x1label=r"$\beta$ ($\times 10^6$)"
        #x1label=r"$\delta$"
        #x1label=r"$\delta$ ($\times 10^{15}$)"
        x_vec=np.array(x_vec)*(10**6)

        if dispersion:
            x1label=r"$\sigma_{\beta0}$ ($\times 10^6$)"
            x_vec=np.array(x_vec)*(10**6)

    elif x_var == 'mag_and_beta':
        x1label=r"mag"
        #if dispersion:
            #x1label=r"$\beta$ ($\times 10^6$)"
            #x_vec=np.array(x_vec)*(10**6)
    elif x_var == 'magnitude':
        x1label=r"mag"
    else:
        x1_label='x'
    
    ax = fig.add_subplot (223)
    ax.errorbar( x_vec, y1_vec, yerr=y1_vec_err, ecolor = color_fmt[0], label=label_e1, fmt=color_fmt, markersize=marker_size, alpha=alpha)
    #ax.errorbar( x_vec, y2_vec, yerr=y2_vec_err, ecolor = color_fmt, label=label_e2, fmt=color_fmt+'x-', markersize=marker_size, alpha=alpha)
    plt.axhline(y=0.,color='k',ls='solid')
    if  e_key[0] == 0.0 and e_key[1] == 0.0 and label_type == 'ellipticity':
        plt.axhline(y=1e-5, color='r',ls='-', label='1e-5') # requirement
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)


    lx=ax.set_xlabel(x1label, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)

    if x_var == 'beta':
        y1label=r"$\Delta$e$_1$ ($\times 10^4$)"
        #y1label=r"$d_{\Delta e_1} (\times 10^4)$" # For diferences multi vs fixed beta
        if dispersion:
            y1label=r"$\sigma_{\Delta e_1}$"

    elif x_var == 'mag_and_beta':
        y1label=r"$\Delta$e$_1/\beta$"
        if dispersion:
            y1label=r"$\sigma_{\Delta e_1}$ / $\sigma_{\beta}$ "
            #y2_vec=y2_vec*10**-4
        #y1label=r"$\Delta$e$_1/\gamma$"
        #y1label=r"$\Delta$e$_1/\delta$"



    elif x_var == 'magnitude':
        y1label=r"$\Delta$e$_1$"
    else:
        y1_label='y'

    #y1label=r"$\Delta$e$_1/\beta$"
    #y1label=r"$\Delta$e$_1$"
    #y1label= y1label + " (Difference fixed vs no fixed " + r"$\beta$)"


    ly=ax.set_ylabel(y1label, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('log')
    #plt.ylim ([1e2, 1e6])

    #plt.ylim ([-1e-4, 8e4])
    #plt.ylim ([-0.15, 0.15])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
    if label_type == 'ellipticity':
        plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=11)
    #if plot_pos== 321:
    ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
    
    #plt.grid(grid, which='both', ls='-', alpha=0.5)
    #plt.grid(grid)


    ax = fig.add_subplot (224)
    #ax.errorbar( x_vec, y1_vec, yerr=y1_vec_err, ecolor = color_fmt, label=label_e1, fmt=color_fmt+'s-', markersize=marker_size, alpha=alpha)
    ax.errorbar( x_vec, y2_vec, yerr=y2_vec_err, ecolor = color_fmt[0], label=label_e2, fmt=color_fmt, markersize=marker_size, alpha=alpha)
    plt.axhline(y=0.,color='k',ls='solid')
    if  e_key[0] == 0.0 and e_key[1] == 0.0 and label_type == 'ellipticity':
        plt.axhline(y=1e-5, color='r',ls='-', label='1e-5') # requirement
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)

    if x_var == 'beta':
        x1label=r"$\beta$ ($\times 10^6$)"
        #x1label=r"$\gamma$ ($\times 10^{10}$)"
        #x1label=r"$\beta$"
        #x1label=r"$\delta$ ($\times 10^15$)"
        #x1label=r"$\delta$ ($\times 10^{15}$)"
        if dispersion:
            x1label=r"$\sigma_{\beta0}$ ($\times 10^6$)"
            x_vec=np.array(x_vec)*(10**0)

    elif x_var == 'mag_and_beta':
        x1label=r"mag"
        #if dispersion:
            #x1label=r"$\beta$ ($\times 10^6$)"
    elif x_var == 'magnitude':
        x1label=r"mag"
    else:
        x1_label='x'

    #x1label=r"mag"
    #x1label=r"$\beta$"

    lx=ax.set_xlabel(x1label, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)


    if x_var == 'beta':
        y1label=r"$\Delta$e$_2$" #($\times 10^3$)"
        #y1label=r"$d_{\Delta e_2} (\times 10^4)$" # For diferences multi vs fixed beta
        if dispersion:
            y1label=r"$\sigma_{\Delta e_2}$"

    elif x_var == 'mag_and_beta':
        y1label=r"$\Delta$e$_2/\beta$"

        if dispersion:
            y1label=r"$\sigma_{\Delta e_2}$ / $\sigma_{\beta}$"

        #y1label=r"$\Delta$e$_2/\gamma$"
        #y1label=r"$\Delta$e$_2/\delta$"


    elif x_var == 'magnitude':
        y1label=r"$\Delta$e$_2$"

    else:
        y1_label='y'


    #y1label=r"$\Delta$e$_2/\beta$"
    #y1label=r"$\Delta$e$_2$"
    #y1label= y1label + " (Difference fixed vs no fixed " + r"$\beta$)"

    ly=ax.set_ylabel(y1label, visible=visible_y)
    #ly.set_fontsize(font_size)
    ax.set_yscale('log')
    #plt.ylim ([1e2, 1e6])

    #plt.ylim ([-20, 2600])
    #plt.ylim ([-0.15, 0.15])
    xmin, xmax=plt.xlim()
    delta=(xmax-xmin)
    plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
    if label_type == 'ellipticity':
        plt.title(lam_key+" (%g $\mu$m)"%wavelength_dict[lam], fontsize=11)
    #if plot_pos== 321:
    ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)



    ax = fig.add_subplot (211)
    if  e_key[0] == 0.0 and e_key[1] == 0.0:
        ax.errorbar( x_vec, y3_vec, yerr=y3_vec_err, ecolor = color_fmt[0], label=label, fmt=color_fmt, markersize=marker_size, alpha=alpha)
    #ax.errorbar( x_vec, theory_delta_r_gauss, yerr=None, ecolor = 'k', label='theory Gauss', fmt='r-', markersize=marker_size, alpha=1.)
    plt.axhline(y=0.,color='k',ls='solid')
    #if  e_key[0] == 0.0 and e_key[1] == 0.0 and lam_key == 'H158' and (label_type == 'ellipticity' or label_type == 'magnitude'):
        #plt.axhline(y=1e-4, color='r',ls='-', label='1e-4') # requirement
        #if x_var == 'magnitude' and profile_type == 'gaussian':
    #ax.errorbar(x_vec, ratio_vec_fixed_flux , yerr=None, ecolor = 'k', label='', fmt='k-', markersize=marker_size, alpha=1.)
    #plt.axhspan(-m_req, m_req, facecolor='0.5', alpha=0.3)
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)

    if x_var == 'beta':
        x2label=r"$\beta$ ($\times 10^6$)"
        #x2label=r"$\gamma$ ($\times 10^{10}$)"
        #x2label=r"$\delta$ ($\times 10^15$)"
        #x2label=r"$\delta$ ($\times 10^{15}$)"
        ax.set_yscale('linear')
        #y2label=r"$d_{\Delta R/R} (\times 10^4)$"  # For diferences multi vs fixed beta
        if dispersion:
            x2label=r"$\sigma_{\beta0}$ ($\times 10^6$)"
            x_vec=np.array(x_vec)*(10**0)
            y2label=r"$\sigma_{\Delta R/R}$"


    elif x_var == 'mag_and_beta':
        x2label=r"mag"
        ax.set_yscale('log')
        plt.ylim ([1e2, 1e5])
        y2label=r"$\Delta R/R/\beta$"
        if dispersion:
            #x2label=r"$\beta$ ($\times 10^6$)"
            y2label=r"$\sigma_{\Delta R/R}$ / $\sigma_{\beta}$ "
    elif x_var == 'magnitude':
        x2label=r"mag"
        ax.set_yscale('linear')
        #plt.ylim ([50, 3e4])
        y2label="$\Delta R/R$"

    else:
        x2_label='x'

    #x2label=r"mag"

    #x2label=r"$\beta$"
    lx=ax.set_xlabel(x2label, visible=visible_x)
    #lx.set_fontsize(font_size)
    ax.set_xscale('linear')
    ax.set_yticklabels(ax.get_yticks(), visible= visible_y)

    #y2label = y2label + " (Difference fixed vs no fixed " + r"$\beta$)"

    ly=ax.set_ylabel(y2label, visible=visible_y)
    #ly.set_fontsize(font_size)
    #ax.set_yscale('linear')
    #ax.set_yscale('log')
    #plt.ylim ([50, 3e4])

    #plt.ylim([0., 0.35])
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

    plt.title(title)
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

plot_positions_six={'Z087':321,'Y106':322, 'J129':323, 'W149':324, 'H158':325, 'F184':326}


if x_var == 'magnitude' or  x_var== 'mag_and_ellipticity':

    x_vec=mag_gal_vec
    x_label=r"mag"


    if type == 'bf':
        string_g= "BF: BaseCDModel" + "\n" + "Gaussian ($\sigma$=%g'')* Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= "BF: BaseCDModel" + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= "BF: BaseCDModel" + "\n" + "WFIRST PSF"

    elif type == 'nl':
        string_g= r"Non-linearity: $I \mapsto I-\beta I^{2}$  " + r"($\beta$=%g)" %(beta0) + "\n" + "Gaussian ($\sigma$=%g'') * Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= r"Non-linearity: $I \mapsto I-\beta I^{2}$  " + r"($\beta$=%g)" %(beta0) + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= r"Non-linearity: $I \mapsto I-\beta I^{2}$  " + r"($\beta$=%g)" %(beta0) + "\n" + "WFIRST PSF"
    else:
        print "invalid type (nor 'bf' nor 'nl')"
        sys.exit(1)
elif x_var== 'mag_and_beta':
    x_vec=mag_gal_vec
    x_label=r"mag_object"
    
    
    if type == 'bf':
        string_g= "BF: BaseCDModel" + "\n" + "Gaussian ($\sigma$=%g'')* Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= "BF: BaseCDModel" + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= "BF: BaseCDModel" + "\n" + "WFIRST PSF"
    elif type == 'nl':
        string_g= r"Non-linearity: $I \mapsto I-\beta I^{2}$" + "\n" + "Gaussian ($\sigma$=%g'') * Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= r"Non-linearity: $I \mapsto I-\beta I^{2}$" + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= r"Non-linearity: $I \mapsto I-\beta I^{2}$ " + "\n" + "WFIRST PSF"
    else:
        print "invalid type (nor 'bf' nor 'nl')"
        sys.exit(1)


elif x_var == 'beta':
    x_vec=beta_vec
    x_label=r"Nonlinearity parameter $\beta$"

    if type == 'bf':
        string_g= "BF: BaseCDModel" + "\n" + "Gaussian ($\sigma$=%g'')* Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= "BF: BaseCDModel" + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= "BF: BaseCDModel" + "\n" + "WFIRST PSF"
    elif type == 'nl':
        string_g= r"Non-linearity: $I \mapsto I-\beta I^{2}$." + "mag_object=%g" %(mag_gal_vec[0]) + "\n" + "Gaussian ($\sigma$=%g'') * Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= r"Non-linearity: $I \mapsto I-\beta I^{2}$. " + "mag_object=%g" %(mag_gal_vec[0]) + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= r"Non-linearity: $I \mapsto I-\beta I^{2}$ " + "mag_object=%g" %(mag_gal_vec[0]) + "\n" + "WFIRST PSF"
    else:
        print "invalid type (nor 'bf' nor 'nl')"
        sys.exit(1)

elif x_var == 'ellipticity':
    #x_vec=beta_vec
    x_label=r"$e_2$"
    
    
    if type == 'bf':
        string_g= "BF: BaseCDModel" + "\n" + "Gaussian ($\sigma$=%g'')* Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= "BF: BaseCDModel" + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= "BF: BaseCDModel" +  + "mag_object=%g" %(mag_gal_vec[0]) + "\n" + "WFIRST"
    elif type == 'nl':
        string_g= r"Non-linearity: $I \mapsto I-\beta I^{2}$. " + "mag_object=%g" %(mag_gal_vec[0]) + "\n" + "Gaussian ($\sigma$=%g'') * Pixel (%g arcsec/pix), no noise." %(gal_sigma, pixel_scale)
        string_o= r"Non-linearity: $I \mapsto I-\beta I^{2}$ " + "mag_object=%g" %(mag_gal_vec[0]) + "\n" + "OpticalPSF (tel_diam=%g m, obscuration=%g) * Pixel (%g arcsec/pix), no noise. "%(tel_diam, obscuration_optical, pixel_scale)
        string_w= r"Non-linearity: $I \mapsto I-\beta I^{2}$. " + "mag_object=%g" %(mag_gal_vec[0]) + "\n" + "WFIRST PSF"
    else:
        print "invalid type (nor 'bf' nor 'nl')"
        sys.exit(1)


else:
    print "invalid 'x_var' (nor 'magnitude' nor 'beta' nor 'mag_and_beta' nor 'ellipticity')"
    sys.exit(1)

if profile_type == 'gaussian':
    string=string_g
elif profile_type == 'optical':
    string=string_o
elif profile_type == 'wfirst':
    string=string_w
else:
    print "wrong profile_type"
    sys.exit(1)



### For the normalized profiles (e.g., \deltaR / \beta vs mag)
def get_slope (x, y):
    fitfunc = lambda p, x: p[0]*x + p[1]
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    p0 = [1., 1.]
    print "Fitting x, y: ", x, y
    p1, success = optimize.leastsq(errfunc, p0[:], args=(x,y))
    print 'pendiente and intercept, and success:', p1[0] , p1[1], success
    return p1[0]

def get_slope2 (x, y):
    p=np.polyfit(x,y,1)
    print 'pendiente and intercept from np.polyfit', p[0] , p[1]
    return p[0], p[1]

if x_var == 'mag_and_beta':
    beta_vec=np.array(beta_vec)
    slope_dict={}
    #gamma_temp={}
    
    for lam in wavelength_dict:
        slope_dict[lam] =[[],[],[]]  # e1, e2, size
        #gama_temp[lam] =[[],[],[]]
    
    fig = plt.figure()
    for e in e_vec:
        for m_gal in mag_gal_vec:
            for lam in wavelength_dict:
                print "mgal: %g, lam: %s" %(m_gal,lam)
                if profile_type == 'gaussian':
                    e1_inter_vec=gauss_no_noise[m_gal][lam][e][0][0]
                    e2_inter_vec=gauss_no_noise[m_gal][lam][e][1][0]
                    size_inter_vec=gauss_no_noise[m_gal][lam][e][2][0]
                elif profile_type == 'optical':
                    e1_inter_vec=optical_no_noise[m_gal][lam][e][0][0]
                    e2_inter_vec=optical_no_noise[m_gal][lam][e][1][0]
                    size_inter_vec=optical_no_noise[m_gal][lam][e][2][0]
                elif profile_type == 'wfirst':
                    e1_inter_vec=wfirst_no_noise[m_gal][lam][e][0][0]
                    e2_inter_vec=wfirst_no_noise[m_gal][lam][e][1][0]
                    size_inter_vec=wfirst_no_noise[m_gal][lam][e][2][0]
                else:
                    print "Wrong 'profile_type'. "
                    sys.exit(1)

            
                slope_e1= get_slope2 ( np.array(beta_means), e1_inter_vec)[0]  #delta_e1
                slope_dict[lam][0].append(slope_e1)
               
                slope_e2= get_slope2 ( np.array(beta_means), e2_inter_vec)[0]  #delta_e2
                slope_dict[lam][1].append(slope_e2)
            
                slope_r= get_slope2 (np.array(beta_means), size_inter_vec)[0]    #delta_r
                slope_dict[lam][2].append(slope_r)

                if slope_e1 == 1. or slope_e2 == 1. or slope_r == 1.:
                    print "Fit to line about point failed."
                    sys.exit(1)



    for lam in wavelength_dict:
        temp=[]
        print "lam", lam
        print "slope_dict[lam][0] , slope_dict[lam][1], slope_dict[lam][2]: ", slope_dict[lam][0] , slope_dict[lam][1], slope_dict[lam][2]
        plot_function_e_and_r (fig, mag_gal_vec, slope_dict[lam][0] , slope_dict[lam][1], slope_dict[lam][2], x1label=x_label, y1label=r"$\Delta e$/$\gamma$", y2label=r"$\Delta R/R/\gamma$", lam_key=lam)


        ###### Fitting a power law (or a line in log space) log y = log A + B (m - m0)
        B, logA = get_slope2 (np.log10(np.array( slope_dict[lam][2]) ) ,  np.array(mag_gal_vec) - 20  )
        print "lambda: %s, B: %g, logA: %g, A:% g" % (lam, B, logA, 10**(logA))


        #print "mag_gal_vec, e1_inter_vec, e2_inter_vec, size_inter_vec", mag_gal_vec, e1_inter_vec , e2_inter_vec, size_inter_vec
        #print "HOLA "
        
        #plot_function_e_and_r (fig, mag_gal_vec, e1_inter_vec , e2_inter_vec, size_inter_vec, x1label=x_label, y1label=r"$\Delta e$", y2label=r"$\gamma$", lam_key=lam)

    #plt.suptitle(string, fontsize=13)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    pp.savefig(fig)
    plt.close()

elif x_var == 'ellipticity':
    beta_vec=np.array(beta_vec)
    slope_dict={}
    for lam in wavelength_dict:
        slope_dict[lam] =[[],[]]
    
    #e_component_vec= np.array([-0.05, -0.025, 0., 0.025, 0.05])

    fig = plt.figure()
    for lam in wavelength_dict:
        e1_component_vec, e2_component_vec = [],[]
        for e in e_vec:
            for m in mag_gal_vec:
                if profile_type == 'gaussian':
                    e1_inter_vec=gauss_no_noise[lam][m][e][0][0]
                    e2_inter_vec=gauss_no_noise[lam][m][e][1][0]
                    #size_inter_vec=gauss_no_noise[lam][m][e][2]
                elif profile_type == 'optical':
                    e1_inter_vec=optical_no_noise[lam][m][e][0][0]
                    e2_inter_vec=optical_no_noise[lam][m][e][1][0]
                    #size_inter_vec=optical_no_noise[lam][m][e][2]
                else:
                    print "Wrong 'profile_type'. "
                    sys.exit(1)
            
            
                slope_e1= get_slope ( beta_vec, e1_inter_vec)  #delta_e1
                if e in e1_vec:
                    e1_component_vec.append(e[0])
                    slope_dict[lam][0].append(slope_e1)
            
                slope_e2= get_slope ( beta_vec, e2_inter_vec )  #delta_e2
                if e in e2_vec:
                    e2_component_vec.append(e[1])
                    slope_dict[lam][1].append(slope_e2)         #slope_r was found to be ind. of e

    for lam in wavelength_dict:
        temp=[]
        print "lam", lam
        print " e1_component_vec, slope_dict[lam][0] , e2_component_vec, slope_dict[lam][1]",  e1_component_vec, slope_dict[lam][0] , e2_component_vec, slope_dict[lam][1]
        plot_function_e_and_r (fig, e1_component_vec, slope_dict[lam][0] , e2_component_vec, slope_dict[lam][1], x1label=r"$e_1$", x2label=r"$e_2$", y1label=r"$\Delta e_1$/$\beta$", y2label=r"$\Delta e_2$/$\beta$", lam_key=lam)

        ## Print the data in ASCII files



    plt.suptitle(string, fontsize=13)
    print "string", string
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    pp.savefig(fig)
    plt.close()

elif x_var == 'mag_and_ellipticity':
    beta_vec=np.array(beta_vec)
    slope_dict, slope_dict2={},{}
    for lam in wavelength_dict:
        slope_dict[lam], slope_dict2[lam] = {}, [[],[]]
        for m in mag_gal_vec:
            slope_dict[lam][m]=[[],[]]
    #e_component_vec= np.array([-0.05, -0.025, 0., 0.025, 0.05])

    fig = plt.figure()
    for lam in wavelength_dict:
        e1_component_vec, e2_component_vec = [],[]
        for e in e_vec:
            if e in e1_vec:
                e1_component_vec.append(e[0])
            if e in e2_vec:
                e2_component_vec.append(e[1])
            
            for m in mag_gal_vec:
                if profile_type == 'gaussian':
                    e1_inter_vec=gauss_no_noise[lam][m][e][0][0]
                    e2_inter_vec=gauss_no_noise[lam][m][e][1][0]
                #size_inter_vec=gauss_no_noise[lam][m][e][2]
                elif profile_type == 'optical':
                    e1_inter_vec=optical_no_noise[lam][m][e][0][0]
                    e2_inter_vec=optical_no_noise[lam][m][e][1][0]
                #size_inter_vec=optical_no_noise[lam][m][e][2]
                else:
                    print "Wrong 'profile_type'. "
                    sys.exit(1)
                
                
                slope_e1= get_slope ( beta_vec, e1_inter_vec)  #delta_e1
                if e in e1_vec:
                    slope_dict[lam][m][0].append(slope_e1)
                slope_e2= get_slope ( beta_vec, e2_inter_vec )    #delta_e2
                if e in e2_vec:
                    slope_dict[lam][m][1].append(slope_e2) #slope_r was found to be ind. of e

    for lam in wavelength_dict:
        for m in mag_gal_vec:
            e1_component_vec, slope_dict[lam][m][0] = np.array(e1_component_vec), np.array(slope_dict[lam][m][0])
            e2_component_vec, slope_dict[lam][m][1] = np.array(e2_component_vec), np.array(slope_dict[lam][m][1])
            print "e1_component_vec, slope_dict[lam][m][0]", e1_component_vec, slope_dict[lam][m][0]
            print "e2_component_vec, slope_dict[lam][m][1]", e2_component_vec, slope_dict[lam][m][1]
            slope_e1= get_slope (e1_component_vec, slope_dict[lam][m][0])
            slope_e2= get_slope (e2_component_vec, slope_dict[lam][m][1])
            slope_dict2[lam][0].append(slope_e1)
            slope_dict2[lam][1].append(slope_e2)


    for lam in wavelength_dict:
            temp=[]
            print "lam", lam
            print "mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1]", mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1]
            plot_function_e_and_r (fig, mag_gal_vec, slope_dict2[lam][0] , mag_gal_vec, slope_dict2[lam][1], x1label=x_label, y1label=r"$\Delta e_1$/$\beta$/$e_1$", y2label=r"$\Delta e_2$/$\beta$/$e_2$", lam_key=lam)
    plt.suptitle(string, fontsize=13)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    pp.savefig(fig)
    plt.close()

elif x_var == 'beta':
    x_vec=beta_means
    
    for e in e_vec:
        fig = plt.figure()
        for m in mag_gal_vec:
            #if m == 18:
            #    position=221
            #    x_label=''
            #elif m == 20:
            #    position=222
            #    x_label=''
            #elif m == 22:
            #    position=223
            #elif m == 24:
            #    position=224
            #else:
            #    print "Non-existent magnitude"
            #    sys.exit(1)
            for lam in wavelength_dict:
                e1_inter_vec, e2_inter_vec, size_inter_vec= [],[], []
                e1_inter_vec_err, e2_inter_vec_err, size_inter_vec_err= [],[], []
                
                
                if profile_type == 'gaussian':
                    e1_inter_vec.append(gauss_no_noise[m][lam][e][0][0])
                    e2_inter_vec.append(gauss_no_noise[m][lam][e][1][0])
                    size_inter_vec.append(gauss_no_noise[m][lam][e][2][0])
                
                    e1_inter_vec_err.append(gauss_no_noise[m][lam][e][0][1])
                    e2_inter_vec_err.append(gauss_no_noise[m][lam][e][1][1])
                    size_inter_vec_err.append(gauss_no_noise[m][lam][e][2][1])
                elif profile_type == 'optical':
                    e1_inter_vec.append(optical_no_noise[m][lam][e][0][0])
                    e2_inter_vec.append(optical_no_noise[m][lam][e][1][0])
                    size_inter_vec.append(optical_no_noise[m][lam][e][2][0])
                
                    e1_inter_vec_err.append(optical_no_noise[m][lam][e][0][1])
                    e2_inter_vec_err.append(optical_no_noise[m][lam][e][1][1])
                    size_inter_vec_err.append(optical_no_noise[m][lam][e][2][1])
                elif profile_type == 'wfirst':
                    e1_inter_vec.append(wfirst_no_noise[m][lam][e][0][0])
                    e2_inter_vec.append(wfirst_no_noise[m][lam][e][1][0])
                    size_inter_vec.append(wfirst_no_noise[m][lam][e][2][0])
                    
                    e1_inter_vec_err.append(wfirst_no_noise[m][lam][e][0][1])
                    e2_inter_vec_err.append(wfirst_no_noise[m][lam][e][1][1])
                    size_inter_vec_err.append(wfirst_no_noise[m][lam][e][2][1])

                else:
                    
                    print "Wrong 'profile_type'. "
                    sys.exit(1)
                print "x_vec, e1_inter_vec, e2_inter_vec, size_inter_vec", x_vec, e1_inter_vec[0], e2_inter_vec[0], size_inter_vec[0]
                print "e1_inter_vec_err[0], e2_inter_vec_err[0], size_inter_vec_err[0] * sqrt (n) ", sqrt_n* np.array(e1_inter_vec_err[0]), sqrt_n*np.array(e2_inter_vec_err[0]), sqrt_n*np.array(size_inter_vec_err[0])
                print "e1_inter_vec_err[0], e2_inter_vec_err[0], size_inter_vec_err[0]", e1_inter_vec_err[0], e2_inter_vec_err[0], size_inter_vec_err[0]
                print "PLOTTING: ", lam, m, e
                plot_function_e_and_r (fig, x_vec, e1_inter_vec[0], e2_inter_vec[0], size_inter_vec[0], y1_vec_err=e1_inter_vec_err[0], y2_vec_err=e2_inter_vec_err[0], y3_vec_err=size_inter_vec_err[0], x1label=x_label, lam_key=lam, e_key=e, title="mag: %g" %m)

        #plt.suptitle(string + "actual m:%g " %m, fontsize=11)
        #plt.suptitle( "m:%g " %m, fontsize=10)
        fig.tight_layout()
        plt.subplots_adjust(top=0.85)
        pp.savefig(fig)

    plt.close()


elif x_var == 'magnitude':
    x_vec=mag_gal_vec
    
    for e in e_vec:
        fig = plt.figure()
        for c in beta_means:
        
            for lam in wavelength_dict:
                e1_inter_vec, e2_inter_vec, size_inter_vec= [],[], []
                e1_inter_vec_err, e2_inter_vec_err, size_inter_vec_err= [],[], []
                
                
                if profile_type == 'gaussian':
                    e1_inter_vec.append(gauss_no_noise[c][lam][e][0][0])
                    e2_inter_vec.append(gauss_no_noise[c][lam][e][1][0])
                    size_inter_vec.append(gauss_no_noise[c][lam][e][2][0])
                    
                    e1_inter_vec_err.append(gauss_no_noise[c][lam][e][0][1])
                    e2_inter_vec_err.append(gauss_no_noise[c][lam][e][1][1])
                    size_inter_vec_err.append(gauss_no_noise[c][lam][e][2][1])
                elif profile_type == 'optical':
                    e1_inter_vec.append(optical_no_noise[c][lam][e][0][0])
                    e2_inter_vec.append(optical_no_noise[c][lam][e][1][0])
                    size_inter_vec.append(optical_no_noise[c][lam][e][2][0])
                    
                    e1_inter_vec_err.append(optical_no_noise[c][lam][e][0][1])
                    e2_inter_vec_err.append(optical_no_noise[c][lam][e][1][1])
                    size_inter_vec_err.append(optical_no_noise[c][lam][e][2][1])
                elif profile_type == 'wfirst':
                    e1_inter_vec.append(wfirst_no_noise[c][lam][e][0][0])
                    e2_inter_vec.append(wfirst_no_noise[c][lam][e][1][0])
                    size_inter_vec.append(wfirst_no_noise[c][lam][e][2][0])
                    
                    e1_inter_vec_err.append(wfirst_no_noise[c][lam][e][0][1])
                    e2_inter_vec_err.append(wfirst_no_noise[c][lam][e][1][1])
                    size_inter_vec_err.append(wfirst_no_noise[c][lam][e][2][1])

                else:
        
                    print "Wrong 'profile_type'. "
                    sys.exit(1)
                print "x_vec, e1_inter_vec, e2_inter_vec, size_inter_vec", x_vec, e1_inter_vec[0], e2_inter_vec[0], size_inter_vec[0]
                print "e1_inter_vec_err[0], e2_inter_vec_err[0], size_inter_vec_err[0] * sqrt (n) ", sqrt_n* np.array(e1_inter_vec_err[0]), sqrt_n*np.array(e2_inter_vec_err[0]), sqrt_n*np.array(size_inter_vec_err[0])
                print "e1_inter_vec_err[0], e2_inter_vec_err[0], size_inter_vec_err[0]", e1_inter_vec_err[0], e2_inter_vec_err[0], size_inter_vec_err[0]
                print "PLOTTING: ", lam, c, e
                plot_function_e_and_r (fig, x_vec, e1_inter_vec[0], e2_inter_vec[0], size_inter_vec[0], y1_vec_err=e1_inter_vec_err[0], y2_vec_err=e2_inter_vec_err[0], y3_vec_err=size_inter_vec_err[0], x1label=x_label, lam_key=lam, e_key=e, title=" ")

        #plt.suptitle(string + "actual m:%g " %m, fontsize=11)
        #plt.suptitle( "m:%g " %m, fontsize=10)
        fig.tight_layout()
        plt.subplots_adjust(top=0.85)
        pp.savefig(fig)
    
    plt.close()



else:
    print "need to implement x_var: ", x_var



## Optical no noise, Delta_e, all filters
#fig = plt.figure()
#for lam in wavelength_dict:
#    for e in e_vec:
#        plot_function_r (fig, mag_gal_vec, gauss_no_noise[lam][e][2], xlabel="mag_object", lam_key=lam, e_key=e, type='o')
#plt.suptitle(string_o, fontsize=11)
#fig.tight_layout()
#plt.subplots_adjust(top=0.85)
#pp.savefig(fig)
#plt.close()




pp.close()






