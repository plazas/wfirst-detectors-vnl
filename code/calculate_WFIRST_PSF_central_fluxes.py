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
logger = logging.getLogger("tests_hsm_interleaving")


from scipy import optimize
import galsim


import galsim.wfirst as wfirst
filters = wfirst.getBandpasses (AB_zeropoint=True)

N=1
k=64
base_size=1.5*k + 3  ## odd, so peak flux is in one pixel
pixel_scale=0.11/N
flux_dict={}
new_params = galsim.hsm.HSMParams(max_amoment=60000000, max_mom2_iter=1000000000,  max_moment_nsig2=25)
big_fft_params = galsim.GSParams(maximum_fft_size=int(512*k))
e=(0.,0.)

index_dict={}
i=0
bandpass=['Y106', 'J129', 'F184', 'H158']
mags=[18.3]
for lam in bandpass:
    for mag in mags:
        index_dict[i]=(lam,mag)
        i+=1
        
        
def flux_function (index):
        lam, mag = index_dict[index]
        f=filters[lam]
        b=galsim.sed.SED(f)
        c=b.withMagnitude(mag, f)
        gal_flux=c.calculateFlux(f)
        gal=galsim.Convolve( wfirst.getPSF(SCAs=18,approximate_struts=False, wavelength=filters[lam])[18].withFlux(gal_flux), galsim.Pixel(pixel_scale), gsparams=big_fft_params)
        im=gal.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale, method='no_pixel')
        max_val=np.max(im.array)
        #print "bounds: ", im.bounds
        #print "center: ", im.center
        central_flux=im(im.center())
        #print "bandpass: %s, mag: %g, total_flux: %g, max_val: %g, cenral flux: %g " % (lam, mag, gal_flux, central_flux, max_val)
        return lam, mag, gal_flux, central_flux, max_val



from multiprocessing import cpu_count
from multiprocessing import Pool

processes=cpu_count()
use_cores=6
print "I have ", processes, "cores here. Using: %g" %use_cores
pool = Pool(processes=use_cores)

results=pool.map (flux_function, range(len(mags)*len(bandpass)))
for line in results:
    print "bandpass, mag, total_flux, central flux, max_val: ", line


