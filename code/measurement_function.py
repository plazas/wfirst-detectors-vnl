#!/usr/bin/python
import numpy as np
import sys
from sim2 import *   ## where all the BF stuff is; for the four 'a' matrices
import galsim

import matplotlib
matplotlib.use('Pdf')


import matplotlib.cm as cm  # color bar, to plot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf  import PdfPages


pp=PdfPages("wfirst_psf.pdf")

def my_imshow(im, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return '%8e @ [%4i, %4i]' % (im[y, x], x, y)
        except IndexError:
            return ''
    img = ax.imshow(im, **kwargs)
    ax.format_coord=format_coord
    return img


def measurement_function(profile, noise=None, beta=3.566e-7, base_size='1024', type='nl', n='6', pixel_scale=0.11,  new_params=galsim.hsm.HSMParams(max_amoment=60000000, max_mom2_iter=10000000,  max_moment_nsig2=10000)):
    """
    This function receives a GSObject profile and applies one of the two sensor effects: 
    1) NL (cuadratic, with a single /beta parameter)
    2) BF (cdModel implemented in GalSim for CCDs. Antilogus et al. 2014)
    
    Return: None, it is a void function. But the input vectors e1_inter_vec=[], e2_inter_vec=[], size_inter_vec=[] should be filled.
    
    """

    if type == 'nl':
        method='oversampling'
        f=lambda x,beta : x - beta*x*x
    elif type == 'bf':
        method='interleaving'
    else:
        print "ERROR in call to 'measurement_function': wrong type (nor 'nl' nor 'bf')"
        sys.exit(1)


    if method == 'oversampling':   ## NL does not need interleaving
        print "METHOD: oversampling"

        #### Calculate moments without effect
        print "Applied FLUX in electrons: ", profile.getFlux()
        image=profile.drawImage(image=galsim.Image(base_size, base_size), scale=pixel_scale/n, method='no_pixel')
        print "Maximum flux: ", np.amax(image.array)
        print "Flux of image after being drawn (using np.sum(image.array)): ", np.sum(image.array)
        print image.array.shape
        image*=n*n
        print "Maximum flux: ", np.amax(image.array)
        print "Flux of image after adjusting n*n(using np.sum(image.array)): ", np.sum(image.array)
        #image
        #sys.exit()
        
        #if not noise == None:
        #    read_noise = galsim.GaussianNoise(sigma=noise/(n**2))
        #    image.addNoise(read_noise)


        """
        before=image.array
        fig=plt.figure()
        ax=fig.add_subplot(221)
        plt.imshow(before, cmap='cubehelix', norm=LogNorm())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title ('J129: no VNL')
        plt.colorbar()
        """

        results=image.FindAdaptiveMom(hsmparams=new_params)
        ref_e1=results.observed_shape.e1
        ref_e2=results.observed_shape.e2
        ref_s=results.moments_sigma
        print "Image shape, before interleave: ", image.array.shape
        print "ref_e1, ref_e2, ref_s", ref_e1, ref_e2, ref_s

        #### Calculate moments with the effect

        image=profile.drawImage(image=galsim.Image(base_size, base_size), scale=pixel_scale/n, method='no_pixel')
        if not noise == None:
            read_noise = galsim.GaussianNoise(sigma=noise/(n**2))
            image.addNoise(read_noise)
        print "Flux of image after being drawn (using np.sum(image.array)): ", np.sum(image.array)

        #image*=n*n
        #print "Image.added_flux after image/=n*n", image.added_flux
        image*=n*n   ###  need to adjust flux per pixel

        image.applyNonlinearity(f,beta)
        #sys.exit()
        print "Flux of image after VNL (using np.sum(image.array)): ", np.sum(image.array)

        """
        #my_imshow(image.array)
        #plt.imshow(image.array, cmap='gray', norm=LogNorm())
        #plt.colorbar()
        #pp.savefig()
        after=image.array
        ax=fig.add_subplot(222)
        plt.imshow(after, cmap='cubehelix', norm=LogNorm())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title ('J129: with VNL')
        plt.colorbar()


        diff= (before - after)/before # VNL attenuates
        #ax=fig.add_subplot(212)
        ax=plt.subplot2grid ( (2,2), (1,1), colspan=2 )
        #ax.set_position ([0.1, 0.5, 0.5, 0.5])
        plt.imshow(diff, cmap='cubehelix', norm=LogNorm())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title ('fractional difference')
        plt.colorbar()

        plt.tight_layout()
        pp.savefig()


        pp.close()
        sys.exit()
        """


        results=image.FindAdaptiveMom(hsmparams=new_params)
        print "results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma ", results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma
        print "Differences: ", results.observed_shape.e1 - ref_e1, results.observed_shape.e2 - ref_e2, (results.moments_sigma - ref_s) / ref_s
        # Output values
        e1_out=results.observed_shape.e1 - ref_e1
        e2_out=results.observed_shape.e2 - ref_e2
        size_out=(results.moments_sigma - ref_s) / ref_s
        return e1_out, e2_out, size_out

    if method == 'interleaving':
        print "METHOD: Interleaving"

        ## Interleave the profiles with NO EFFECT
        im_list=[]
        offsets_list=[]
        #create list of images to interleave-no effect
        # First, draw a big image from which to obtain n^2 subimages.  This is to avoid calling draImage n^2 times
        big_image=galsim.Image
        
        
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
                    (aL,aR,aB,aT) = readmeanmatrices()
                    cd = galsim.cdModel.BaseCDModel (aL,aR,aB,aT)
                    im=cd.applyForward(im)
                im_list.append(im)


        image2=galsim.utilities.interleaveImages(im_list=im_list, N=(n,n), offsets=offsets_list, add_flux=True)
        print "Image shape, after interleave: ", image2.array.shape
        if not noise == None:
            read_noise = galsim.GaussianNoise(sigma=noise)
            image2.addNoise(read_noise)
        results=image2.FindAdaptiveMom(hsmparams=new_params)
        print "results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma ", results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma
        print "Differences: ", results.observed_shape.e1 - ref_e1, results.observed_shape.e2 - ref_e2, (results.moments_sigma - ref_s) / ref_s
        #e1_inter_vec.append  (results.observed_shape.e1 - ref_e1)
        #e2_inter_vec.append  (results.observed_shape.e2 - ref_e2)
        #size_inter_vec.append ( (results.moments_sigma - ref_s) / ref_s)

        # Output values
        e1_out=results.observed_shape.e1 - ref_e1
        e2_out=results.observed_shape.e2 - ref_e2
        size_out=(results.moments_sigma - ref_s) / ref_s

        return e1_out, e2_out, size_out


