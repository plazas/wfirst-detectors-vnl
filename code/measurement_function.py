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
import matplotlib.patches as patches

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


def measurement_function(profile, noise=None, beta=3.566e-7, base_size='1024', type='nl', n='3', offset = (0.,0.), pixel_scale=0.11,  new_params=galsim.hsm.HSMParams(max_amoment=60000000, max_mom2_iter=10000000000,  max_moment_nsig2=25)):
    """
    This function receives a GSObject profile and applies one of the two sensor effects: 
    1) NL (cuadratic, with a single /beta parameter)
    2) BF (cdModel implemented in GalSim for CCDs. Antilogus et al. 2014)
    
    Return: None, it is a void function. But the input vectors e1_inter_vec=[], e2_inter_vec=[], size_inter_vec=[] should be filled.
    
    """
    
    
    #print "INSIDE meas. function: ", beta
    
    # Figure out how many times we are going to go through the whole rendering process
    # Even if n_offsets = 0, we are going to draw once.
    #if n_offsets == 0:
    #    n_iter = 1
    #else:
    #    n_iter = n_offsets
    
    draw_wfirst_psf=False
    #offset_input=(0.0, 0.0)

    if type == 'nl':
        method='oversampling'
        #method='interleaving'   # Just temporal
        #f=lambda  x,beta : x - beta*x*x*x*x
        f=lambda x,beta : x - beta*x*x
        #f=lambda x,beta : x - beta*x*x
        #f=lambda x, (b,g,d) : x + b*x*x + g*x*x*x + d*x*x*x*x
        #f=lambda x, b : x + b*x*x*x*x   #+ g*x*x*x + d*x*x*x*x


    elif type == 'bf':
        method='interleaving'
    else:
        print "ERROR in call to 'measurement_function': wrong type (nor 'nl' nor 'bf')"
        sys.exit(1)


    if method == 'oversampling':   ## NL does not need interleaving
        print "METHOD: oversampling"

        #### Calculate moments without effect
        print "Applied FLUX in electrons: ", profile.getFlux()
        
        # Do several realizations at differen centroid offsets
        
        """
        vec_ud=[]
        for ind in range(n_iter):
            ud=galsim.UniformDeviate()
            vec_ud.append(ud)
            if n_offsets > 0:
                offset=(ud(), ud())
                # For the high-res image, have to find how many high-res pixels the offset is, and then take
                # only the sub-pixel part.
                offset_highres = (offset[0]*n % 1, offset[1]*n % 1)
            else:
                offset = (0., 0.)
                offset_highres = (0., 0.)
    
            image=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64 ), scale=pixel_scale/n, method='no_pixel', offset=offset_highres)
            #print "Maximum flux: ", np.amax(image.array)
            #print "Flux of image after being drawn (using np.sum(image.array)): ", np.sum(image.array)
            #print image.array.shape
            image_mult=(n*n)*image
            #print "Maximum flux: ", np.amax(image.array)
            #print "Flux of image after adjusting n*n(using np.sum(image.array)): ", np.sum(image.array)
            if ind == 0:
                image_sum= image_mult
            else:
                image_sum+=image_mult
                

        image=image_sum/float(n_iter)
        """
        offset= (0.0, 0.0)
        print "Offset: ", offset
        image=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale/n, method='no_pixel', offset=offset)

        image=(n*n)*image
        #image
        #sys.exit()
        
        #if not noise == None:
        #    read_noise = galsim.GaussianNoise(sigma=noise/(n**2))
        #    image.addNoise(read_noise)
        # IMAGE
        if draw_wfirst_psf == True:
            k=64
            delta=15
            bm, bp = 0.5*(1.5*k)-delta, 0.5*(1.5*k) + delta
            bounds=galsim.BoundsI(bm,bp,bm,bp)
            before=image[bounds].array

            fig=plt.figure()
            ax=fig.add_subplot(223)
            plt.imshow(before, cmap='cubehelix', norm=LogNorm(), interpolation='nearest' )
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title ('Y106: no NL (core)')
            plt.colorbar()

            before_all=image.array
            ax=fig.add_subplot(221)
            plt.imshow((before_all), cmap='cubehelix', norm=LogNorm(), interpolation='nearest')
            plt.colorbar()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title ('Y106: no NL (full stamp)')
            ax.add_patch( patches.Rectangle( (0.5*(1.5*k) - delta, 0.5*(1.5*k) - delta),2*delta,2*delta, fill=False))

        
        #IMAGE

        results=image.FindAdaptiveMom(hsmparams=new_params)
        ref_e1=results.observed_shape.e1
        ref_e2=results.observed_shape.e2
        ref_s=results.moments_sigma
        #print "Image shape, before interleave: ", image.array.shape
        print "ref_e1, ref_e2, ref_s", ref_e1, ref_e2, ref_s
        
        #### Calculate moments with the effect
        
        
        """
        # Do several realizations at differen centroid offsets
        for ind in range(n_iter):
            #ud=galsim.UniformDeviate()
            ud=vec_ud[ind]
            if n_offsets > 0:
                offset=(ud(), ud())
                # For the high-res image, have to find how many high-res pixels the offset is, and then take
                # only the sub-pixel part.
                offset_highres = (offset[0]*n % 1, offset[1]*n % 1)
            else:
                offset = (0., 0.)
                offset_highres = (0., 0.)
            
            image=profile.drawImage(image=galsim.Image(base_size, base_size), scale=pixel_scale/n, method='no_pixel', offset=offset_highres)
            #print "Maximum flux: ", np.amax(image.array)
            #print "Flux of image after being drawn (using np.sum(image.array)): ", np.sum(image.array)
            #print image.array.shape
            image_mult=(n*n)*image
            #print "Maximum flux: ", np.amax(image.array)
            #print "Flux of image after adjusting n*n(using np.sum(image.array)): ", np.sum(image.array)
            if ind == 0:
                image_sum= image_mult
            else:
                image_sum+=image_mult

        image=image_sum/float(n_iter)
        """
        image=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale/n, method='no_pixel', offset=offset)
        image=(n*n)*image
        image.applyNonlinearity(f,beta)
        #sys.exit()
        print "Flux of image after VNL (using np.sum(image.array)): ", np.sum(image.array)


        if draw_wfirst_psf == True:
            # IMAGE
            after=image[bounds].array
            print "drawing fractional difference "
            diff= (before - after)/before # VNL attenuates
            ax=fig.add_subplot(122)
            #ax=plt.subplot2grid ( (2,2), (1,1), colspan=2 )
            #ax.set_position ([0.1, 0.5, 0.5, 0.5])
            #print "diff: ", diff
            #sys.exit()
            plt.imshow((diff), cmap='gnuplot2', norm=LogNorm(), interpolation='nearest')
            plt.colorbar()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title ('NL vs no NL: \n fractional difference (core)')

            plt.tight_layout()
            pp.savefig()
            pp.close()
            sys.exit()
        
        #IMAGE


        #results=image.FindAdaptiveMom(hsmparams=new_params)
        #print "results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma ", results.observed_shape.e1, results.observed_shape.e2, results.moments_sigma
        #print "Differences: ", results.observed_shape.e1 - ref_e1, results.observed_shape.e2 - ref_e2, (results.moments_sigma - ref_s) / ref_s
        # Output values
        #e1_out=results.observed_shape.e1 - ref_e1
        #e2_out=results.observed_shape.e2 - ref_e2
        #size_out=(results.moments_sigma - ref_s) / ref_s
        #return e1_out, e2_out, size_out


        results=image.FindAdaptiveMom(hsmparams=new_params)
        obs_e1=(results.observed_shape.e1)
        obs_e2=(results.observed_shape.e2)
        obs_s=(results.moments_sigma)



        d_e1=(obs_e1 - ref_e1)
        d_e2=(obs_e2 - ref_e2)
        d_s=(obs_s/ref_s -1.)
        print "obs_e1: %.16g, obs_e2 : %.16g, obs_s: %.16g" %(obs_e1, obs_e2, obs_s)
        print "Differences: d_e1: %.16g, d_e2 : %.16g, d_s: %.16g" %(d_e1, d_e2, d_s)
        
        # Output values
        e1_out= d_e1
        e2_out= d_e2
        size_out= d_s

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
                im=galsim.Image(base_size, base_size, dtype=np.float64)
                offset=galsim.PositionD(offset_input[0] - (i+0.5)/n+0.5, offset_input[1] - (j+0.5)/n+0.5)     ## Add randon uniform offset
                offsets_list.append(offset)
                #print "Offset: ", offset
                profile.drawImage(image=im, scale=pixel_scale, offset=offset, method='no_pixel')
                im_list.append(im)


        image=galsim.utilities.interleaveImages(im_list=im_list, N=(n,n), offsets=offsets_list, add_flux=True)
        print "Image shape, after interleave, no effect: ", image.array.shape
        print "Flux of image after interleave (using np.sum(image.array)): ", np.sum(image.array)
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
                im=galsim.Image(base_size, base_size, dtype=np.float64)
                offset=galsim.PositionD(offset_input[0]  -(i+0.5)/n+0.5, offset_input[1] - (j+0.5)/n+0.5)
                offsets_list.append(offset)
                #print "Offset: ", offset
                im=profile.drawImage(image=im, scale=pixel_scale, offset=offset, method='no_pixel')
                if type == 'bf':
                    #cd = PowerLawCD(5, 5.e-7, 5.e-7, 1.5e-7, 1.5e-7, 2.5e-7, 2.5e-7, 1.3)
                    (aL,aR,aB,aT) = readmeanmatrices()
                    cd = galsim.cdmodel.BaseCDModel (aL,aR,aB,aT)
                    im=cd.applyForward(im)
                elif type == 'nl':
                    im.applyNonlinearity(f,beta)
                im_list.append(im)


        image2=galsim.utilities.interleaveImages(im_list=im_list, N=(n,n), offsets=offsets_list, add_flux=True)
        print "Image shape, after interleave: ", image2.array.shape
        print "Flux of image after interleave (using np.sum(image.array)): ", np.sum(image.array)
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


