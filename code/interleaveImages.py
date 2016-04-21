def interleaveImages(im_list,N,offsets=None,add_flux=True,suppress_warnings=False):
    """
    Interleaves two or more images and outputs a larger image. 

    The sampling length of simulated images can be set arbitrarily using the `pixel_scale' argument
    in drawImage() routine appropriately. However, pixel level detector effects can be included 
    only on images drawn at the native pixel scale, which are typically undersampled. Nyquist 
    sampled images that also include the effects of detector non-idealities can be obtained by
    drawing muliple undersampled images (with the detector effects included) that are offset from
    each other by a fraction of a pixel. This is equivalent to obtaining a finer sampled image from
    a dither sequence, except that this routine handles only equispaced offsets. The dither sequence
    must be a list of galsim.Images instances supplied through 'im_list'.

    @param im_list           A list containing the galsim.Image instances to be interleaved.
    @param N                 Number of images to interleave in either directions. It can be of type
                             `int' if equal number of images are interleaved in both directions or
                             a tuple of two integers, containing the number of images in x and y
                             directions respectively.
    @param offsets           A list containing the offsets as galsim.PositionD instances
                             corresponding to every image in `im_list'. The offsets must be equally
                             spaced and must span an entire pixel area. The offset values must
                             be symmetric around zero, hence taking positive and negative values.
                             The default offset ordering is to vary the offset in x from positive to
                             negative for every offset in y which should go from positive to 
                             negative. Providing `offsets' is highly recommended. [default:None]
    @param add_flux          Should the routine add the fluxes of all the images (True) or average
                             them (False)?
    @param suppress_warnings Suppresses the warnings about the pixel scale of the output, if True.

    @returns the interleaved image
    """
    
    import galsim
    import numpy as np
    
    if isinstance(N,int):
        n1,n2 = N,N
    elif isinstance(N,tuple):
        n1,n2 = N
        if not (isinstance(n1,int) and  isinstance(n2,int)):
            raise TypeError("'N' has to be of type int or a tuple of two integers")
    else:
        raise TypeError("'N' has to be of type int or a tuple of two integers")

    if len(im_list)<2:
        raise TypeError("'im_list' needs to have at least two instances of galsim.Image")

    if (n1*n2 != len(im_list)):
        raise ValueError("'N' is incompatible with the number of images in 'im_list'")

    if offsets is not None:
        if len(im_list)!=len(offsets):
            raise ValueError("'im_list' and 'offsets' must be lists of same length")
        for offset in offsets:
            if not isinstance(offset,galsim.PositionD):
                raise TypeError("'offsets' must be a list of galsim.PositionD instances")

    if isinstance(im_list[0],galsim.Image):
        y_size, x_size = im_list[0].array.shape
        scale = im_list[0].scale
    else:
        raise TypeError("'im_list' must be a list of galsim.Image instances")

    for im in im_list[1:]:
        if not isinstance(im,galsim.Image):
            raise TypeError("'im_list' must be a list of galsim.Image instances")

        if im.array.shape != (y_size,x_size):
            raise ValueError("All galsim.Image instances in 'im_list' must be of the same size")
 
        if im.scale != scale:
            raise ValueError("All galsim.Image instance in 'im_list' must have the same pixel scale")

    img_array = np.zeros((n2*y_size,n1*x_size))
    # The tricky part - going from (x,y) Image coordinates to array indices
    if offsets is None:
        # default offset settings
        for j in xrange(n2):
            for i in xrange(n1):
                img_array[j::n2,i::n1] = im_list[n1*j+i].array[:,:]
    else:
        # DX[i'] = -(i+0.5)/n+0.5 = -i/n + 0.5*(n-1)/n
        #    i  = -n DX[i'] + 0.5*(n-1)
        for k in xrange(len(offsets)):
            dx, dy = offsets[k].x, offsets[k].y
            i = int(round((n1-1)*0.5-n1*dx))
            j = int(round((n2-1)*0.5-n2*dy))
            img_array[j::n2,i::n1] = im_list[k].array[:,:]

    if add_flux is True:
        img = galsim.Image(img_array)
    else:
        img = galsim.Image((1.0/(len(im_list)))*img_array)

    if (n1==n2):
        if scale is not None:
            img.scale = im_list[0].scale*(1./n1)
    elif suppress_warnings is False:
        import warnings
        warnings.warn("Interleaved image could not be assigned a pixel scale automatically")
    return img
    
def test_interleaveImages():
    import numpy as np
    import galsim
    
    # 1) Interleave arrays as Images
    x_size, y_size = 16,10
    n1, n2 = 2, 5
    im_list = []
    for i in xrange(n1):
        for j in xrange(n2):
            x = np.arange(-x_size+2.*i/n1,x_size+2.*i/n1,2)
            y = np.arange(-y_size+2.*j/n2,y_size+2.*j/n2,2)
            X,Y = np.meshgrid(y,x)
            im = galsim.Image(100.*X+Y)
            im_list.append(im)

    im1 = interleaveImages(im_list,N=(n2,n1),suppress_warnings=True)
    x = np.arange(-x_size,x_size,2.0/n1)
    y = np.arange(-y_size,y_size,2.0/n2)
    X,Y = np.meshgrid(y,x)
    im2 = galsim.Image(100.*X+Y)
    np.testing.assert_array_almost_equal(im1.array,im2.array,decimal=11,\
        err_msg="Interleave failed for dummy image")
    assert im1.scale == None

    # 2a) With galsim Gaussian
    g = galsim.Gaussian(sigma=3.7,flux=1000.)
    gal = galsim.Convolve([g,galsim.Pixel(1.0)])
    im_list = []
    n = 2
    for j in xrange(n):
        for i in xrange(n):
            im = galsim.Image(16*n,16*n)
            offset = galsim.PositionD(-(i+0.5)/n+0.5,-(j+0.5)/n+0.5)
            gal.drawImage(image=im,scale=1.0,method='no_pixel',offset=offset)
            #gal.drawImage(image=im,scale=1.0,method='no_pixel',offset=galsim.PositionD(1.*i/n,1.*j/n))
            im_list.append(im)

    # Input to N as an int
    img = interleaveImages(im_list,n)
    im = galsim.Image(16*n*n,16*n*n)
    g = galsim.Gaussian(sigma=3.7,flux=1000.*n*n)
    gal = galsim.Convolve([g,galsim.Pixel(1.0)])
    gal.drawImage(image=im,method='no_pixel',offset=galsim.PositionD(0.0,0.0),scale=1.0/n)
    np.testing.assert_array_equal(img.array,im.array,\
        err_msg="Interleaved Gaussian images do not match")
    assert im.scale == img.scale

    # 2b) With im_list and offsets permuted
    offset_list = []
    # An elegant way of generating the default offsets
    DX = np.arange(0.0,-1.0,-1.0/n)
    DX -= DX.mean()
    DY = DX
    for dy in DY:
        for dx in DX:
            offset = galsim.PositionD(dx,dy)
            offset_list.append(offset)

    np.random.seed(42) # for generating the same random permutation everytime
    rand_idx = np.random.permutation(len(offset_list))
    im_list_randperm = [im_list[idx] for idx in rand_idx]
    offset_list_randperm = [offset_list[idx] for idx in rand_idx]
    # Input to N as a tuple
    img_randperm = interleaveImages(im_list_randperm,(n,n),offsets=offset_list_randperm)

    np.testing.assert_array_equal(img_randperm.array,img.array,\
        err_msg="Interleaved images do not match when 'offsets' is supplied")
    assert img_randperm.scale == img.scale

    # 3a) Increase resolution along one direction - square to rectangular images
    n = 2
    g = galsim.Gaussian(sigma=3.7,flux=100.)
    g1 = g.shear(g=1.*(n**2-1)/(n**2+1),beta=0.0*galsim.radians)
    gal1 = g1 #galsim.Convolve([g1,galsim.Pixel(1.0)])
    im_list = []
    offset_list = []
  
    # Generating offsets in a natural way
    DY = np.arange(0.0,1.0,1.0/(n*n))
    DY -= DY.mean()
    for dy in DY:
        im = galsim.Image(16,16)
        offset = galsim.PositionD(0.0,dy)
        offset_list.append(offset)
        gal1.drawImage(im,offset=offset,scale=1.0,method='no_pixel')
        im_list.append(im)

    img = interleaveImages(im_list,N=(1,n**2),offsets=offset_list,add_flux=False,suppress_warnings=True)
    im = galsim.Image(16,16*n*n)
    g = galsim.Gaussian(sigma=3.7*n,flux=100.)
    gal = galsim.Convolve([g,galsim.Pixel(1.0)])
    gal.drawImage(image=im,scale=1.,method='no_pixel')

    np.testing.assert_array_equal(im.array,img.array,err_msg="Sheared gaussian not interleaved correctly")
    assert img.scale is None

    # 3b) Increase resolution along one direction - rectangular to square images
    n = 2
    g = galsim.Gaussian(sigma=3.7,flux=100.)
    g2 = g.shear(g=1.*(n**2-1)/(n**2+1),beta=90.*galsim.degrees)
    gal2 = g2
    im_list = []
    offset_list = []

    # Generating offsets in a natural way
    DX = np.arange(0.0,1.0,1.0/n**2)
    DX -= DX.mean()
    for dx in DX:
         offset = galsim.PositionD(dx,0.0)
         offset_list.append(offset)
         im = galsim.Image(16,16*n*n)
         gal2.drawImage(im,offset=offset,scale=1.0,method='no_pixel')
         im_list.append(im)

    img = interleaveImages(im_list,N=(n**2,1),offsets=offset_list,suppress_warnings=True)
    im = galsim.Image(16*n*n,16*n*n)
    g = galsim.Gaussian(sigma=3.7,flux=100.*n*n)
    gal = g
    gal.drawImage(image=im,scale=1./n,method='no_pixel')

    np.testing.assert_array_equal(im.array,img.array,err_msg="Sheared gaussian not interleaved correctly")
    assert img.scale is None
    
if __name__=='__main__':
    test_interleaveImages()
