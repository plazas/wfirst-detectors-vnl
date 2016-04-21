+def test_interleave():
+ # 1) Dummy image
+ x_size, y_size = 16,10
+ n1, n2 = 2, 5
+ im_list = []
+ for i in xrange(n1):
+ for j in xrange(n2):
+ x = np.arange(-x_size+2.*i/n1,x_size+2.*i/n1,2)
+ y = np.arange(-y_size+2.*j/n2,y_size+2.*j/n2,2)
+ X,Y = np.meshgrid(y,x)
+ im = galsim.Image(100.*X+Y)
+ im_list.append(im)
+
+ im1 = interleave(im_list,n2,n1)
+ x = np.arange(-x_size,x_size,2.0/n1)
+ y = np.arange(-y_size,y_size,2.0/n2)
+ X,Y = np.meshgrid(y,x)
+ im2 = galsim.Image(100.*X+Y)
+
+ print "Size of 'im1' = ", im1.array.shape
+ print "Size of 'im2' = ", im2.array.shape
+
+ #fig,ax = plt.subplots(1)
+ #ax.hist(im1.array.reshape(n1*n2*x_size*y_size,1)-im2.array.reshape(n1*n2*x_size*y_size,1),20)
+ #plt.show()
+ np.testing.assert_array_almost_equal(im1.array,im2.array,decimal=11,err_msg="Interleave failed")
+
+ # 2) With galsim Gaussian
+ g = galsim.Gaussian(sigma=3.7,flux=1000.)
+ gal = g#alsim.Convolve([g,galsim.Pixel(1.0)])
+ im_list = []
+ n = 1
+ for j in xrange(n):
+ for i in xrange(n):
+ im = galsim.Image(16*n,16*n)
+ gal.drawImage(image=im,scale=1.0,method='no_pixel',offset=galsim.PositionD(-(i+0.5)/n+0.5,-(j+0.5)/n+0.5))
+ im_list.append(im)
+
+ img = interleave(im_list,n,n)
+ print "Size of the Gaussian = ", img.FindAdaptiveMom().moments_sigma/n, img.FindAdaptiveMom().moments_centroid, img.array.max()
+ im = galsim.Image(16*n*n,16*n*n)
+ g = galsim.Gaussian(sigma=3.7,flux=1000.*n*n)
+ gal = g#alsim.Convolve([g,galsim.Pixel(1.0)])
+ gal.drawImage(image=im,method='no_pixel',offset=galsim.PositionD(0.0,0.0),scale=1.0/n)
+ print " must match this - ", im.FindAdaptiveMom().moments_sigma/n, im.FindAdaptiveMom().moments_centroid, im.array.max()
+ print "Central part of the images:"
+ print np.round(img.array[8*n*n-2:8*n*n+3,8*n*n-2:8*n*n+3],3)
+ print np.round(im.array[8*n*n-2:8*n*n+3,8*n*n-2:8*n*n+3],3)
+ np.testing.assert_array_equal(img.array,im.array,err_msg="Gaussian images dont match")
+
+ # 3) With actual WFIRST PSFs
+ import galsim.wfirst as wfirst
+ filters = wfirst.getBandpasses(AB_zeropoint=True)
+ base_size = 32
+ n = 2
+ img_size = n*base_size
+ img = galsim.Image(img_size,img_size)
+ superimg = galsim.Image(img_size,img_size)
+
+ for filter_name, filter_ in filters.iteritems():
+ im_list = []
+ PSFs = wfirst.getPSF(SCAs=7,approximate_struts=True,wavelength=filter_)
+ PSF = galsim.Convolve([PSFs[7],galsim.Pixel(wfirst.pixel_scale)])
+ PSF.drawImage(image=superimg,scale=wfirst.pixel_scale/n,method='no_pixel')
+ print "Max pix val = ", superimg.array.max()
+ for j in xrange(n):
+ for i in xrange(n):
+ im = galsim.Image(base_size,base_size)
+ offset = galsim.PositionD(-(i+0.5)/n+0.5,-(j+0.5)/n+0.5)
+ PSF.drawImage(image=im,scale=wfirst.pixel_scale,method='no_pixel',offset=offset)
+ im_list.append(im)
+
+ img = interleave(im_list,n,n)
+ np.testing.assert_array_almost_equal(img.array,n*n*superimg.array,decimal=6,err_msg='WFIRST PSFs disagree for '+filter_name)
+ print "Test passed for "+filter_name
