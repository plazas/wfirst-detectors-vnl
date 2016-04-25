import numpy as np

def wcentroid(image, xcoords, ycoords, sigma=None, tol=1.e-4, maxnits=100, info=False, warn=True):
    """Find the weighted (or unweighted if sigma is not set) centroid of an
    image.

    This is defined relative to user-specified x and y coordinates for the
    image, specifed as arrays, as generated by (e.g.) np.meshgrid.
    
    Iterates until |(dx, dy)| < tol (default tol = 1.e-4) pixels, up until
    maxnits iterations (default maxnits = 100).
    
    Supplied xcoords & ycoords must be same size as image or an exception is
    raised.

    Returns x, y, dx, dy, weight_c, xcoords_c, ycoords_c
    (last three of these are: weight function image, new xcoords and new
    ycoords.  New coords have been recentred using the new centroid).
    """
    # Handle exceptions for shape mismatches    
    if xcoords.shape != image.shape:
        raise ValueError('Shape of supplied XCOORDS does not match input image')
    if ycoords.shape != image.shape:
        raise ValueError('Shape of supplied YCOORDS does not match input image')
    # First, if no sigma is specified just calculate unweighted centroid
    if sigma == None:
        imsum = image.sum()
        xc = (xcoords * image).sum() / imsum
        yc = (ycoords * image).sum() / imsum
        weight = np.ones(image.shape) # define this for output
        dxc, dyc = 0., 0.
        xnew, ynew = xcoords, ycoords
    # Otherwise, proceed to iteratively find the weighted centroid
    else:
        sigma2 = sigma**2
        xc, yc, dxc, dyc = 0., 0., 0., 0.
        nits = 0
        eps2 = 1.e7
        # Define the functions without dots for a touch extra speed...
        pi = np.pi
        npexp = np.exp
        npsum = np.sum
        # Small centroid iteration boost factor to compensate for weighting (improves speed)
        boost = 4. / 3.
        while eps2 > tol**2 and nits <= maxnits:
	    xnew, ynew = xcoords - xc, ycoords - yc
	    weight = npexp(-.5 * (xnew**2 + ynew**2) / sigma2)
            wimsum = npsum(image * weight)
            dxc = boost * npsum(xnew * image * weight) / wimsum
            dyc = boost * npsum(ynew * image * weight) / wimsum
            xc, yc = xc + dxc, yc + dyc
            eps2 = dxc**2 + dyc**2
            if info == True:
                print 'wcentroid: xc = %e +- %e  yc = %e +- %e' % \
                      (xc, dxc, yc, dyc)
            if nits == maxnits and warn == True:
                print 'wcentroid warning: MAXNITS reached in iterative centroid search!'
            nits += 1
    return xc, yc, dxc, dyc, weight, xnew, ynew

def crapsky(image, nedge=3):
    """Approximately estimate the sky background in an image from the median in an nedge wide
    pixel border.
    """
    x, y = np.meshgrid(np.arange(image.shape[1], dtype=float),
                       np.arange(image.shape[0], dtype=float))
    hwx, hwy = .5 * float(image.shape[1] - 1), .5 * float(image.shape[0] -1)
    x -= hwx  # subtract the half width to centre x and y
    y -= hwy
    # Can use "+" to behave like "or" for array bools...
    sky = image[((abs(x) > (hwx - float(nedge))) + (abs(y) > (hwy - float(nedge))))]
    return np.median(sky)

def wmoments(image, xcoords, ycoords, weight):
    """Calculate the weighted quadrupole moments of a supplied image.

    Uses a supplied, assumed pre-centroided weight array, and matching size
    x and y coordinate arrays.

    Outputs (Ixx, Ixy, Iyy)   [flux normalized moments in units pixels^2]
    """
    # Handle exceptions for shape mismatches
    if xcoords.shape != image.shape:
        raise ValueError('Shape of supplied XCOORDS does not match input image')
    if ycoords.shape != image.shape:
        raise ValueError('Shape of supplied YCOORDS does not match input image')
    if weight.shape != image.shape:
        raise ValueError('Shape of supplied WEIGHT does not match input image')
    Ixx = (xcoords * xcoords * image * weight).sum() / (image * weight).sum()
    Ixy = (xcoords * ycoords * image * weight).sum() / (image * weight).sum()
    Iyy = (ycoords * ycoords * image * weight).sum() / (image * weight).sum()
    return Ixx, Ixy, Iyy

def measure_moments(image, sigma=None, xc_guess=None, yc_guess=None, info=False,
                    warn=False, tol=1.e-4, maxnits=100 ,skysub=True):
    """Measure weighted second moments of an image.
    
    Supplied an image, and an x,y centroid guess specified relative to the lower left pixel centre
    (default = image centre), this code finds a better centroid estimate from wcentroid() and uses
    this to calculate image moments using wmoments().

    If sigma is set, measure_moments() calculates weighted moments with a Gaussian of width sigma,
    but uses no weight if sigma is not set.    Has basic checks to reject all-zero images.

    Outputs (xcentroid, ycentroid, dxcentroid, dycentroid, Ixx, Ixy, Iyy)

    All dimensions in pixels or pixels^2.

    NO warning is given if centroid drifts away from the edge (often a sign of failure)!  Check
    the output...
    """
    imss = image.squeeze() # remove flat dimensions if any 
                           #(useful for multiprocessing when taking split cube input)
    if skysub is True: imss -= crapsky(image, nedge=3)  #Sky subtraction
    # Test that image is not empty
    if (imss == 0.).all() == True:
        if warn == True:
            print "Refusing to measure moments for all-zero image"
        return 0., 0., 0., 0., 0., 0., 0.
    else:
        x, y = np.meshgrid(np.arange(imss.shape[1] ,dtype=float),
                           np.arange(imss.shape[0] ,dtype=float))
        # If x and y centroid guess is given use that to set origin of coords, otherwise just use
        # image center
        if xc_guess == None:
            xorigin = .5 * float(imss.shape[1] - 1)
        else:
            xorigin = xc_guess
        if yc_guess == None:
            yorigin = .5 * float(imss.shape[0] - 1)
        else:
            yorigin = yc_guess
    # Subtract off these origins
        x -= xorigin
        y -= yorigin
    # Iteratively find the centroid for the supplied weight, relative to this origin
        xc, yc, dxc, dyc, weight, xnew, ynew = wcentroid(imss, x, y, info=info,
                                                         sigma=sigma, tol=tol,
                                                         maxnits=maxnits, warn=warn)
    # Then get the weighted second moments
        Ixx, Ixy, Iyy = wmoments(imss, xnew, ynew, weight)    
        if info == True:
            print 'measure_moments: Ixx = %e Ixy = %e Iyy = %e' % (Ixx, Ixy, Iyy)
        return xorigin + xc, yorigin + yc, dxc, dyc, Ixx, Ixy, Iyy

def measure_e1e2R2(image, sigma=None, xc_guess=None, yc_guess=None, info=False,
                   warn=False, tol=1.e-4, maxnits=100 ,skysub=True):
    """Measure elliptiticies and radius^2 from weighted second moments of an image.

    Wrapper for measure_moments() that does the extra step of generating an ellipticity measure
    (RRG definition) and <R^2> from the output moments.

    If sigma is set, calculates weighted moments with a Gaussian of width sigma, but uses no weight
    if sigma is not set.  Has basic checks to reject all-zero images.

    Outputs (e1, e2, R^2, xcentroid, ycentroid, dxcentroid, dycentroid)

    All dimensions in pixels or pixels^2.

    NO warning is given if centroid drifts away from the edge (often a sign of failure)!  Check
    the output...
    """
    #crapsky() and squeeze() already being called in measure_moments()
    #imss = (image - crapsky(image, nedge=3)).squeeze()  # sky sub and remove flat dimensions if any
                                         # (useful for multiprocessing when taking split cube input)
    xc, yc, dxc, dyc, Ixx, Ixy, Iyy = measure_moments(image, sigma=sigma,
                                                      xc_guess=xc_guess,
                                                      yc_guess=yc_guess,
                                                      info=info, warn=warn,
                                                      tol=tol, maxnits=maxnits,
                                                      skysub=skysub)
    if (Ixx == 0. and Iyy == 0.): # Ensure we are not dividing by zero in case of an empty image
        R2 = 0.
        e1 = 0.
        e2 = 0.
    else:
        R2 = Ixx + Iyy
        e1 = (Ixx - Iyy) / R2
        e2 = 2. * Ixy / R2
    if info == True:
        print 'measure_e1e2R2: e1 = %e e2 = %e R^2 = %e' % (e1, e2, R2)
    return e1, e2, R2, xc, yc, dxc, dyc

def measure_e1e2R2_cube(cube, sigma=None, xc_guess=None, yc_guess=None, info=False,
                        warn=False, tol=1.e-4, maxnits=100 ,skysub=True ,multi=False):
    """Apply measure_e1e2R2() to a cube of spot images.

    Outputs (e1, e2, R^2, xcentroid, ycentroid, dxcentroid, dycentroid), each as numpy arrays
    of length N where N is the number of spots in the cube.
    """
    from functools import partial
    worker_func = partial(measure_e1e2R2, sigma=sigma, xc_guess=xc_guess, yc_guess=yc_guess,
                          info=info, warn=warn, tol=tol, maxnits=maxnits ,skysub=skysub)

    if multi:
        from multiprocessing import Pool
        p = Pool()
        mymap = p.map
    else:
        mymap = map

    output = np.array(mymap(worker_func, np.vsplit(cube, cube.shape[0])))

    e1, e2, R2, xc, yc, dxc, dyc = (output[:, 0], output[:, 1], output[:, 2], output[:, 3],
                                    output[:, 4], output[:, 5], output[:, 6])

    if multi:
        p.close()
        p.join()

    return e1, e2, R2, xc, yc, dxc, dyc

def measure_e1e2R2_cube_multi(cube, sigma=None, xc_guess=None, yc_guess=None, info=False,
                              warn=False, tol=1.e-4, maxnits=100 ,skysub=True):
    """Apply measure_e1e2R2() to a cube of spot images using multiple
    cores to improve performance.

    Outputs (e1, e2, R^2, xcentroid, ycentroid, dxcentroid, dycentroid), each as numpy arrays
    of length N where N is the number of spots in the cube.

    Verified that same output is created as non-threaded version using md5sum.
    """
    from multiprocessing import Pool
    from functools import partial
    worker_func = partial(measure_e1e2R2, sigma=sigma, xc_guess=xc_guess, yc_guess=yc_guess,
                          info=info, warn=warn, tol=tol, maxnits=maxnits ,skysub=skysub)
    p = Pool()
    output = np.array(p.map(worker_func, np.vsplit(cube, cube.shape[0])))
    e1, e2, R2, xc, yc, dxc, dyc = (output[:, 0], output[:, 1], output[:, 2], output[:, 3],
                                    output[:, 4], output[:, 5], output[:, 6])
    p.close()
    p.join()
    return e1, e2, R2, xc, yc, dxc, dyc

def sim_reduce_ellips(path='.' ,groupbyfile=False):
    """Load ellips_ files generated by reduce.py and calculate basic stats on them.

    Outputs: means, stds, maxes and mins of (e1, e2, R2, xc, yc, dxc, dyc)

    Makes assumptions about simulation, (spots arranged in square)
    
    When groupbyfile=True, e1 e2 etc. are tabulated by file and averaged over all positions in the file
    When groupbyfile=False, e1 e2 etc. are tabulated by position and averaged over all files
    """
    import glob
    import math
    from sys import exit
    
    #Find all ellips_ files in PWD and load filenames into string
    filelist=glob.glob(path+'/ellips*.npy')
    if len(filelist) is 0:
        exit("sim_reduce_ellipse: No files found matching ellips*.npy")
    filelist.sort()
   
    testfile=np.load(filelist[0])
    nspots = testfile.shape[0]
    nside = math.sqrt(nspots)   #Assumes square!
    del testfile

    #This creates a 1D array of indices for a square with the border removed
    box=np.arange(nside**2 ,dtype=int).reshape(nside,nside)
    box=box[1:nside-1,1:nside-1].ravel()

    #Load each ellips_ file and add contents to master table
    big_ellips_tab=[]

    for ellips_file in filelist:
        ellips_tab = np.load(ellips_file)
        ellips_tab = ellips_tab[box ,:]
        big_ellips_tab.append(ellips_tab)

    big_ellips_tab=np.array(big_ellips_tab)

    #Calculate basic stats for all files
    #Results will have same dimensions as one of the files if groupbyfile=False
    
    if groupbyfile:
        axis = 1
    else:
        axis = 0
        
    Navg = big_ellips_tab.shape[axis]

    means = np.mean(big_ellips_tab ,axis=axis).transpose()
    medians = np.median(big_ellips_tab ,axis=axis).transpose()
    stds = np.std(big_ellips_tab ,axis=axis).transpose()
    maxes = np.amax(big_ellips_tab ,axis=axis).transpose()
    mins = np.amin(big_ellips_tab ,axis=axis).transpose()
    # return means ,medians ,stds ,maxes ,mins
    return {'means':means ,'medians':medians ,'stds':stds ,'maxes':maxes ,'mins':mins ,'N':Navg}

def sim_cubify(file ,stampsize):
	"""Load simulated FITS file and reshape into a cube of postage stamps.
	
	Outputs: numpy array of shape (Nstamps,stampsize,stampsize)
	
	Makes assumptions about simulation, (square stamps arranged in square grid)
	"""
	import pyfits

	image = pyfits.getdata(file)		# shape is (ngrid*stampsize ,ngrid*stampsize)
	ngrid = image.shape[0]/stampsize	# Assumes image and grid are squares

	cube=np.array(np.array_split(image ,ngrid ,axis=0)) # shape is (ngrid ,stampsize ,ngrid*stampsize)
	cube=np.array(np.array_split(cube ,ngrid ,axis=2))	# shape is (ngrid ,ngrid ,stampsize ,stampsize)
	cube=cube.reshape((-1 ,stampsize ,stampsize))		# shape is (ngrid**2 ,stampsize ,stampsize)

	#This creates a 1D array of indices for a square with the border removed
	box=np.arange(ngrid**2 ,dtype=int).reshape(ngrid,ngrid)
	box=box[1:ngrid-1,1:ngrid-1].ravel()

	cube=cube[box]
	
	return cube

def sim_rm_border(cube):
	import math
	ngrid=math.sqrt(cube.shape[0])
	
	#This creates a 1D array of indices for a square with the border removed
	box=np.arange(ngrid**2 ,dtype=int).reshape(ngrid,ngrid)
	box=box[1:ngrid-1,1:ngrid-1].ravel()

	return cube[box]