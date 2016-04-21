#!/usr/bin/python
# sim1: simulate effect of B/F on shape measurement if it's not accounted for

import numpy as np
import os
import sys
import math

def secondmoments ( stamp ):
# calculates second moments and returns list [ q11 q22 q12 ]
# this is less elegant than it could be
# stamp needs to be a background-subtracted image
	moments = [ 0, 0, 0, 0, 0, 0 ] # q11 q22 q12 norm x y
        centx = (stamp.xmin+stamp.xmax+1)/2.
        centy = (stamp.ymin+stamp.ymax+1)/2.
	for x in range(stamp.xmin,stamp.xmax+1):
		for y in range(stamp.ymin,stamp.xmax+1):
			moments[0] += stamp.at(x,y)*(x-centx)*(x-centx)		# q11
			moments[1] += stamp.at(x,y)*(y-centy)*(y-centy)		# q22
			moments[2] += stamp.at(x,y)*(x-centx)*(y-centy)		# q12
			moments[3] += stamp.at(x,y)
			moments[4] += stamp.at(x,y)*(x-centx)
			moments[5] += stamp.at(x,y)*(y-centy)
	moments[0] /= moments[3]
	moments[1] /= moments[3]
	moments[2] /= moments[3]
	moments[4] /= moments[3]
	moments[5] /= moments[3]
	
	return moments

def readmeanmatrices(filename="fct_model.tab", dmax=8, factor=1):
# read shift coefficients for all chips from file and return a^L, a^R, a^B, a^T matrices
	f=np.genfromtxt(filename)
	d=np.zeros(len(f[0]))
	for i in range(len(f[0])):
         d[i] = np.average(f[:,i])

        nmax = int(d[2])
        if(nmax<dmax):
         print "input file with matrices does not go out far enough; I am devastated!"
         exit(0)

        nmat = 2*nmax+1

	# give me empty matrices

	aL = np.zeros([nmat,nmat])
	aR = np.zeros([nmat,nmat])
	aB = np.zeros([nmat,nmat])
	aT = np.zeros([nmat,nmat])

	# fill aR, aT on their positive side

	ioffset=4 # d index of first matrix element
	dx=1	  # lag of first matrix element
	dy=0
	# aR has nmax-1 rows and nmax columns

        while(dx<=nmax and dy<=nmax):
          aR[nmax+dy,nmax+dx]=d[ioffset]
          ioffset = ioffset+1
          dy = dy+1
          if(dy>nmax):
            dx=dx+1
            dy=0

	ioffset=ioffset+3
	dx=0
	dy=1

	while(dx<=nmax and dy<=nmax):
          aT[nmax+dy,nmax+dx]=d[ioffset]
          ioffset = ioffset+1
          dy = dy+1
          if(dy>nmax):
            dx=dx+1
            dy=1


	for dx in range(-nmax,nmax+1):
          for dy in range(-nmax,nmax+1):
            aR[dy+nmax,dx+nmax] = get_aR(dx,dy,aR,aT,nmax)
            aT[dy+nmax,dx+nmax] = get_aT(dx,dy,aR,aT,nmax)
            aL[dy+nmax,dx+nmax] = get_aL(dx,dy,aR,aT,nmax)
            aB[dy+nmax,dx+nmax] = get_aB(dx,dy,aR,aT,nmax)

	return (factor*aL,factor*aR,factor*aB,factor*aT) # multiply by hypothetical gain of 4

def get_aR(dx, dy, aR, aT, nmax):
  if(dx<=0): 
    return -get_aR(1-dx,dy,aR,aT,nmax)
  if(dy<0):
    return get_aR(dx,-dy,aR,aT,nmax)

  if(dx>nmax or dy>nmax):
    return 0

  return aR[nmax+dy,nmax+dx]

def get_aT(dx, dy, aR, aT, nmax):
  if(dy<=0): 
    return -get_aT(dx,1-dy,aR,aT,nmax)
  if(dx<0):
    return get_aT(-dx,dy,aR,aT,nmax)

  if(dx>nmax or dy>nmax):
    return 0

  return aT[nmax+dy,nmax+dx]

def get_aL(dx, dy, aR, aT, nmax):
  return -get_aR(dx+1,dy,aR,aT,nmax)

def get_aB(dx, dy, aR, aT, nmax):
  return -get_aT(dx,dy+1,aR,aT,nmax)



try:
    import galsim
    from galsim.cdmodel import *
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
    from galsim.cdmodel import *
    
(aL,aR,aB,aT) = readmeanmatrices()

if(len(sys.argv)!=5):
  print "syntax: sim1.py g1gal g2gal g1psf g2psf"
  sys.exit (1)
    
gal_flux=1000
gal_e1=float(sys.argv[1])
gal_e2=float(sys.argv[2])
gal_sigma=0.5/0.27 / 2.35 # Gaussian sigma such that FWHM~0.5''

psf_sigma=0.9/0.27 / 2.35 # Gaussian sigma such that FWHM~0.9''
psf_e1=float(sys.argv[3])
psf_e2=float(sys.argv[4])

psf_fluxmax=15000
    
# Define the true galaxy profile
gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
gal = gal.shear(galsim.Shear(g1=gal_e1, g2=gal_e2))

# Define the PSF profile
psf = galsim.Gaussian(flux=1., sigma=psf_sigma) # PSF flux should always = 1
psf = psf.shear(galsim.Shear(g1=psf_e1, g2=psf_e2))

galobserved = galsim.Convolve([gal, psf])
galobserved = galobserved.shift(0.5,0.5) # shift to be centered on a pixel

galimage_nocd = galobserved.drawImage(scale=1)

psf.setFlux(psf.getFlux()*psf_fluxmax/psf.xValue(0,0))
psfobserved = psf.shift(0.5,0.5) # shift to be centered on a pixel
psfimage_nocd = psfobserved.drawImage(scale=1)

cd = BaseCDModel(aL, aR, aB, aT)

galimage_cd = cd.applyForward(galimage_nocd)
psfimage_cd = cd.applyForward(psfimage_nocd)

sm_psfimage_nocd=secondmoments(psfimage_nocd)
sm_psfimage_cd  =secondmoments(psfimage_cd)

sm_galimage_nocd=secondmoments(galimage_nocd)
sm_galimage_cd  =secondmoments(galimage_cd)

# estimated pre-seeing moments
gal_q11_nocd = (sm_galimage_nocd[0]-sm_psfimage_nocd[0])
gal_q22_nocd = (sm_galimage_nocd[1]-sm_psfimage_nocd[1])
gal_q12_nocd = (sm_galimage_nocd[2]-sm_psfimage_nocd[2])
gal_denom_nocd = gal_q11_nocd + gal_q22_nocd + 2.*math.sqrt(gal_q11_nocd*gal_q22_nocd-gal_q12_nocd*gal_q12_nocd)

gal_q11_cd = (sm_galimage_cd[0]-sm_psfimage_cd[0])
gal_q22_cd = (sm_galimage_cd[1]-sm_psfimage_cd[1])
gal_q12_cd = (sm_galimage_cd[2]-sm_psfimage_cd[2])
gal_denom_cd = gal_q11_cd + gal_q22_cd + 2.*math.sqrt(gal_q11_cd*gal_q22_cd-gal_q12_cd*gal_q12_cd)

print "# truthe1 truthe2 obse1 obse2"
print (gal_q11_nocd - gal_q22_nocd) / gal_denom_nocd,2.*gal_q12_nocd/gal_denom_nocd,(gal_q11_cd - gal_q22_cd) / gal_denom_cd,2.*gal_q12_cd/gal_denom_cd
