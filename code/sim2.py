#!/usr/bin/python
# sim2: simulate effect of B/F on shape measurement if it's not accounted for

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

	return (aL, aR,aB,aT) # multiply by hypothetical gain of 4

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


