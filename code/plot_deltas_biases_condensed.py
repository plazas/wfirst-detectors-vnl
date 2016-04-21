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


### DATA
pp=PdfPages("out.pdf")
print "Output PDF: out.pdf"
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
prop = fm.FontProperties(size=7)
marker_size=7.0
alpha=0.7
loc_label = "upper right"
visible_x, visible_y = True, True


def get_slope2 (x, y):
    p=np.polyfit(x,y,1)
    print 'pendiente and intercept from np.polyfit', p[0] , p[1]
    return p[0], p[1]

mag_gal_vec=[18.3, 19, 20, 21,22]

x_to_fit= np.array(mag_gal_vec) - 20

#f
x_vec, e1_inter_vec_f, e2_inter_vec_f, size_inter_vec_f = mag_gal_vec, [-108.29139500855059, -55.946409702317347, -22.223219275476843, -8.7404623627699412, -3.4719705581708729] , [1323.8191604613526, 680.70739507692065, 267.39388704305117, 105.81687092786153, 42.03990101824769] , [10795.163275703149, 5576.2713818977873, 2197.2339003826874, 871.10019975635669, 346.20499960059698]
e1_err_f, e2_err_f, size_err_f = 0., 0., 0.
B, logA = get_slope2 (x_to_fit,  np.log10(np.array(size_inter_vec_f) ) )
print "lambda: F184, B: %g, logA: %g, A:% g" % ( B, logA, 10**(logA))


# h
e1_inter_vec_h, e2_inter_vec_h, size_inter_vec_h = [-297.39458113910734, -152.78998762368852, -59.653073549256455, -23.757107555868448, -9.387731552153765], [3846.8986749651299, 1943.5659050943575, 755.66023588173732, 297.53893613815922, 117.95014142999977] , [23265.655850308336, 11830.500038463049, 4613.512965506111, 1821.8338967573131, 723.20214426270547]
e1_err_h, e2_err_h, size_err_h = 0.,0.,0.
B, logA = get_slope2 (x_to_fit,   np.log10(np.array(size_inter_vec_h) )   )
print "lambda: H158, B: %g, logA: %g, A:% g" % ( B, logA, 10**(logA))

# Y
e1_inter_vec_y, e2_inter_vec_y, size_inter_vec_y= [320.14586031436227, 130.48574328426642, 42.919069528560335, 15.528872609135394, 6.1057507991792548] , [8216.9920206072711, 4248.7084865566467, 1671.2695360184875, 662.23740577701733, 262.64786720251431] ,  [41879.488542040635, 21144.155995791851, 8203.0514058540375, 3232.8943927278601, 1282.0072485679859]
e1_err_y, e2_err_y, size_err_y = 0.,0.,0.
B, logA = get_slope2 (x_to_fit,   np.log10(np.array(size_inter_vec_y) ) )
print "lambda: Y106, B: %g, logA: %g, A:% g" % ( B, logA, 10**(logA))

# J

e1_inter_vec_j, e2_inter_vec_j, size_inter_vec_j= [-382.8527405858228, -205.44417202472343, -81.76174014804981, -32.790005207079794, -13.42035830020936] , [7475.3463268283549, 3776.4906883238159, 1464.3669128416539, 577.56900787357415, 229.61199283605788] ,  [35085.406685903297, 17681.255684383254, 6853.0660160146017, 2699.6364119794266, 1071.2169132669835]
e1_err_j, e2_err_j, size_err_j = 0., 0., 0.
B, logA = get_slope2 (x_to_fit,   np.log10(np.array(size_inter_vec_j) )  )
print "lambda: J129, B: %g, logA: %g, A:% g" % ( B, logA, 10**(logA))









fig=plt.figure()

ax = fig.add_subplot (211)
ax.errorbar( x_vec, size_inter_vec_f, yerr= size_err_f, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, size_inter_vec_j, yerr= size_err_j, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, size_inter_vec_y, yerr= size_err_y, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, size_inter_vec_h, yerr= size_err_h, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yscale('log')
plt.ylim([1e2,1e5])
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta R/R/\beta$"
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (223)
ax.errorbar( x_vec, np.abs(np.array(e1_inter_vec_f)), yerr= e1_err_f, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array(e1_inter_vec_j)), yerr= e1_err_j, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array( e1_inter_vec_y)), yerr= e1_err_y, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array( e1_inter_vec_h)), yerr= e1_err_h, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yscale('log')
plt.ylim([1,1e3])
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$|\Delta e_1/\beta|$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.04*delta, xmax + 0.04*delta])


ax = fig.add_subplot (224)
ax.errorbar( x_vec, np.abs(np.array(e2_inter_vec_f)), yerr= e2_err_f, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec,np.abs(np.array( e2_inter_vec_j)), yerr= e2_err_j, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array(e2_inter_vec_y)), yerr= e2_err_y, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array(e2_inter_vec_h)), yerr= e2_err_h, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$|\Delta e_2/\beta|$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
ax.set_yscale('log')
plt.ylim([10,5e4])
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.04*delta, xmax + 0.04*delta])

fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig()
pp.close()




