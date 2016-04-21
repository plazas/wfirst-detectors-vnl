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
x_to_fit=np.array(mag_gal_vec) - 20

#f
x_vec, e1_inter_vec_f, e2_inter_vec_f, size_inter_vec_f = mag_gal_vec,  [3869.3743443753633, 2029.8530467241653, 807.91379437279306, 321.60147592455075, 128.0109840970016] , [5800.0605347991941, 3034.13436794681, 1205.4187608311115, 479.45330736941571, 190.80151784750458] , [5774.918210331628, 3006.0700791221775, 1190.3780678069836, 472.85569387464221, 188.07380754253032]
e1_err_f, e2_err_f, size_err_f = 0., 0., 0.
B, logA = get_slope2 (x_to_fit,  np.log10(np.array(size_inter_vec_f) ) )
print "lambda: F184, B: %g, logA: %g, A:% g" % ( B, logA, 10**(logA))


# h
e1_inter_vec_h, e2_inter_vec_h, size_inter_vec_h = [7539.7555864112956, 3957.1737993625366, 1575.4274708250584, 627.19376643953399, 249.69091084555274] , [12155.002819481904, 6339.2600192588707, 2513.4122264252555, 998.99552995297176, 397.45329860394867] , [13269.147042068615, 6844.6759916126193, 2694.4627428926474, 1067.9538467908421, 424.40689887131305]
e1_err_h, e2_err_h, size_err_h = 0.,0.,0.
B, logA = get_slope2 (x_to_fit,  np.log10(np.array(size_inter_vec_h) )  )
print "lambda: H158, B: %g, logA: %g, A:% g" % ( B, logA, 10**(logA))

# Y
e1_inter_vec_y, e2_inter_vec_y, size_inter_vec_y=  [9231.0404009457779, 4869.8473893781911, 1945.5367177256292, 775.62075676703512, 308.95483931299447],  [18497.060454073588, 9701.9443937235428, 3860.5236202364113, 1536.5620240446606, 611.68383127287041] , [24844.894665144788, 12779.439481581463, 5020.1607652373059, 1987.9663606399508, 789.7873718160929]
e1_err_y, e2_err_y, size_err_y = 0.,0.,0.
B, logA = get_slope2 (x_to_fit,  np.log10(np.array(size_inter_vec_y) )  )
print "lambda: Y106, B: %g, logA: %g, A:% g" % ( B, logA, 10**(logA))

# J

e1_inter_vec_j, e2_inter_vec_j, size_inter_vec_j= [9171.6254795867007, 4826.5380487426119, 1924.8462727404608, 766.8411154118578, 305.36006544469484], [16749.839693774658, 8739.8037869256641, 3466.0927752781104, 1377.6726832323261, 548.11875572057659], [20777.534916585388, 10658.259034890016, 4180.2543420429047, 1654.4000160289575, 657.08504386882328]
e1_err_j, e2_err_j, size_err_j = 0., 0., 0.
B, logA = get_slope2 (x_to_fit,  np.log10(np.array(size_inter_vec_j) )  )
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
y1label=r"$\sigma_{\Delta R/R}/\sigma_{\beta}$"
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
plt.ylim([50,5e4])
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\sigma_{\Delta e_1}/\sigma_{\beta}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.04*delta, xmax + 0.04*delta])


ax = fig.add_subplot (224)
ax.errorbar( x_vec, (np.array(e2_inter_vec_f)), yerr= e2_err_f, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec,(np.array( e2_inter_vec_j)), yerr= e2_err_j, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array(e2_inter_vec_y)), yerr= e2_err_y, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array(e2_inter_vec_h)), yerr= e2_err_h, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\sigma_{\Delta e_2}/\sigma_{\beta}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
ax.set_yscale('log')
plt.ylim([50,5e4])
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.04*delta, xmax + 0.04*delta])

fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig()
pp.close()




