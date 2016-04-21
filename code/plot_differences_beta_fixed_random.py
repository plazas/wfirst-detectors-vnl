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
pp=PdfPages("differences_beta.pdf")
print "Output PDF: differences_beta.pdf"
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


#f
x_vec, e1_inter_vec_f, e2_inter_vec_f, size_inter_vec_f = [  3.56600000e-15 , 2.49620000e-07 , 3.56600000e-07 ,7.13200000e-07, 1.06980000e-06 , 1.24810000e-06 , 1.60470000e-06], [0.0, 8.2397367805329276e-07, 1.1806981638074267e-06, 2.3676827549934563e-06, 3.5655964165924071e-06, 4.1620340198278425e-06, 5.3687626495957806e-06], [0.0, 7.0784240960394817e-07, 1.0156258940692553e-06, 2.0391121506690979e-06, 3.0566751956944554e-06, 3.5738199949270076e-06, 4.5949965715404854e-06], [0.0, 1.7928754357367759e-07, 2.6534556448897526e-07, 5.4025313130159038e-07, 7.5778868417068593e-07, 9.4544297977772028e-07, 1.2442555524003717e-06]

e1_err_f, e2_err_f, size_err_f = [0.0, 2.6224083062769433e-06, 3.7472672551296486e-06, 7.5010549361763659e-06, 1.1261221179569041e-05, 1.3143709277524262e-05, 1.6913210187122968e-05], [0.0, 2.7220887086048229e-06, 3.8902984198375446e-06, 7.7909144365836392e-06, 1.1701897686631045e-05, 1.3661162892244518e-05, 1.7587538058464506e-05], [0.0, 2.8192515074944341e-06, 4.0340149123846128e-06, 8.1115818337589009e-06, 1.2233608688537826e-05, 1.4311303994714718e-05, 1.8501044240766993e-05]

# h
e1_inter_vec_h, e2_inter_vec_h, size_inter_vec_h = [0.0, 8.2090497016898061e-07, 1.1827517300842977e-06, 2.3722182959318856e-06, 3.5973917692900052e-06, 4.2047817260026588e-06, 5.4432358592730084e-06] ,[0.0, 1.3448297977448897e-06, 1.9285827875138716e-06, 3.8506835699084194e-06, 5.7872384786600287e-06, 6.7557394504542952e-06, 8.7090581655505087e-06], [0.0, -5.3883742311801016e-07, -7.4903607972085728e-07, -1.5520860557205041e-06, -2.3596914205636575e-06, -2.7286779168277966e-06, -3.589646408115859e-06]

e1_err_h, e2_err_h, size_err_h = [0.0, 4.4768641088519975e-06, 6.3971961053053682e-06, 1.280467444232737e-05, 1.9221753878661585e-05, 2.2433610077340785e-05, 2.8863273501592323e-05], [0.0, 4.9865382172056008e-06, 7.1275774683216313e-06, 1.4282005027358702e-05, 2.1462933181136462e-05, 2.5063173694445085e-05, 3.2283088887311515e-05], [0.0, 5.4801958823207398e-06, 7.850977666535866e-06, 1.5850822261137003e-05, 2.4003873868203209e-05, 2.813972454387645e-05, 3.6532084348128527e-05]

# Y
e1_inter_vec_y, e2_inter_vec_y, size_inter_vec_y=  [0.0, 1.830607652661062e-07, 2.6661902665724156e-07, 5.3007155656432931e-07, 8.0110505223301988e-07, 9.3445181846618654e-07, 1.2074783444405644e-06], [0.0, 7.4371695518604672e-07, 1.0466575623122033e-06, 2.1141767502155794e-06, 3.1729042530051489e-06, 3.7060678005201854e-06, 4.7667324542996496e-06], [0.0, -8.0685631759048551e-07, -1.1843698203519967e-06, -2.3400234002339799e-06, -3.5426035194200626e-06, -4.1512477381581109e-06, -5.3594310262174236e-06]

e1_err_y, e2_err_y, size_err_y = [0.0, 2.0456902778175367e-06, 2.9217348690719771e-06, 5.8387094221367765e-06, 8.75063631005264e-06, 1.0204668300764022e-05, 1.3108884152535996e-05] , [0.0, 2.8213766610818379e-06, 4.0301813076236204e-06, 8.0579492908505552e-06, 1.2082864100091543e-05, 1.409419119802001e-05, 1.8114492362935758e-05] , [0.0, 3.7845237963647166e-06, 5.4132348027292402e-06, 1.0870500516312001e-05, 1.6371892585382233e-05, 1.9139167758893932e-05, 2.4706745853562787e-05]


# J

e1_inter_vec_j, e2_inter_vec_j, size_inter_vec_j= [0.0, 4.9970112740648287e-07, 7.1251764893351387e-07, 1.451885327696835e-06, 2.2090226411836111e-06, 2.6008859276773107e-06, 3.3828336745517911e-06], [0.0, 1.8149614334389553e-06, 2.6029348373690643e-06, 5.2082538604744652e-06, 7.8171491622941461e-06, 9.1144442558019349e-06, 1.1745840311023769e-05] , [0.0, -1.7212480749655232e-06, -2.4426586235291435e-06, -4.9859953313133746e-06, -7.4994757796296304e-06, -8.8270378285756723e-06, -1.1523127491780936e-05]

e1_err_j, e2_err_j, size_err_j = [0.0, 5.4661188377220419e-06, 7.8071574364461447e-06, 1.5602849276717925e-05, 2.3384352853389511e-05, 2.7268940104445324e-05, 3.5024041297423918e-05] , [0.0, 6.8505244312056665e-06, 9.7900022015114505e-06, 1.9603567063548297e-05, 2.9438486530272767e-05, 3.436317112112581e-05, 4.4225025149984028e-05] , [0.0, 8.4507400884607647e-06, 1.2116325801001284e-05, 2.4529684585997115e-05, 3.7248875351746112e-05, 4.3726342150102651e-05, 5.6922577214205918e-05]



fig=plt.figure()

ax = fig.add_subplot (211)
ax.errorbar( x_vec, size_inter_vec_f, yerr= size_err_f, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, size_inter_vec_j, yerr= size_err_j, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, size_inter_vec_y, yerr= size_err_y, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, size_inter_vec_h, yerr= size_err_h, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"$\beta$"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$d_{\Delta R/R}$"
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (223)
ax.errorbar( x_vec, e1_inter_vec_f, yerr= e1_err_f, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, e1_inter_vec_j, yerr= e1_err_j, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, e1_inter_vec_y, yerr= e1_err_y, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, e1_inter_vec_h, yerr= e1_err_h, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"$\beta$"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$d_{\Delta e_1}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.04*delta, xmax + 0.04*delta])


ax = fig.add_subplot (224)
ax.errorbar( x_vec, e2_inter_vec_f, yerr= e2_err_f, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, e2_inter_vec_j, yerr= e2_err_j, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, e2_inter_vec_y, yerr= e2_err_y, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, e2_inter_vec_h, yerr= e2_err_h, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"$\beta$"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$d_{\Delta e_2}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)

xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.04*delta, xmax + 0.04*delta])

fig.tight_layout()
plt.subplots_adjust(top=0.85)
pp.savefig()
pp.close()




