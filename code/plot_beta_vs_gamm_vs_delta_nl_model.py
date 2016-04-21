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
pp=PdfPages("nlmodel_params.pdf")
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


##### Beta + gamma + delta, Y FILTER
x_vec, e1_inter_vec_all, e2_inter_vec_all, size_inter_vec_all =[18, 19, 20, 21, 22], [0.064210713841021061, -0.0010889200493693369, -0.00014413334429264069, -2.534128725528717e-05, -5.8161094784736633e-06] ,[0.052525006234645844, 0.01688631623983386, 0.0018866807222366333, 0.00032569468021392822, 7.5653195381164551e-05] ,[2.5526699059453413, 0.10859529476811192, 0.012273416883259669, 0.0020890731352281033, 0.0004763645110001935]
### Only beta        , Y FILTER
x_vec_b, e1_inter_vec_b, e2_inter_vec_b, size_inter_vec_b = [18, 19, 20, 21, 22], np.abs(np.array([-0.00014127884060144424, -5.5456534028053284e-05, -2.1913088858127594e-05, -8.6994841694814501e-06, -3.4617260098474684e-06] )) ,[0.0017975568771362305, 0.00069883465766906738, 0.00027564913034439087, 0.0001093372702598433, 4.3466687202467491e-05] ,   [0.010945443674033051, 0.0042734834154574131, 0.0016882306208148012, 0.00067003410431931698, 0.00026642577715607629]
                                                                                                                                                            #### Only gamma, Y FILTER

x_vec_g, e1_inter_vec_g, e2_inter_vec_g, size_inter_vec_g =  [18, 19, 20, 21, 22] ,[0.0047559230588376522, -0.00061251968145370483, -9.6344389021394902e-05, -1.5014782547950745e-05, -2.3161992430686951e-06] ,[0.072810575366020175, 0.0087003707885742326, 0.0012448355555534224, 0.00019387900829315186, 3.0651688575744629e-05], [0.64970709443326502, 0.055500959464566346, 0.0080924069299339596, 0.0012662052569676341, 0.00020024229512949354]

### Only delta, Y FILTER
x_vec_d, e1_inter_vec_d, e2_inter_vec_d, size_inter_vec_d = x_vec, e1_inter_vec, e2_inter_vec, size_inter_vec =[18, 19, 20, 21, 22], [0.0039549041539430618, 0.00036302581429481333, 2.4141743779182434e-05, 1.5543773770349684e-06, 1.0337680578405284e-07] ,np.abs(np.array([-0.034576743841171258, -0.0050085186958312711, -0.00033611059188842773, -2.131611108781295e-05, -1.3485550880432129e-06])) ,np.abs(np.array([-0.30715563177965199, -0.035219667135785837, -0.0023301511708837541, -0.00014746476122584617, -9.306024381761091e-06] ))


fig=plt.figure()

ax = fig.add_subplot (111)
ax.errorbar( x_vec, np.abs(np.array( size_inter_vec_all )) , yerr= None, ecolor = 'r', label='All', fmt='r-', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array( size_inter_vec_b )) , yerr= None, ecolor = 'b', label=r'Only $\beta$', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array( size_inter_vec_g )) , yerr= None, ecolor = 'g', label=r'Only $\gamma$', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec, np.abs(np.array( size_inter_vec_d )) 
            , yerr= None, ecolor = 'y', label='Only $\delta$', fmt='y-.x', markersize=marker_size, alpha=alpha)

plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label="mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta R/R$"
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
ax.set_yscale('log')
pp.savefig()
pp.close()


"""

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
"""

