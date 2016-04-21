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
mpl.rc('font', family='serif',weight='normal', size=6.0 )
mpl.rc('text',  color='black', usetex=False)
mpl.rc('axes',  edgecolor='black', linewidth=1, grid=False, titlesize=10, labelsize=9, labelweight='normal',labelcolor='black')
mpl.rc('axes.formatter', limits=[-4,4])
mpl.rcParams['xtick.major.size']=6
mpl.rcParams['xtick.minor.size']=3
mpl.rcParams['xtick.major.pad']=7
mpl.rcParams['xtick.minor.pad']=7
mpl.rcParams['xtick.labelsize']= '8'
mpl.rcParams['xtick.minor.width']= 1.0
mpl.rcParams['xtick.major.width']= 1.0
mpl.rcParams['ytick.major.size']=6
mpl.rcParams['ytick.minor.size']=3
mpl.rcParams['ytick.major.pad']=7
mpl.rcParams['ytick.minor.pad']=7
mpl.rcParams['ytick.labelsize']= '8'
mpl.rcParams['ytick.minor.width']= 1.0
mpl.rcParams['ytick.major.width']= 1.0
mpl.rc ('legend', numpoints=1, fontsize='8', shadow=False, frameon=False)

## Plot parameters
plt.subplots_adjust(hspace=0.01, wspace=0.01)
prop = fm.FontProperties(size=7)
marker_size=5.0
alpha=0.6
loc_label = "upper right"
visible_x, visible_y = True, True


beta0=0.5e-6  # midpoint from ramge in hilbert 2014
gamma0=0.25e-10
delta0=1.5e-15



#1. Data vectors

# 1.1 Figure 3
# 1.1.1 beta
# 1.1.1.1 J filter


x_vec=[18.2, 19, 20, 21, 22]
x_vec_b=x_vec
x_vec_g=x_vec
x_vec_d=x_vec


e1_inter_vec_j_b, e2_inter_vec_j_b, size_inter_vec_j_b = np.abs(np.array([420.36176684320918, 214.83749151240497, 104.68071586249808, 32.03749656668284, 12.144447419153382])) , np.abs(np.array([-6920.099271509047, -3458.8576090366896, -1438.2600914953925, -565.648052086706, -227.09369657995424])), np.abs(np.array( [-32305.734825478667, -16183.327919449261, -6615.6040552958184, -2663.4685724945775, -1072.0500463743549] ))
# 1.1.1.2 Y filter
e1_inter_vec_y_b, e2_inter_vec_y_b, size_inter_vec_y_b = np.abs(np.array([-60.722224120165414, -58.263576431388422, -33.304127654195618, -8.8661909108908254, -5.8859586717083987] )), np.abs(np.array([-8286.8335747308702, -4083.5142100770067, -1628.3988952615184, -673.53243015799137, -259.87624844275712] )) , np.abs(np.array([-38720.709293881584, -19409.466250773035, -7931.2993517632067, -3193.8028365986788, -1274.7249253318828]))
# 1.1.1.3 H filter
e1_inter_vec_h_b, e2_inter_vec_h_b, size_inter_vec_h_b = np.abs(np.array([297.98597097421379, 140.70421320820381, 61.057537633461422, 27.19462738268308, 9.350478649332052] )), np.abs(np.array([-3604.8889160162207, -1762.2113781562746, -734.62724685865942, -294.14892196488017, -118.31521987915441])), np.abs(np.array([-22251.317310545452, -11052.638630183485, -4490.3385845813873, -1809.1414498845465, -723.65377078217341]))
# 1.1.1.4 F filter
e1_inter_vec_f_b, e2_inter_vec_f_b, size_inter_vec_f_b= np.abs(np.array([114.60857284532469, 33.155083656397693, 18.36568117080818, 6.6868960859793205, 3.5390257833308731])), np.abs(np.array( [-1320.0938222550753, -657.886266705342, -267.47584342862058, -100.58284503209985, -42.766332364497842])), np.abs(np.array([-11058.54719048536, -5390.5906775633794, -2163.397407438611, -870.15242247633068, -334.66999149594744]))





# 1.1.2 gamma
# 1.1.2.1  J FILTER
e1_inter_vec_j_g, e2_inter_vec_j_g, size_inter_vec_j_g= np.abs(np.array([7265526.8013495831, 9416136.890651118, 1474004.2388446054, 277068.46594740968, 48521.906135731464])) , np.abs(np.array([-923770666.12244713, -183933973.31239012, -27664005.756380644, -4336237.9074013112, -706315.04058005009] )), np.abs(np.array([-4999202831.5444841, -907234968.60109305, -135589256.2897799, -21281278.044793427, -3353627.6758505367]))
# 1.1.2.2  Y FILTER
e1_inter_vec_y_g, e2_inter_vec_y_g, size_inter_vec_y_g= np.abs(np.array([-161808729.17175123, -11908262.968063993, -805407.76252884872, -135973.0958931653, -24400.651455272251])),np.abs(np.array( [-802141427.99377704, -222690403.46145204, -35390257.835368603, -5623698.2345663868, -891089.43939485576])),np.abs(np.array( [-6313444754.0214987, -1191568261.1540039, -178538536.51810166, -28050874.12347516, -4440512.6115476163]))
# 1.1.2.2  H FILTER
e1_inter_vec_h_g, e2_inter_vec_h_g, size_inter_vec_h_g =np.abs(np.array( [25256443.768739872, 6612110.8829978295, 932533.29396385851, 120513.14115489514, 25052.577257329678] )), np.abs(np.array([-399628281.59332764, -76018273.83041878, -11393427.848824168, -1809746.0269900174, -286847.35297740233])), np.abs(np.array( [-2500946516.7233295, -490384051.0187062, -75020974.186789453, -11805012.917376271, -1874219.3201859344]))
# 1.1.2.2  F FILTER
e1_inter_vec_f_g, e2_inter_vec_f_g, size_inter_vec_f_g=np.abs(np.array([5067000.1655807206, 1076934.8591569478, 218115.74697425004, 62212.347984226959, 4516.9144869717902])) , np.abs(np.array([-65040960.907927334, -14035776.257512109, -2196058.6309384201, -365450.97827980557, -54389.238358931543])), np.abs(np.array([-566388463.10390401, -124551212.98349115, -19566254.157332681, -3113632.6240188498, -478099.66332579067]))



# 1.1.3 delta
# 1.1.3.1  J FILTER
e1_inter_vec_j_d, e2_inter_vec_j_d, size_inter_vec_j_d=np.abs(np.array([-1952429302036.1848, 557992607355.20398, 43679028749.503281, 689178705.33683872, 279396772.36730582] )),np.abs(np.array([-10695680975914.994, -6405860185623.6318, -447183847427.64935, -29057264328.280769, -1937150955.2002261])) ,np.abs(np.array([-86305516549169.812, -32920054266571.906, -2316989330196.688, -147891738375.71017, -8331871398.3705854]))
# 1.1.3.2  Y FILTER
e1_inter_vec_y_d, e2_inter_vec_y_d, size_inter_vec_y_d=np.abs(np.array([370139885926633.25, -2697110176086.6221, 241510570049.26147, -27976930141.276432, 167638063.53484565])), np.abs(np.array( [-2862289547918.605, -10135024785995.768, -646710395812.73682, -40680170061.148254, -2235174179.0771551])), np.abs(np.array([-88020745436203.656, -45430883524787.867, -3499856310970.1724, -208715675124.04361, -14008003611.510523]))
# 1.1.3.3  H FILTER
e1_inter_vec_h_d, e2_inter_vec_h_d, size_inter_vec_h_d= np.abs(np.array( [-2065007574857.0254, 261869281530.40262, 4675239324.5179682, -4582107066.9870005, -204890966.38072169])) , np.abs(np.array([-9445212781428.6406, -2244263887405.7153, -155046582221.57245, -22947788238.664734, -372529030.12376648])), np.abs(np.array([-90322418533892.172, -16042473964387.455, -1048404631361.9247, -72886298098.098602, -5206173587.7909451]))
# 1.1.3.4  F FILTER
e1_inter_vec_f_d, e2_inter_vec_f_d, size_inter_vec_f_d=np.abs(np.array([195833854377.24411, 26565976440.889374, 1760199665.9972961, 88475644.675209135, 18626451.535678566])), np.abs(np.array([-1981891691684.5752, -242553651332.93283, -14640390872.747494, -931322575.51755822, -74505806.10801889])), np.abs(np.array([-19687509669609.867, -2358824613740.7739, -152393103020.07159, -8964496148.7337151, -1195252433.6944551]))





fig=plt.figure()

ax = fig.add_subplot (331)
ax.errorbar( x_vec_b, beta0*size_inter_vec_f_b, yerr=0., ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*size_inter_vec_j_b, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*size_inter_vec_y_b, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*size_inter_vec_h_b, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\beta$ ($\times$10$^{6}$)"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta R/R/(\Delta\beta)/\beta_0$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
ax.set_yscale('log')
#plt.ylim([ np.min(size_inter_vec_y_b) - 0.5*delta , np.max (size_inter_vec_y_b) + 0.5*delta])
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])

ax = fig.add_subplot (332)
ax.errorbar( x_vec_g, gamma0*size_inter_vec_f_g, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*size_inter_vec_j_g, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*size_inter_vec_y_g, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*size_inter_vec_h_g, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\gamma$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
ax.set_yscale('log')
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta R/R/(\Delta\gamma)/\gamma_0$"
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])

ax = fig.add_subplot (333)
ax.errorbar( x_vec_d, delta0*size_inter_vec_f_d, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*size_inter_vec_j_d, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*size_inter_vec_y_d, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*size_inter_vec_h_d, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\delta$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#plt.ylim([8e8,2e15])
ax.set_yscale('log')
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
y1label=r"$\Delta R/R/(\Delta\delta)/\delta_0$"
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])



ax = fig.add_subplot (334)
ax.errorbar( x_vec_b, beta0*e1_inter_vec_f_b, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*e1_inter_vec_j_b, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*e1_inter_vec_y_b, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*e1_inter_vec_h_b, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta$$e_1/(\Delta\beta)/\beta_0$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
plt.ylim ([1e-7, 1e-3])
ax.set_yscale('log')
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])

ax = fig.add_subplot (335)
ax.errorbar( x_vec_g, gamma0*e1_inter_vec_f_g, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*e1_inter_vec_j_g, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*e1_inter_vec_y_g, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*e1_inter_vec_h_g, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
#x1label=r"$\gamma$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
y1label=r"$\Delta$e$_1/(\Delta\gamma)/\gamma_0$"
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
ax.set_yscale('log')
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])

ax = fig.add_subplot (336)
ax.errorbar( x_vec_d, delta0*e1_inter_vec_f_d, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*e1_inter_vec_j_d, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*e1_inter_vec_y_d, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*e1_inter_vec_h_d, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
#x1label=r"$\delta$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
ax.set_yscale('log')
y1label=r"$\Delta$e$_1/(\Delta\delta)/\delta_0$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])




ax = fig.add_subplot (337)
ax.errorbar( x_vec_b, beta0*e2_inter_vec_f_b, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*e2_inter_vec_j_b, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*e2_inter_vec_y_b, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, beta0*e2_inter_vec_h_b, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
plt.ylim ([1e-6, 1e-2])
ax.set_yscale('log')
y1label=r"$\Delta$e$_2/(\Delta\beta)/\beta_0$"
#y1label=r"$\Delta$$e_2$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])

ax = fig.add_subplot (338)
ax.errorbar( x_vec_g, gamma0*e2_inter_vec_f_g, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*e2_inter_vec_j_g, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*e2_inter_vec_y_g, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, gamma0*e2_inter_vec_h_g, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
ax.set_yscale('log')
y1label=r"$\Delta$e$_2/ (\Delta\gamma)/\gamma_0$"
#plt.ylim([1e4,1e9])
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])

ax = fig.add_subplot (339)
ax.errorbar( x_vec_d, delta0*e2_inter_vec_f_d, yerr=0. , ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*e2_inter_vec_j_d, yerr=0. , ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*e2_inter_vec_y_d, yerr=0. , ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, delta0*e2_inter_vec_h_d, yerr=0. , ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
plt.ylim ([1e-8, 1e-1])
y1label=r"$\Delta$e$_2/(\Delta\delta)/\delta_0$"
ymin, ymax=plt.ylim()
delta=(ymax-ymin)
ax.set_yscale('log')
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=11)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.2*delta, xmax + 0.2*delta])


fig.tight_layout()
#fig.suptitle ("AB magnitude = 18.2", size=11)
plt.subplots_adjust(top=0.925)
pp.savefig()
pp.close()




