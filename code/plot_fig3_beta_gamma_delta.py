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
print "Output PDF: differences_beta.pdf"
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

#1. Data vectors

# 1.1 Figure 3
# 1.1.1 beta
# 1.1.1.1 J filter
x_vec_b, e1_inter_vec_j_b, e2_inter_vec_j_b, size_inter_vec_j_b = np.array( [ -5.00000000e-07  , -1.00000000e-07   ,3.00000000e-07 ,   7.00000000e-07 , 1.10000000e-06 ,  1.50000000e-06])*10**6 , [-0.00021272294223308535, -4.2989179491996906e-05, 0.00012941453605890308, 0.00030160140246152843, 0.00047147337347269096, 0.00063724443316459656] , [0.0039529228210449189, 0.00076279819011688672, -0.0022085237503051813, -0.0049733328819274959, -0.0075457829236984285, -0.0099390122294425957] , [0.018515934777148139, 0.0035670493838327834, -0.010318889836495551, -0.023240594886296963, -0.035286957522515018, -0.046537422407278213]
e1_inter_vec_err_j_b, e2_inter_vec_err_j_b, size_inter_vec_err_j_b = [5.6006621431844732e-07, 1.3435483477906667e-07, 3.6390089796752115e-07, 6.0888137626608492e-07, 8.082446665369096e-07, 1.0972429786985725e-06] , [5.3661860095955111e-07, 2.0776814680166717e-07, 4.7715593650159766e-07, 3.3201231744844207e-07, 6.5639464524827132e-07, 6.7711925834496319e-07] , [3.8007295715207425e-07, 1.6381264835417145e-07, 3.137918167614637e-07, 4.4029093021237828e-07, 5.566581295667138e-07, 5.6788705048260129e-07]
# 1.1.1.2 Y filter
e1_inter_vec_y_b, e2_inter_vec_y_b, size_inter_vec_y_b =[0.00013941109180450398,2.1184459328651706e-05, -4.6083182096480492e-05, -7.2070211172103749e-05, -6.5093785524367864e-05, -3.2083243131637989e-05],  [0.0044386291503906231, 0.00087390065193175602, -0.0025755649805069068, -0.0058926290273666419, -0.0090662795305252106, -0.012090309262275691] , [0.022137503591776922, 0.0042706671375168171, -0.012364133669142912, -0.027856289829182191, -0.042291993892839025, -0.055753397721941822]
e1_inter_vec_err_y_b, e2_inter_vec_err_y_b, size_inter_vec_err_y_b= [2.6562823486115767e-07, 1.0675634287323235e-07, 1.7075043536866993e-07, 1.4013034873473323e-07, 2.4006452834227046e-07, 3.4564193369223766e-07], [4.8093377135092292e-07, 1.6901464316914643e-07, 3.2475726219992627e-07, 5.303956080391777e-07, 6.8661239332305241e-07, 6.1761917603329752e-07] ,[2.3665481608787172e-07, 1.3448119877902952e-07, 2.185310634801978e-07, 2.271802796873705e-07, 3.122537898754403e-07, 2.7044922931501044e-07]
# 1.1.1.3 H filter
e1_inter_vec_h_b, e2_inter_vec_h_b, size_inter_vec_h_b = [-0.00015961017459631015, -3.1192004680633751e-05, 9.1382451355457092e-05, 0.00020818136632442453, 0.00031944528222084025, 0.00042523015290498698] , [0.0020357933640480098, 0.00039295673370361492, -0.001139042675495146, -0.0025694057345390293, -0.0039068561792373613, -0.00515872001647949] , [0.012382469852867227, 0.0024055781624980985, -0.0070157112606671259, -0.015925473646690381, -0.024363390735585987, -0.032365105704480229]
e1_inter_vec_err_h_b, e2_inter_vec_err_h_b, size_inter_vec_err_h_b= [2.6122671128765842e-07, 1.2120187187405426e-07, 2.6885793023378638e-07, 3.1455344941620805e-07, 3.0492681921083033e-07, 3.8160676279961572e-07] , [4.5043467070731279e-07, 1.9481557258996284e-07, 3.3886642834184077e-07, 3.1989291431167554e-07, 4.7392196178491014e-07, 6.0184356744414275e-07] , [3.1240780841393544e-07, 1.0503790492783727e-07, 2.380689164656097e-07, 2.7232420149069819e-07, 2.5925396457442972e-07, 3.3035335577890411e-07]
# 1.1.1.4 F filter
e1_inter_vec_f_b, e2_inter_vec_f_b, size_inter_vec_f_b= [-5.8489367365837581e-05, -1.1510420590639392e-05, 3.4158229827880753e-05, 7.8469887375831531e-05, 0.0001214088313281533, 0.00016320319846272437] , [0.00071207612752914649, 0.00013976320624351586, -0.00041121497750282177, -0.00094178050756454381, -0.0014530345797538729, -0.001945931911468503] , [0.0058310365433445902, 0.0011493408530477466, -0.0033986635006534628, -0.0078185437776821191, -0.01211551373484423, -0.016294710579913937]
e1_inter_vec_err_f_b, e2_inter_vec_err_f_b, size_inter_vec_err_f_b= [  2.89032221e-07  , 1.18872817e-07,   1.97515417e-07 ,  2.80648244e-07 , 3.40135226e-07  , 4.37232670e-07], [  3.58661880e-07   ,1.88067767e-07,   2.62957410e-07 ,  4.01557293e-07, 5.21198867e-07 ,  4.32964412e-07], [  2.65324235e-07  , 1.25727543e-07  , 1.42965321e-07  , 2.53815391e-07, 2.89878749e-07, 2.94333629e-07]

# 1.1.2 gamma
# 1.1.2.1  J FILTER
x_vec_g, e1_inter_vec_j_g, e2_inter_vec_j_g, size_inter_vec_j_g=np.array([ -1.00000000e-10 , -7.00000000e-11 , -4.00000000e-11 , -1.00000000e-11, 2.00000000e-11 ,  5.00000000e-11])*10**10 , [0.011481633381918073, 0.0033262273296713828, -0.00040610827505588526, -0.00037327527999877862, 0.00086812142282724448, 0.0020744068175554274] , [0.064603211283683779, 0.057771367430686948, 0.035224276185035708, 0.0079800063371658261, -0.013502306938171395, -0.028227810263633728], [0.63590092784432684, 0.39348054322697407, 0.18965851081483315, 0.039291714314269276, -0.065751211847121321, -0.13997941341787576]
e1_inter_vec_err_j_g, e2_inter_vec_err_j_g, size_inter_vec_err_j_g =[7.3714206965115952e-06, 9.1150143699992667e-06, 7.4115719105790388e-06, 2.7285980342693877e-06, 4.5743368414924502e-06, 6.4807325502101948e-06] , [1.3456619496897275e-06, 1.2567443941645239e-06, 1.0774409669850231e-06, 5.9059270622313825e-07, 8.9654599417248142e-07, 1.5697495701747001e-06] , [7.3594185884100142e-06, 7.7237560271399624e-06, 5.7158215856266666e-06, 1.8295688304991362e-06, 2.292394844898379e-06, 3.7703911835462797e-06]
# 1.1.2.2  Y FILTER
e1_inter_vec_y_g, e2_inter_vec_y_g, size_inter_vec_y_g= [0.029430430121719836, 0.015600242365617306, 0.0053824764862656594, 0.00048976555466651917, 5.7445019483566287e-05, 0.0011861462146043778], [0.03894305944442749, 0.04122909605503082, 0.033214145302772516, 0.009709605574607854, -0.018756307363510135, -0.041090931296348575] ,[0.71855833250621304, 0.46793173927568349, 0.24034279126532979, 0.051592153317361271, -0.086100009975810382, -0.17956472065048149]
e1_inter_vec_err_y_g, e2_inter_vec_err_y_g, size_inter_vec_err_y_g= [3.3341561698316026e-07, 7.9100041071586485e-07, 3.1631065936403704e-06, 3.1628868759598266e-06, 2.1773072987088839e-05, 0.00012024753973774364], [4.0989018224854329e-07, 4.9947127496941632e-07, 7.9381751448644767e-07, 6.2691223525806744e-07, 2.0623513726568695e-06, 1.0814577064763792e-05], [4.6294256013828391e-07, 6.6149269693917225e-07, 2.494388700270322e-06, 2.1405133184919888e-06, 1.1779649769933861e-05, 5.5419206123844006e-05]
# 1.1.2.2  H FILTER
e1_inter_vec_h_g, e2_inter_vec_h_g, size_inter_vec_h_g = [-0.0013250709325075155, -0.0015975294634699821, -0.0010119108483195307, -0.00024654835462570177, 0.00045879449695348699, 0.0010460473224520677] ,[0.049377147853374478, 0.030987952947616578, 0.015325825214385986, 0.0032953056693077122, -0.0056969785690307575, -0.012420115172863002], [0.32305503297771798, 0.19439055155471943, 0.096630367076765603, 0.0213076636704265, -0.03804532605483691, -0.085761477629959823]
e1_inter_vec_err_h_g, e2_inter_vec_err_h_g, size_inter_vec_err_h_g= [1.2002251447221837e-05, 1.4030060485332542e-05, 1.237625936789185e-05, 4.01440574086137e-06, 1.0236447690539293e-05, 3.1873263718234063e-05], [7.9339133447248589e-07, 8.8263662079376579e-07, 8.1927955638169263e-07, 6.0129861662922332e-07, 1.0305531904094347e-06, 1.7830145433196965e-06] ,[4.9009354911582868e-06, 6.9470089262710829e-06, 5.8734886656562968e-06, 1.91091157807731e-06, 4.4642127776788779e-06, 1.3483479824628367e-05]
# 1.1.2.2  F FILTER
e1_inter_vec_f_g, e2_inter_vec_f_g, size_inter_vec_f_g=[-0.0005272696912288667, -0.00036087999120354685, -0.00020111316815018662, -4.8985853791237084e-05, 9.5384269952773914e-05, 0.00023164574056863752] ,[0.0070245645940303801, 0.0046900698542594928, 0.0025589682161808037, 0.00061157360672950824, -0.0011700893938541396, -0.0028019309043884268] , [0.059972965339097281, 0.040571763098873746, 0.022429879406526253, 0.0054305407860494271, -0.010527852507270534, -0.025534424772119833]
e1_inter_vec_err_f_g, e2_inter_vec_err_f_g, size_inter_vec_err_f_g=[5.1056606138171768e-07, 5.164858453882987e-07, 4.8619773779010582e-07, 2.6111756348926725e-07, 2.8330629677279429e-07, 6.0225016937828292e-07], [3.4272508776473593e-07, 4.0167748777233684e-07, 4.5293157373999848e-07, 3.5642674274665272e-07, 4.3214795894765482e-07, 5.39522876192917e-07] ,[3.4054657295129468e-07, 3.3982026450161438e-07, 3.351783860165878e-07, 2.6373866827523519e-07, 2.8662953369979434e-07, 2.8329504789462062e-07]

# 1.1.3 delta
# 1.1.3.1  J FILTER
x_vec_d, e1_inter_vec_j_d, e2_inter_vec_j_d, size_inter_vec_j_d= np.array([ -1.00000000e-15 ,  -4.00000000e-16  , 2.00000000e-16 ,  8.00000000e-16, 1.40000000e-15 ,  2.00000000e-15])*10**15 ,[0.0059139642864465712, -0.00050786107778549177, 0.00077195327728986764, 0.00307374257594347, 0.0048099128343164922, 0.0061603431496769188] ,[0.058581705093383792, 0.028459012508392334, -0.012209473550319674, -0.037745391428470616, -0.051303016543388369, -0.058579500615596775] , [0.48355172510978972, 0.1586372548678987, -0.062177471593994206, -0.19844783151689685, -0.28438508804461871, -0.34040752026363685]
e1_inter_vec_err_j_d, e2_inter_vec_err_j_d, size_inter_vec_err_j_d= [5.4898300639056636e-06, 9.939851524930626e-06, 2.7783648102706584e-05, 0.00031973904676793423, 0.0011549013687920055, 0.0025585860023642137] ,[1.3485721789857574e-06, 1.4953937054925076e-06, 2.25987298608946e-06, 2.2569030226551392e-05, 7.21225345840278e-05, 0.00014507857906562673] ,[1.3477954191714852e-05, 4.7548313894978945e-06, 1.3889569298200912e-05, 0.00013834924898513951, 0.00040477971861512609, 0.00081699157520406665]
# 1.1.3.2  Y FILTER
e1_inter_vec_y_d, e2_inter_vec_y_d, size_inter_vec_y_d= [0.023601562455296515, 0.0047751702740788469, 6.4481347799301425e-05, 0.0032457004860043524, 0.0067280624934937805, 0.010405947247054428] ,[0.038414582610130303, 0.03015007019042968, -0.018783626556396486, -0.061005321741104124, -0.080507738590240477, -0.089618221223354333] ,[0.62284625084687062, 0.22407664495645604, -0.089511104691340937, -0.26993132650555135, -0.36458149870024847, -0.4182719144522109]
e1_inter_vec_err_y_d, e2_inter_vec_err_y_d, size_inter_vec_err_y_d= [7.0157455817682711e-05, 8.8166138306043507e-05, 0.00017637338298393343, 0.0021093390603538239, 0.0063236319530591573, 0.012576025063459581] ,[1.2339320393284889e-05, 1.2004590375880231e-05, 1.886533837166805e-05, 0.00019997513807405224, 0.00058962256163250346, 0.0010567332622614929], [7.0938137138432152e-05, 7.4967699650275448e-05, 0.00010955241898099397, 0.00088676859795751599, 0.0024824064428367056, 0.0039843200577199585]
# 1.1.3.3  H FILTER
e1_inter_vec_h_d, e2_inter_vec_h_d, size_inter_vec_h_d=[-0.0014336044713854795, -0.00066950198262929962, 0.0003237440437078474, 0.0011835725232958792, 0.0018239566311240194, 0.0022956963069736954] ,[0.028601739108562469, 0.0098362684249877937, -0.004166335761547086, -0.014163000285625454, -0.021244479268789293, -0.026277481019496916] ,[0.19251012278442783, 0.066579087273389445, -0.029219085609901597, -0.10386545051279676, -0.16305345769525709, -0.21047232233510763]
e1_inter_vec_err_h_d, e2_inter_vec_err_h_d, size_inter_vec_err_h_d= [6.4520842233149284e-05, 3.644656965488704e-05, 2.2090896731697005e-05, 0.00011492072654606751, 0.00021632456729508244, 0.00034622466793215431], [6.1255862934875585e-06, 2.923516686210975e-06, 1.5361360117327357e-06, 6.8038958260981162e-06, 1.2042818777639035e-05, 1.8272839658969043e-05], [4.4147075982803107e-05, 2.0081941624942871e-05, 1.1654803117917417e-05, 4.8752672680530327e-05, 9.0746224937214215e-05, 0.0001315157681941337]
# 1.1.3.4  F FILTER
e1_inter_vec_f_d, e2_inter_vec_f_d, size_inter_vec_f_d= [-0.00018544929102063171, -7.3326267302036566e-05, 3.6160238087177556e-05, 0.00014232281595468515, 0.00024537004530429825, 0.00034481646493077271] ,[0.0023996627330780044, 0.00093331381678581407, -0.00045374229550361716, -0.0017658713459968571, -0.0030072550475597377, -0.0041823482513427719] ,[0.022445042975058192, 0.0088082131526880224, -0.0043222854761152171, -0.0169734662960228, -0.029169998792587158, -0.040935131547155341]
e1_inter_vec_err_f_d, e2_inter_vec_err_f_d, size_inter_vec_err_f_d=[1.3758312095292173e-06, 6.7740205032074659e-07, 3.7953533171982795e-07, 1.8077322775089194e-06, 3.3465507757387878e-06, 5.083503815291833e-06],[4.6285328619469529e-07, 3.6666413840576553e-07, 3.1105459263705687e-07, 4.2388384884219797e-07, 5.3826958653128934e-07, 5.8380334439324752e-07], [8.1390733687682763e-07, 4.3766958216886538e-07, 2.3361993176303752e-07, 7.4570213550805558e-07, 1.371486406653596e-06, 2.1979198768794297e-06]

fig=plt.figure()

ax = fig.add_subplot (331)
ax.errorbar( x_vec_b, size_inter_vec_f_b, yerr= size_inter_vec_err_f_b, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_j_b, yerr= size_inter_vec_err_j_b, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_y_b, yerr= size_inter_vec_err_y_b, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_h_b, yerr= size_inter_vec_err_h_b, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\beta$ ($\times$10$^{6}$)"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta R/R$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (332)
ax.errorbar( x_vec_g, size_inter_vec_f_g, yerr= size_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_j_g, yerr= size_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_y_g, yerr= size_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_h_g, yerr= size_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\gamma$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (333)
ax.errorbar( x_vec_d, size_inter_vec_f_d, yerr= size_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_j_d, yerr= size_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_y_d, yerr= size_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_h_d, yerr= size_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\delta$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])



ax = fig.add_subplot (334)
ax.errorbar( x_vec_b, e1_inter_vec_f_b, yerr= e1_inter_vec_err_f_b, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_j_b, yerr= e1_inter_vec_err_j_b, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_y_b, yerr= e1_inter_vec_err_y_b, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_h_b, yerr= e1_inter_vec_err_h_b, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\beta$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta$$e_1$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (335)
ax.errorbar( x_vec_g, e1_inter_vec_f_g, yerr= e1_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e1_inter_vec_j_g, yerr= e1_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e1_inter_vec_y_g, yerr= e1_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e1_inter_vec_h_g, yerr= e1_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\gamma$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (336)
ax.errorbar( x_vec_d, e1_inter_vec_f_d, yerr= e1_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e1_inter_vec_j_d, yerr= e1_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e1_inter_vec_y_d, yerr= e1_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e1_inter_vec_h_d, yerr= e1_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\delta$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])




ax = fig.add_subplot (337)
ax.errorbar( x_vec_b, e2_inter_vec_f_b, yerr= e2_inter_vec_err_f_b, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e2_inter_vec_j_b, yerr= e2_inter_vec_err_j_b, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e2_inter_vec_y_b, yerr= e2_inter_vec_err_y_b, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e2_inter_vec_h_b, yerr= e2_inter_vec_err_h_b, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"$\beta$ ($\times$10$^{6}$)"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$\Delta$$e_2$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (338)
ax.errorbar( x_vec_g, e2_inter_vec_f_g, yerr= e2_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e2_inter_vec_j_g, yerr= e2_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e2_inter_vec_y_g, yerr= e2_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e2_inter_vec_h_g, yerr= e2_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"$\gamma$ ($\times$10$^{10}$)"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (339)
ax.errorbar( x_vec_d, e2_inter_vec_f_d, yerr= e2_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e2_inter_vec_j_d, yerr= e2_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e2_inter_vec_y_d, yerr= e2_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e2_inter_vec_h_d, yerr= e2_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"$\delta$ ($\times$10$^{15}$)"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])


fig.tight_layout()
fig.suptitle ("AB magnitude = 18.2", size=11)
plt.subplots_adjust(top=0.925)
pp.savefig()
pp.close()




