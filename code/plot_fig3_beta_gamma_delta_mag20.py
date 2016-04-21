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
x_vec_b, e1_inter_vec_j_b, e2_inter_vec_j_b, size_inter_vec_j_b = np.array( [ -5.00000000e-07  , -1.00000000e-07   ,3.00000000e-07 ,   7.00000000e-07 , 1.10000000e-06 ,  1.50000000e-06])*10**6, [-4.0997136384248873e-05, -8.2098040729759388e-06, 2.4632550776004842e-05, 5.7555641978979234e-05, 9.0472828596830144e-05, 0.00012340050190687182] , [0.00072645500302314812, 0.00014430016279220385, -0.00042998254299163817, -0.00099652171134948864, -0.0015553593635559088, -0.0021065868437290195] , [0.0033970039449040691, 0.00067465072010078895, -0.002009896077927099, -0.004657331470892043, -0.007268347710655155, -0.0098435724947273408]
e1_inter_vec_err_j_b, e2_inter_vec_err_j_b, size_inter_vec_err_j_b =[1.4602670044147811e-07, 7.4715790476258791e-08, 1.2418410406101544e-07, 1.9935653131517995e-07, 2.5841539638900897e-07, 3.066413507421353e-07] , [2.5512994319547518e-07, 1.1175632474417928e-07, 2.1089770892658489e-07, 2.9536131342430344e-07, 3.794562484790833e-07, 4.4282467957195134e-07] , [1.8263536399959815e-07, 7.1117330119861813e-08, 1.3959104382572474e-07, 2.2141722510263212e-07, 2.7406801354702718e-07, 3.0251176154070349e-07]

# 1.1.1.2 Y filter
e1_inter_vec_y_b, e2_inter_vec_y_b, size_inter_vec_y_b = [2.0111277699470137e-05, 3.7987902760502899e-06, -1.076176762580927e-05, -2.3600347340107347e-05, -3.4791249781847454e-05, -4.4412408024073066e-05] ,[0.00083245903253555435, 0.00016595125198364869, -0.00049621954560279265, -0.0011539821326732585, -0.0018071369826793636, -0.0024555519223213138], [0.0040670895602787759, 0.00080789499734549523, -0.0024072403718450319, -0.0055789426896800717, -0.0087078325400336502, -0.011794506626350057]
e1_inter_vec_err_y_b, e2_inter_vec_err_y_b, size_inter_vec_err_y_b= [1.1864734379564019e-07, 5.8163712719328395e-08, 9.5941190281397818e-08, 1.4374027185814329e-07, 1.5457173873505018e-07, 1.6549302992960282e-07] , [2.1002477048602498e-07, 1.1389401133860023e-07, 1.4175469239306994e-07, 2.5145686335067138e-07, 3.569137314894146e-07, 3.7792881884697222e-07] , [1.618839530187561e-07, 7.9087557149001587e-08, 1.0149408032683631e-07, 1.6092108239454125e-07, 2.0118493973545396e-07, 2.503001851138228e-07]

# 1.1.1.3 H filter
e1_inter_vec_h_b, e2_inter_vec_h_b, size_inter_vec_h_b = [-2.9684165492653793e-05, -5.9039611369371066e-06, 1.7673484981059976e-05, 4.1045928373932957e-05, 6.4187878742814043e-05, 8.713915944099416e-05] ,[0.0003742670267820358, 7.4343234300612565e-05, -0.00022161006927490275, -0.00051371201872825835, -0.00080198496580124014, -0.0010865163058042511], [0.0022911068767179498, 0.00045572592590400476, -0.0013598182032241279, -0.0031558613352362787, -0.0049327132110571657, -0.0066906530457138706]
e1_inter_vec_err_h_b, e2_inter_vec_err_h_b, size_inter_vec_err_h_b= [1.2590603964686739e-07, 4.3343247336161631e-08, 1.0168815346490063e-07, 1.6247657585905653e-07, 2.0673633921429684e-07, 2.4284571139015263e-07] , [2.280653145942034e-07, 8.3215881232126786e-08, 1.915567309853835e-07, 2.8184399146950805e-07, 3.5253451720753966e-07, 3.624941001415831e-07] , [1.3141966533338475e-07, 5.4474741732696718e-08, 1.0943849935034192e-07, 1.5165532615972435e-07, 1.9789874229842872e-07, 2.2303553708035492e-07]
# 1.1.1.4 F filter
e1_inter_vec_f_b, e2_inter_vec_f_b, size_inter_vec_f_b= [-1.0967915877699965e-05, -2.1953834220767454e-06, 6.5397983416913985e-06, 1.5227999538183082e-05, 2.3863404057919927e-05, 3.2475083135068223e-05] ,[0.00013307850807905308, 2.6520676910877852e-05, -7.9247690737246218e-05, -0.00018426645547151477, -0.00028853185474872549, -0.00039203584194183233] , [0.0010947846122155069, 0.00021835176687202474, -0.00065327855734805147, -0.001520133243619246, -0.0023822356071770902, -0.0032396519744642403]
e1_inter_vec_err_f_b, e2_inter_vec_err_f_b, size_inter_vec_err_f_b=[9.7242100723072955e-08, 3.4811186869943752e-08, 7.6795548654558762e-08, 1.1428698478166444e-07, 1.4475850676239272e-07, 1.7311643737894242e-07] , [1.5258917755504891e-07, 5.7495351323158576e-08, 1.136071787938126e-07, 1.8360759426070228e-07, 1.9156497532446356e-07, 2.2169695530942759e-07],  [9.5448843856220908e-08, 4.8824778251834579e-08, 7.9508620932152677e-08, 1.195713592459023e-07, 1.3558395830749219e-07, 1.421021565761693e-07]

# 1.1.2 gamma
# 1.1.2.1  J FILTER
x_vec_g, e1_inter_vec_j_g, e2_inter_vec_j_g, size_inter_vec_j_g=np.array([ -1.00000000e-10 , -7.00000000e-11 , -4.00000000e-11 , -1.00000000e-11, 2.00000000e-11 ,  5.00000000e-11])*10**10, [0.00012638360261917061, 8.2691665738820921e-05, 4.4056978076695942e-05, 1.023655757307989e-05, -1.895470544695889e-05, -4.3631829321384464e-05] , [0.0035461293160915408, 0.0024826757609844241, 0.0014185602962970775, 0.00035453498363495153, -0.00070874661207198295, -0.0017706242203712431] ,  [0.018049669698505143, 0.012553481752884434, 0.0071270749950494428, 0.0017702058956671652, -0.0035173694149962654, -0.0087359857044530653]
e1_inter_vec_err_j_g, e2_inter_vec_err_j_g, size_inter_vec_err_j_g = [1.3091058078144712e-06, 9.864792525911711e-07, 5.8502805565213366e-07, 1.6594303334712581e-07, 3.3591872949718113e-07, 8.4541735981391582e-07] , [3.9251081353858669e-07, 3.3565919900094517e-07, 2.8144230683846061e-07, 1.3833735987698227e-07, 1.9130885845231871e-07, 3.7268504213364427e-07] , [1.1156692106458976e-06, 8.4265584500553485e-07, 5.0420623480163848e-07, 1.5263415661516308e-07, 2.9651341928529093e-07, 7.296838889982337e-07]
# 1.1.2.2  Y FILTER
e1_inter_vec_y_g, e2_inter_vec_y_g, size_inter_vec_y_g= [-8.8852616026997516e-05, -6.2039820477366407e-05, -3.5363230854272947e-05, -8.8085047900676553e-06, 1.7619114369153994e-05, 4.3927701190114247e-05] , [0.0011592718958854669, 0.00080714277923106949, 0.0004587544500827795, 0.00011406458914279855, -0.00022695407271385208, -0.00056439235806465479] , [0.0075454149074557098, 0.0052595142030977685, 0.0029927989035545477, 0.00074506072979239283, -0.0014839254761579268, -0.0036943829181871599]

e1_inter_vec_err_y_g, e2_inter_vec_err_y_g, size_inter_vec_err_y_g= [1.3361123482302633e-06, 9.3623743286477995e-07, 5.4641077187262401e-07, 1.4579091593751014e-07, 3.0220387993960585e-07, 7.0414902843464726e-07] , [3.8031943870106787e-07, 3.2336147246917462e-07, 2.5354092597388487e-07, 1.0276128039021973e-07, 2.0062303868060046e-07, 3.0177044200083162e-07] , [8.455569126816435e-07, 6.1537823538072924e-07, 3.6438878650420561e-07, 1.2102116983224583e-07, 1.9866758579486651e-07, 4.6655117499166788e-07]

# 1.1.2.2  H FILTER
e1_inter_vec_h_g, e2_inter_vec_h_g, size_inter_vec_h_g = [-8.8852616026997516e-05, -6.2039820477366407e-05, -3.5363230854272947e-05, -8.8085047900676553e-06, 1.7619114369153994e-05, 4.3927701190114247e-05] ,[0.0011592718958854669, 0.00080714277923106949, 0.0004587544500827795, 0.00011406458914279855, -0.00022695407271385208, -0.00056439235806465479] ,[0.0075454149074557098, 0.0052595142030977685, 0.0029927989035545477, 0.00074506072979239283, -0.0014839254761579268, -0.0036943829181871599]
e1_inter_vec_err_h_g, e2_inter_vec_err_h_g, size_inter_vec_err_h_g= [1.3361123482302633e-06, 9.3623743286477995e-07, 5.4641077187262401e-07, 1.4579091593751014e-07, 3.0220387993960585e-07, 7.0414902843464726e-07] , [3.8031943870106787e-07, 3.2336147246917462e-07, 2.5354092597388487e-07, 1.0276128039021973e-07, 2.0062303868060046e-07, 3.0177044200083162e-07] , [8.455569126816435e-07, 6.1537823538072924e-07, 3.6438878650420561e-07, 1.2102116983224583e-07, 1.9866758579486651e-07, 4.6655117499166788e-07]


# 1.1.2.2  F FILTER
e1_inter_vec_f_g, e2_inter_vec_f_g, size_inter_vec_f_g= [-1.766269095242027e-05, -1.2358003295958112e-05, -7.0518907159567572e-06, -1.7691450193525229e-06, 3.5071885213255535e-06, 8.7873311713336685e-06] ,[0.0002199357748031621, 0.0001537129282951359, 8.7708421051503201e-05, 2.1889396011829862e-05, -4.3702647089957222e-05, -0.00010906495153903795] , [0.0019585322407145722, 0.0013694064893638247, 0.0007816355568044897, 0.00019518298549222379, -0.00038994524270089312, -0.00097373000944672499]
e1_inter_vec_err_f_g, e2_inter_vec_err_f_g, size_inter_vec_err_f_g= [1.2648434179003205e-07, 1.0575445454199874e-07, 9.3111099599963238e-08, 3.4533065258032646e-08, 6.9448495679645787e-08, 8.6686056795513853e-08] , [1.9613808729754063e-07, 1.5975632834432897e-07, 1.3447562702671205e-07, 4.6534512635531953e-08, 9.5700409340996689e-08, 1.2706407658200006e-07] , [1.2707194361163542e-07, 9.7627125360028199e-08, 8.2864991461824229e-08, 4.5876146578174221e-08, 7.5345719465464832e-08, 9.1480616742352361e-08]


# 1.1.3 delta
# 1.1.3.1  J FILTER
x_vec_d, e1_inter_vec_j_d, e2_inter_vec_j_d, size_inter_vec_j_d= np.array([ -1.00000000e-15 ,  -4.00000000e-16  , 2.00000000e-16 ,  8.00000000e-16, 1.40000000e-15 ,  2.00000000e-15])*10**15, [-2.3719817399978725e-05, -9.5229316502809e-06, 4.770169034600154e-06, 1.9156895577907475e-05, 3.3644270151853441e-05, 4.8244595527648802e-05] , [0.00045535683631896891, 0.00018189445137977323, -9.0820789337157649e-05, -0.00036280781030654879, -0.00063404262065887371, -0.00090456873178482058],  [0.0023381489319640571, 0.00093366052523912173, -0.00046604393149447533, -0.0018610269233392629, -0.0032512815060106548, -0.0046368465553966254]
e1_inter_vec_err_j_d, e2_inter_vec_err_j_d, size_inter_vec_err_j_d=[ 5.0873652142937341e-07, 2.1695187081433824e-07, 1.150066302298767e-07, 4.167387724953193e-07, 7.4003336716791023e-07, 1.0530594623513881e-06] , [2.2755863749069311e-07, 1.1851539047100671e-07, 7.8913713456186222e-08, 1.9918321468355114e-07, 2.6510066183986909e-07, 2.9732338870858692e-07] , [3.3494163163064604e-07, 1.5816380190128729e-07, 8.9365684484314169e-08, 2.8675137705809139e-07, 5.0246071388513869e-07, 7.292075613316126e-07]

# 1.1.3.2  Y FILTER
e1_inter_vec_y_d, e2_inter_vec_y_d, size_inter_vec_y_d= [2.1814052015543157e-05, 8.4872916340826907e-06, -4.1643530130387043e-06, -1.6113109886646653e-05, -2.756623551249518e-05, -3.8216877728700636e-05] ,[0.0006459242105484028, 0.00025857388973236331, -0.0001293857395648923, -0.0005179055035114227, -0.00090698719024657755, -0.0012966307997703548] , [0.0033887050195833737, 0.0013531893272503437, -0.00067543165671788417, -0.0026970605742605335, -0.0047117268632919694, -0.0067192686900005658]
e1_inter_vec_err_y_d, e2_inter_vec_err_y_d, size_inter_vec_err_y_d= [3.3625011529076512e-06, 1.3535254729433799e-06, 6.9226750370845241e-07, 2.7757362527791688e-06, 4.8974976556071496e-06, 7.0117570675352856e-06] , [4.3536960098340906e-07, 2.0205227309343573e-07, 1.1655823161087301e-07, 3.8035852458108932e-07, 6.2198482112620074e-07, 8.897981828670007e-07] , [3.0778416116491523e-06, 1.2335178589467589e-06, 6.356713438161834e-07, 2.5285045636205225e-06, 4.4334703105347836e-06, 6.4874193359246875e-06]

# 1.1.3.3  H FILTER
e1_inter_vec_h_d, e2_inter_vec_h_d, size_inter_vec_h_d= [-1.1630943045020138e-05, -4.6473369002343088e-06, 2.3229233920575402e-06, 9.3007180839778904e-06, 1.6284808516502364e-05, 2.324430271983143e-05] ,[0.00015262968838214791, 6.0978904366493922e-05, -3.0456334352493288e-05, -0.00012169979512691665, -0.00021273396909236991, -0.00030354425311088576] , [0.0010556516763289036, 0.00042187996858309649, -0.00021075744298431508, -0.0008422859394147242, -0.00147271073304166, -0.0021020168469510379]
e1_inter_vec_err_h_d, e2_inter_vec_err_h_d, size_inter_vec_err_h_d= [6.3404348752979473e-07, 2.5288877122424432e-07, 1.3399123891730514e-07, 5.1213661682412567e-07, 8.9957983808915748e-07, 1.2567558885477868e-06] , [1.4522668564923607e-07, 7.7251958493442209e-08, 7.287551479184005e-08, 1.3846400409328292e-07, 2.1772288245288e-07, 2.3758353649455761e-07] , [4.7903414002969979e-07, 1.9605760945751058e-07, 9.962190159376e-08, 3.8087841523848522e-07, 6.470848007439548e-07, 9.3297463008026715e-07]

# 1.1.3.4  F FILTER
e1_inter_vec_f_d, e2_inter_vec_f_d, size_inter_vec_f_d= [-1.2650201097133031e-06, -5.0164759159089002e-07, 2.4968758225424501e-07, 1.0019075125455335e-06, 1.7506582662462321e-06, 2.5066174566744285e-06] ,[1.5851333737374949e-05, 6.3433870673183793e-06, -3.1703338026994184e-06, -1.2674592435359607e-05, -2.2172704339026851e-05, -3.1668134033678507e-05], [0.00015047395794661966, 6.0181575380919835e-05, -3.009108630439683e-05, -0.00012033984275523379, -0.00021057485437158907, -0.00030077520294181204]

e1_inter_vec_err_f_d, e2_inter_vec_err_f_d, size_inter_vec_err_f_d= [3.5784116946783619e-08, 1.6526984959673885e-08, 2.2748590517856881e-08, 4.5412084251979924e-08, 5.8045291002061127e-08, 6.5932464503921235e-08] , [4.5344052328200672e-08, 4.268764947952557e-08, 2.1409596942294952e-08, 3.8124465116984809e-08, 7.5171189156464379e-08, 8.4652457706402105e-08] , [4.7072281309559612e-08, 2.7380721616109782e-08, 3.3267870043225655e-08, 4.3630922675137765e-08, 6.5168336956423367e-08, 6.9707402792721399e-08]


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
fig.suptitle ("AB magnitude = 20", size=11)
plt.subplots_adjust(top=0.925)
pp.savefig()
pp.close()




