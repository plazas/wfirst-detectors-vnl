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
x_vec=[18.3, 19, 20, 21, 22]
x_vec_b=x_vec
x_vec_g=x_vec
x_vec_d=x_vec

# 1.1 Figure 3
# 1.1.1 beta
# 1.1.1.1 J filter
e1_inter_vec_j_b, e2_inter_vec_j_b, size_inter_vec_j_b =  [0.00019690167158842084, 0.00010330328717827806, 4.1090967133641158e-05, 1.6333376988768579e-05, 6.49617984890943e-06] ,np.abs(np.array([-0.0033104890584945679, -0.001771166920661928, -0.0007142011821269983, -0.00028577730059623857, -0.00011399552226066728])) , np.abs(np.array([-0.015468129925790559, -0.0082765317459168736, -0.0033382015864973046, -0.0013359327502772077, -0.00053295610537621547] ))
e1_inter_vec_err_j_b, e2_inter_vec_err_j_b, size_inter_vec_err_j_b = [4.5027901288919729e-07, 2.7099822075049764e-07, 1.6483804962879991e-07, 9.0399434375126951e-08, 5.4338115979028633e-08] ,[5.1665882330321957e-07, 3.8714352419471205e-07, 2.6773082645177064e-07, 1.7421866438137307e-07, 8.2560905260495884e-08], [3.8633672946091329e-07, 2.7587950308532963e-07, 1.6565816579355902e-07, 1.0819590261387035e-07, 7.2163445068632627e-08]
# 1.1.1.2 Y filter
e1_inter_vec_y_b, e2_inter_vec_y_b, size_inter_vec_y_b = np.abs(np.array([-6.0582254081964632e-05, -3.8728304207325155e-05, -1.73868052661421e-05, -7.250234484673136e-06, -2.9443949460985703e-06] )) , np.abs(np.array([-0.0038858753442764238, -0.0020605184137821187, -0.00082566767930984111, -0.00032951742410659347, -0.00013132020831108039])) , np.abs(np.array([-0.018537439609319344, -0.0099161888178844696, -0.0039984786445593475, -0.0015999666218593223, -0.00063826367402239131] ))
e1_inter_vec_err_y_b, e2_inter_vec_err_y_b, size_inter_vec_err_y_b= [1.9639530744538168e-07, 1.6050587366354696e-07, 1.223111167188192e-07, 7.4961838729719984e-08, 4.9591240195405975e-08] ,[4.0384005313401515e-07, 3.518402088691094e-07, 1.919970528752604e-07, 1.0285946798243457e-07, 7.0505486261789784e-08] ,[3.0281777684246864e-07, 2.158877468034303e-07, 1.2883778032467686e-07, 7.8181685470450067e-08, 6.1250973727782876e-08]
# 1.1.1.3 H filter
e1_inter_vec_h_b, e2_inter_vec_h_b, size_inter_vec_h_b = np.abs(np.array([0.00013764604926109309, 7.3166899383068153e-05, 2.9389420524239403e-05, 1.1742282658815523e-05, 4.6773254871369101e-06] )) , np.abs(np.array([-0.0017084905505180365, -0.00091335929930210207, -0.00036813631653785607, -0.00014727905392646846, -5.8750808238983152e-05] )) ,  np.abs(np.array([-0.010549136360830175, -0.0056202991322085302, -0.0022602570657424435, -0.00090348901776855176, -0.00036026690774752069] ))
e1_inter_vec_err_h_b, e2_inter_vec_err_h_b, size_inter_vec_err_h_b= [2.6058448509165922e-07, 2.2348437369231719e-07, 1.4832646787638415e-07, 8.8314072387274768e-08, 5.2075527814900713e-08] ,[4.3492770456502384e-07, 3.4859242684343871e-07, 2.2045707737736892e-07, 1.3609671933344581e-07, 9.4278552888972642e-08], [2.6308648042910477e-07, 2.1856459437983719e-07, 1.3137244764161994e-07, 8.557409932406714e-08, 5.5982585986227436e-08]
# 1.1.1.4 F filter
e1_inter_vec_f_b, e2_inter_vec_f_b, size_inter_vec_f_b= np.abs(np.array( [5.1516401581466078e-05, 2.7229278348386288e-05, 1.0894164443015853e-05, 4.3349573388694243e-06, 1.7254613339898716e-06] )) ,np.abs(np.array([-0.00062052235007286, -0.00032897140830755178, -0.00013185013085603576, -5.2637644112108889e-05, -2.0980872213838334e-05] )) , np.abs(np.array([-0.0051374091967324089, -0.0027170575868423787, -0.0010872939630307233, -0.00043376261322903729, -0.00017283056424823861] ))
e1_inter_vec_err_f_b, e2_inter_vec_err_f_b, size_inter_vec_err_f_b=[1.9524122880585939e-07, 1.5400325221101859e-07, 9.2211139359874765e-08, 6.9999482693809907e-08, 5.3099313917246978e-08] ,[2.9871207214388973e-07, 2.0320637017208355e-07, 1.4135832354677853e-07, 9.5641624835236111e-08, 6.4309549498402991e-08] ,[1.6791092499484376e-07, 1.4291496692226087e-07, 9.9742419348795944e-08, 7.4679276925686175e-08, 5.9380826250142824e-08]

# 1.1.2 gamma
# 1.1.2.1  J FILTER
e1_inter_vec_j_g, e2_inter_vec_j_g, size_inter_vec_j_g=np.abs(np.array([-0.00064297635108232453, -0.00022431382909417129, -3.6966064944863247e-05, -5.8864243328569583e-06, -9.3488022685066576e-07])) ,np.abs(np.array([0.017410580813884732, 0.0044732397794723464, 0.00069105848670005941, 0.00010906800627708018, 1.7274171113970115e-05] )),  np.abs(np.array([0.087407041894478787, 0.021918423617126929, 0.0033732758145093712, 0.00053217012229681785, 8.427813662977845e-05] ))
e1_inter_vec_err_j_g, e2_inter_vec_err_j_g, size_inter_vec_err_j_g=[4.1755530512554943e-06, 1.5422316142910765e-06, 2.6405691893799887e-07, 7.9207821493668261e-08, 1.9535755850815507e-08] ,[8.6617029068346218e-07, 6.4009089177767134e-07, 2.5797843770932167e-07, 9.7605563256907843e-08, 1.7038876751351607e-08], [4.0078135204479973e-06, 1.2778627778803934e-06, 2.7961249409556268e-07, 8.0077361931685072e-08, 3.540078210877129e-08]
# 1.1.2.2  Y FILTER
e1_inter_vec_y_g, e2_inter_vec_y_g, size_inter_vec_y_g= np.abs(np.array([0.0016023941896855828, 0.00022506741806864732, 2.6559308171271478e-05, 3.9936788380145335e-06, 6.2670558690979683e-07] )),np.abs(np.array([0.019573820084333424, 0.0055881303548812886, 0.00088649779558182544, 0.00014045894145966187, 2.2265315055851885e-05] )), np.abs(np.array([0.1138750248505032, 0.028824834903240303, 0.0044399551902175195, 0.00070048696860347537, 0.00011094083424912027]))
e1_inter_vec_err_y_g, e2_inter_vec_err_y_g, size_inter_vec_err_y_g=[3.9371389829086751e-06, 1.910269065052068e-06, 3.8858551852386633e-07, 8.0849569054451458e-08, 1.8599274299666921e-08] ,[6.8556372322708642e-07, 5.4190538908321889e-07, 2.1826025372367053e-07, 1.1273632029855858e-07, 3.1398694973214966e-08]  ,[3.4746726155439191e-06, 1.5773749949110623e-06, 3.3487770474988962e-07, 8.524033444978079e-08, 3.202366357730605e-08]
# 1.1.2.2  H FILTER
e1_inter_vec_h_g, e2_inter_vec_h_g, size_inter_vec_h_g = np.abs(np.array([-0.00052423992194235321, -0.00014080004766583445, -2.2071087732911023e-05, -3.4910067915914186e-06, -5.5159442126759616e-07] )), np.abs(np.array([0.0072325474768877018, 0.0018477221578359604, 0.00028596349060535221, 4.5143887400625053e-05, 7.1474164724358306e-06] )), np.abs(np.array([0.046285595624040866, 0.011999762523198365, 0.0018665867085863085, 0.0002949594203055206, 4.6727971812314806e-05] ))
e1_inter_vec_err_h_g, e2_inter_vec_err_h_g, size_inter_vec_err_h_g= [6.6101948654697875e-06, 2.0387236002652375e-06, 3.5254601612602619e-07, 6.366614306159372e-08, 1.733053684334119e-08] ,[7.6229459988004784e-07, 5.289516121421775e-07, 2.132914106409429e-07, 6.5555753023779919e-08, 4.0627121724738819e-08],  [4.3162103029638919e-06, 1.3018150549729136e-06, 2.5814389326722608e-07, 5.7780888781154944e-08, 3.1774424110692337e-08]
# 1.1.2.2  F FILTER
e1_inter_vec_f_g, e2_inter_vec_f_g, size_inter_vec_f_g= np.abs(np.array([-0.00010281892493367206, -2.7936277911067224e-05, -4.4081034138799755e-06, -6.9647096097489281e-07, -1.0929536074413825e-07] )), np.abs(np.array([0.0012921334803104412, 0.00034799527376890363, 5.4777972400189484e-05, 8.6734816432022804e-06, 1.377351582051503e-06] )) , np.abs(np.array([0.011421910254137722, 0.003096164881458161, 0.00048824342912332061, 7.7314925928566593e-05, 1.2254329147878273e-05] ))
e1_inter_vec_err_f_g, e2_inter_vec_err_f_g, size_inter_vec_err_f_g= [3.0561947288736299e-07, 1.5051667544628253e-07, 6.7184678913064233e-08, 1.5193011440276204e-08, 1.3654169624445699e-08] ,[3.9935713879575187e-07, 2.504735465903926e-07, 1.0778826689522119e-07, 4.2951983255195341e-08, 3.9077762534797767e-08], [2.5322440499199287e-07, 1.5096605450915258e-07, 7.1382182762351393e-08, 3.5358605028954566e-08, 2.4459053146101336e-08]

# 1.1.3 delta
# 1.1.3.1  J FILTER
e1_inter_vec_j_d, e2_inter_vec_j_d, size_inter_vec_j_d= np.abs(np.array([0.0038559308042749761, 0.00062159046530723555, 3.6076754331588571e-05, 2.2584572434424659e-06, 1.4249235391608147e-07] )) , np.abs(np.array([-0.046415479108691218, -0.010180546343326569, -0.00067918896675109894, -4.2986571788786733e-05, -2.7129054069516269e-06]  )), np.abs(np.array([-0.25139030299210652, -0.051859464193442735, -0.0034825298218656908, -0.00022060139045079641, -1.39255422258866e-05] ))
e1_inter_vec_err_j_d, e2_inter_vec_err_j_d, size_inter_vec_err_j_d=[0.00062216353723449627, 1.8884949951594682e-05, 7.9960667546142057e-07, 5.7218233710699224e-08, 3.1445069268441539e-09] ,[5.2090580850352196e-05, 1.8485612681719569e-06, 2.632069845864016e-07, 4.85412044459238e-08, 3.5388297114356716e-09] , [0.00031664967326526636, 1.2034640127765004e-05, 5.3798333991205246e-07, 5.1998895600055832e-08, 3.4465983764153874e-08]
# 1.1.3.2  Y FILTER
e1_inter_vec_y_d, e2_inter_vec_y_d, size_inter_vec_y_d=  np.abs(np.array( [0.0041923589492216702, -5.6567788124084714e-05, -2.9280073940754284e-05, -1.9771419465547392e-06, -1.2468546628952026e-07] ))  , np.abs(np.array([-0.073787294179201116, -0.015496996343135831, -0.00097189337015151451, -6.1226487159721512e-05, -3.862231969828933e-06] )) ,  np.abs(np.array([-0.32982131735350551, -0.074786729375086361, -0.0050467899233639334, -0.0003197236716009766, -2.0177746353168267e-05] ))
e1_inter_vec_err_y_d, e2_inter_vec_err_y_d, size_inter_vec_err_y_d=np.abs(np.array( [0.0036940571319651755, 0.00011879904456895412, 5.2375666241501541e-06, 3.1946492486990712e-07, 2.0509101843815907e-08] )) ,np.abs(np.array([0.00043676331755044233, 1.4749439486062841e-05, 6.6295003228236791e-07, 5.4576507288763654e-08, 5.8457390185910433e-09] )) , np.abs(np.array( [0.0019032169935499004, 9.7711332809103044e-05, 4.7679577743480444e-06, 3.0363803847647574e-07, 3.5842463256476226e-08] ))
# 1.1.3.3  H FILTER
e1_inter_vec_h_d, e2_inter_vec_h_d, size_inter_vec_h_d= np.abs(np.array([0.0016179979825392365, 0.0002721070777624846, 1.742012798786179e-05, 1.0961852967738931e-06, 7.0668756961805168e-08] )) , np.abs(np.array([-0.018451258167624475, -0.0034605363756418238, -0.00022788316011428957, -1.4420375227928023e-05, -9.1008841991410684e-07] )), np.abs(np.array([-0.13884910366706085, -0.024208182991226606, -0.0015776716576059247, -9.9749551047645377e-05, -6.2942585680225705e-06] ))
e1_inter_vec_err_h_d, e2_inter_vec_err_h_d, size_inter_vec_err_h_d= np.abs(np.array([0.00014684684435472165, 1.5735974201861641e-05, 9.5835743929463754e-07, 7.1890741074496526e-08, 1.2409522963511591e-08])) ,np.abs(np.array([1.1946212922394299e-05, 1.5994212074466431e-06, 2.2366866319197627e-07, 6.0118891345322573e-08, 2.8614497739762397e-09] )), np.abs(np.array( [8.8623611611446826e-05, 1.1410671466122349e-05, 6.9557459542341527e-07, 5.5444103812212157e-08, 3.0634736403512809e-08] ))
# 1.1.3.4  F FILTER
e1_inter_vec_f_d, e2_inter_vec_f_d, size_inter_vec_f_d= np.abs(np.array( [0.00020148088224232201, 2.9806769452988988e-05, 1.8785335123536802e-06, 1.1697411537152196e-07, 7.5343996286045274e-09] )) , np.abs(np.array( [-0.0024735460057854659, -0.00037379942834377139, -2.3754760622976546e-05, -1.5014037489877869e-06, -9.4771385192246597e-08] )) , np.abs(np.array([-0.023896247224743813, -0.0035584211283920687, -0.00022561053622218385, -1.4241436844301124e-05, -8.994277169549481e-07] ))
e1_inter_vec_err_f_d, e2_inter_vec_err_f_d, size_inter_vec_err_f_d= [2.2799255051447669e-06, 3.0640815860881143e-07, 6.1157170222198703e-08, 2.2597962227897496e-08, 1.906369765484394e-10] ,[5.5287308680572783e-07, 2.4992504794968476e-07, 7.5093536574620079e-08, 2.1409596942902262e-08, 1.8491854736139376e-09] , [1.2576420956984396e-06, 2.1421848073885994e-07, 6.5940281123907586e-08, 3.1052498694016888e-08, 1.3024972388178838e-08]

fig=plt.figure()

ax = fig.add_subplot (331)
ax.errorbar( x_vec_b, size_inter_vec_f_b, yerr= size_inter_vec_err_f_b, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_j_b, yerr= size_inter_vec_err_j_b, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_y_b, yerr= size_inter_vec_err_y_b, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_h_b, yerr= size_inter_vec_err_h_b, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$|\Delta R/R|$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
plt.title (r"$\beta$", size=14)

ax = fig.add_subplot (332)
ax.errorbar( x_vec_g, size_inter_vec_f_g, yerr= size_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_j_g, yerr= size_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_y_g, yerr= size_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_h_g, yerr= size_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
plt.title (r"$\gamma$", size=14)



ax = fig.add_subplot (333)
ax.errorbar( x_vec_d, size_inter_vec_f_d, yerr= size_inter_vec_err_f_g, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_j_d, yerr= size_inter_vec_err_j_g, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_y_d, yerr= size_inter_vec_err_y_g, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_h_d, yerr= size_inter_vec_err_h_g, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])
plt.title (r"$\delta$", size=14)



ax = fig.add_subplot (334)
ax.errorbar( x_vec_b, e1_inter_vec_f_b, yerr= e1_inter_vec_err_f_b, ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_j_b, yerr= e1_inter_vec_err_j_b, ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_y_b, yerr= e1_inter_vec_err_y_b, ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_h_b, yerr= e1_inter_vec_err_h_b, ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$|\Delta$$e_1|$"
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
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yscale('log')
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
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yscale('log')
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
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
plt.ylim ([1e-5, 0.01])
ax.set_yscale('log')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$|\Delta$$e_2|$"
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
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yscale('log')
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
x1label=r"mag"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
#y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
#ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])


fig.tight_layout()
fig.suptitle (" ", size=11)
plt.subplots_adjust(top=0.925)
pp.savefig()
pp.close()




