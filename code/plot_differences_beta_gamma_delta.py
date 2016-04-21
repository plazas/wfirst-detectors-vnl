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
x_vec_b, e1_inter_vec_j_b, e2_inter_vec_j_b, size_inter_vec_j_b = np.array( [ -5.00000000e-07  , -1.00000000e-07   ,3.00000000e-07 ,   7.00000000e-07 , 1.10000000e-06 ,  1.50000000e-06])*10**6 , [-8.2709696143865301e-05, -8.2808844745159291e-05, -8.292195387184585e-05, -8.2961302250623558e-05, -8.3049051463600549e-05, -8.3049908280372501e-05] , [6.0842186212538286e-05, 6.1216205358504973e-05, 6.1700344085690857e-05, 6.2253177165984277e-05, 6.2627792358370128e-05, 6.3064247369765681e-05] , [-0.00010149345507671547, -9.9967310794989435e-05, -9.8434917514588218e-05, -9.68615585779875e-05, -9.5333331296635352e-05, -9.3957857310131843e-05]
e1_inter_vec_err_j_b, e2_inter_vec_err_j_b, size_inter_vec_err_j_b = [0.0011596866666935663, 0.0011602900069242594, 0.0011608507211099237, 0.001161325672026104, 0.0011617347206291401, 0.0011620769434077718],  [0.0017016137913698521, 0.0016998852374510171, 0.0016981006947498366, 0.0016962766176573977, 0.001694401066389432, 0.0016924807452711893] , [0.0022127848405196254, 0.0021940544655270288, 0.0021756323166525955, 0.0021574756010956505, 0.0021395088467121028, 0.0021218229884804704]
# 1.1.1.2 Y filter
e1_inter_vec_y_b, e2_inter_vec_y_b, size_inter_vec_y_b =  [-8.6908657103777399e-05, -8.6976690217852615e-05, -8.730736561119202e-05, -8.7384497746821694e-05, -8.7674073874950077e-05, -8.7848501279949966e-05] ,[4.1804611682895451e-05, 4.2207390069960996e-05, 4.2997896671241599e-05, 4.3304711580247624e-05, 4.411488771435407e-05, 4.4697970151928447e-05] ,  [-0.00014146487282743703, -0.0001393629801294083, -0.00013728840293141475, -0.00013535950839948629, -0.00013330594312434439, -0.00013139105654108251]
e1_inter_vec_err_y_b, e2_inter_vec_err_y_b, size_inter_vec_err_y_b= [0.0011558311850658949, 0.001157571344697281, 0.0011592336476857006, 0.0011608280249478307, 0.0011623424450621111, 0.0011638103796126071] , [0.0018970022745123892, 0.0018981144714412315, 0.0018990393700767201, 0.0018998284513328815, 0.0019004523918119208, 0.0019009624876830929] ,  [0.0026898816900569364, 0.0026689327262117519, 0.0026480968399882092, 0.0026274480297136002, 0.002606975810787145, 0.0025866385582507005]
# 1.1.1.3 H filter
e1_inter_vec_h_b, e2_inter_vec_h_b, size_inter_vec_h_b =  [-6.2000318430364206e-05, -6.2014758586883609e-05, -6.2027205713093431e-05, -6.2039406038820778e-05, -6.2080072239041153e-05, -6.2042591162025853e-05] ,[4.2707696557059139e-05, 4.2929798364639977e-05, 4.3149068951607719e-05, 4.3369829654694573e-05, 4.3617263436316055e-05, 4.3841078877449036e-05] , [-7.2152916572756356e-05, -7.136808692635e-05, -7.0494101673308491e-05, -6.972358898564179e-05, -6.893029931819617e-05, -6.813050194217962e-05]
e1_inter_vec_err_h_b, e2_inter_vec_err_h_b, size_inter_vec_err_h_b= [0.0009417617129320007, 0.00094134973053544601, 0.00094091759964688812, 0.00094046905009569624, 0.00094001442612967059, 0.00093953760320169112] , [0.0012407035926138195, 0.0012389401104846421, 0.0012371691994629631, 0.0012353905368097579, 0.0012336183475366534, 0.0012319027472415193] , [0.0014112359550213405, 0.0014021060632790614, 0.0013930960417167294, 0.0013842007958957323, 0.0013754678215368375, 0.0013668600834573254]
# 1.1.1.4 F filter
e1_inter_vec_f_b, e2_inter_vec_f_b, size_inter_vec_f_b= [-2.8067990206180165e-05, -2.8010644018649195e-05, -2.7962229214608625e-05, -2.7950592339038755e-05, -2.8023961931467005e-05, -2.8046010993421165e-05] ,[1.0404288768760816e-05, 1.0554790496826588e-05, 1.0656230151653428e-05, 1.0701715946196538e-05, 1.0684207081793975e-05, 1.0830163955689309e-05] , [-4.0340360021899889e-05, -4.0092345186444331e-05, -3.9886164178611906e-05, -3.9727793259598519e-05, -3.9439139848685969e-05, -3.9258656763643262e-05]
e1_inter_vec_err_f_b, e2_inter_vec_err_f_b, size_inter_vec_err_f_b= [0.00047100119717654714, 0.00047077997638046828, 0.00047055009349334357, 0.00047034800276928219, 0.00047013497022424327, 0.00046990513556448656] , [0.00059835343293861338, 0.0005978333642529147, 0.00059731247142504236, 0.00059680613253141896, 0.00059629079539242149, 0.00059577571221745916] , [0.00062315998074495256, 0.00062112108675811738, 0.00061909233579155831, 0.00061710814441836214, 0.00061514398170558267, 0.00061316537980472569]

# 1.1.2 gamma
# 1.1.2.1  J FILTER
x_vec_g, e1_inter_vec_j_g, e2_inter_vec_j_g, size_inter_vec_j_g=np.array([ -1.00000000e-10 , -7.00000000e-11 , -4.00000000e-11 , -1.00000000e-11, 2.00000000e-11 ,  5.00000000e-11])*10**10, [-8.5093807429075132e-05, -8.5422787815332417e-05, -8.5551617667080549e-05, -8.5841184481978515e-05, -8.6254421621560952e-05, -8.6642540991308095e-05] ,  [7.7620893716811852e-05, 7.8361779451371628e-05, 7.9017579555510915e-05, 8.0180168151857138e-05, 8.0908983945848496e-05, 8.1770718097714807e-05] ,  [-0.00016781824135416956, -0.00016541723729288749, -0.00016302595389563023, -0.00016052496585954645, -0.00015826005109739749, -0.00015584238304028108]
e1_inter_vec_err_j_g, e2_inter_vec_err_j_g, size_inter_vec_err_j_g = [0.0012075457744127646, 0.0012124533679309295, 0.0012172562901659721, 0.0012219844195735395, 0.0012266273089846025, 0.0012311532119371276] ,  [0.0019811502201320069, 0.0019862566136391643, 0.0019912936340963547, 0.0019961203853301225, 0.0020008508568122325, 0.0020054855978127745] ,  [0.0036496186485231305, 0.0036174219594182449, 0.0035856318759729501, 0.0035541789550982972, 0.003523108174792879, 0.0034923622806322671]
# 1.1.2.2  Y FILTER
e1_inter_vec_y_g, e2_inter_vec_y_g, size_inter_vec_y_g= [-9.9148098379373705e-05, -9.9838264286514387e-05, -0.00010016866959631445, -0.00010052807629108404, -0.00010086208581924457, -0.00010132803581655017] ,[4.7985166311265983e-05, 4.8429071903197121e-05, 4.9399882555007658e-05, 5.0233751535415373e-05, 5.112782120704956e-05, 5.2210986614254775e-05] , [-0.00025315865234560198, -0.00024956841510141768, -0.00024630176147146464, -0.00024288452239344506, -0.00023940074555903746, -0.00023605004423706832]
e1_inter_vec_err_y_g, e2_inter_vec_err_y_g, size_inter_vec_err_y_g=[0.0012629538119416387, 0.0012707307698250701, 0.0012784353833264655, 0.0012860438841085567, 0.0012935610005408472, 0.0013010127092019483]  ,[0.0024089518854543465, 0.0024251543889381249, 0.0024410231667300361, 0.0024565594578417529, 0.0024717043991983293, 0.002486471321689227] , [0.0048777554459071004, 0.0048392814533008161, 0.0048005903158706764, 0.0047617788955751587, 0.0047228253162403429, 0.0046837303259650584]
# 1.1.2.2  H FILTER
e1_inter_vec_h_g, e2_inter_vec_h_g, size_inter_vec_h_g = [-5.7022003456950238e-05, -5.7115438394248432e-05, -5.7219723239540992e-05, -5.7322522625327093e-05, -5.7422220706939737e-05, -5.7545201852917579e-05] ,[6.0793906450272026e-05, 6.1070919036878145e-05, 6.1357840895652361e-05, 6.1640590429320181e-05, 6.1924904584871037e-05, 6.223067641258212e-05] , [-9.2139390951377913e-05, -9.1241977947758014e-05, -9.0370595778023206e-05, -8.9496610525173763e-05, -8.864084685590523e-05, -8.7787035499303778e-05]
e1_inter_vec_err_h_g, e2_inter_vec_err_h_g, size_inter_vec_err_h_g= [0.00085864160043035728, 0.00086009172485664552, 0.0008615239442926573, 0.00086293519175402195, 0.00086432541951942234, 0.00086571381291195568] , [0.0011607819627452557, 0.0011616743725370133, 0.001162567180063838, 0.001163446859541908, 0.0011643135883102269, 0.0011651743148957789] , [0.0018834230237530429, 0.0018731233133647404, 0.0018629505934877942, 0.0018528846004595872, 0.0018429234287375808, 0.0018331061060357552]
# 1.1.2.2  F FILTER
e1_inter_vec_f_g, e2_inter_vec_f_g, size_inter_vec_f_g=[-1.537501811981196e-05, -1.5445300377905463e-05, -1.5288456343114481e-05, -1.5396745875477853e-05, -1.5336317010223882e-05, -1.5383795835076931e-05] ,[1.3828426599502216e-05, 1.3842321932316243e-05, 1.3933219015598714e-05, 1.3878904283046723e-05, 1.3955123722552837e-05, 1.393940299749423e-05] , [-2.6087574934747783e-05, -2.6017652680048186e-05, -2.5863465143757258e-05, -2.5893944075240371e-05, -2.5772028348953756e-05, -2.5737366034679354e-05]
e1_inter_vec_err_f_g, e2_inter_vec_err_f_g, size_inter_vec_err_f_g=[0.00024153237467615965, 0.00024163250537225536, 0.00024171086038696996, 0.0002418032981343501, 0.00024187881968963196, 0.00024196679881693464] , [0.00030297391178489899, 0.00030303109026507514, 0.00030307850209280681, 0.0003031216481335844, 0.00030315790720962689, 0.00030319991492282742] , [0.00044686266119436512, 0.00044626004492886583, 0.00044563719067931032, 0.00044496690030104095, 0.00044432808429678116, 0.00044367228960969017]

# 1.1.3 delta
# 1.1.3.1  J FILTER
x_vec_d, e1_inter_vec_j_d, e2_inter_vec_j_d, size_inter_vec_j_d= np.array([ -1.00000000e-15 ,  -4.00000000e-16  , 2.00000000e-16 ,  8.00000000e-16, 1.40000000e-15 ,  2.00000000e-15])*10**15 ,[-2.2167647257447485e-05, -2.2084861993789533e-05, -2.2187419235706192e-05, -2.2201724350452493e-05, -2.2270604968070812e-05, -2.2201947867870314e-05] , [1.9492655992510711e-05, 1.9617825746533757e-05, 1.966208219525173e-05, 1.9809007644654152e-05, 1.9934624433518845e-05, 1.9998252391815185e-05] , [-6.2840622489132019e-05, -6.2594828551094668e-05, -6.2369864607756401e-05, -6.2066441017646351e-05, -6.1733161101706899e-05, -6.1545691148895407e-05]
e1_inter_vec_err_j_d, e2_inter_vec_err_j_d, size_inter_vec_err_j_d= [0.00033057121258586568, 0.00033113767382653711, 0.00033171265632989664, 0.00033227635161669713, 0.00033284459903565908, 0.00033341224152066854] , [0.00068510402385633864, 0.00068607353669231401, 0.00068706562298907086, 0.00068805456078832583, 0.00068902550858258925, 0.00068998512776277318] ,  [0.001452092110713483, 0.001448533884946746, 0.0014449769839947949, 0.0014414412316730285, 0.0014379377010548947, 0.0014343913268366488]
# 1.1.3.2  Y FILTER
e1_inter_vec_y_d, e2_inter_vec_y_d, size_inter_vec_y_d= [-2.9933257028460571e-05, -2.9972018674019556e-05, -3.0019301921129348e-05, -3.0031241476532449e-05, -2.9938481748103991e-05, -2.992129884660244e-05] ,[7.681995630239025e-06, 7.6556205749492286e-06, 7.6949596405032072e-06, 7.9296529292760534e-06, 7.8161060810066915e-06, 7.7089667319979453e-06] , [-0.00010461696385005159, -0.00010434591004309324, -0.00010402722921097007, -0.00010359578439193773, -0.00010318885348303608, -0.00010301025213752091]
e1_inter_vec_err_y_d, e2_inter_vec_err_y_d, size_inter_vec_err_y_d= [0.00036697760148499326, 0.00036793286254763153, 0.00036888072182310837, 0.00036983183834220429, 0.00037077678421759122, 0.00037174247954168467] , [0.00094789252773108841, 0.00095103892829324172, 0.00095418246685463821, 0.00095728469474985701, 0.00096039394485582577, 0.00096348787983602409] , [0.0021457293743681678, 0.002140733187542646, 0.0021357025713913238, 0.0021306376759554005, 0.0021255626730891097, 0.0021205257818772711]
# 1.1.3.3  H FILTER
e1_inter_vec_h_d, e2_inter_vec_h_d, size_inter_vec_h_d=[-1.3027042150497576e-05, -1.3035414740443177e-05, -1.3043368235228832e-05, -1.3052057474849891e-05, -1.3060551136732118e-05, -1.3068951666353588e-05] ,[1.6404688358307164e-05, 1.6420483589172781e-05, 1.6445219516741104e-05, 1.6471445560427982e-05, 1.6491636633873818e-05, 1.6511976718890237e-05] , [-2.7732599819046965e-05, -2.7662316567260704e-05, -2.7640190358326768e-05, -2.753997164717359e-05, -2.7502877708704963e-05, -2.7456022207423823e-05]
e1_inter_vec_err_h_d, e2_inter_vec_err_h_d, size_inter_vec_err_h_d= [0.00020719794693002116, 0.00020734032391264541, 0.00020748245919419123, 0.00020762436107730058, 0.00020776600634861631, 0.00020790753400577481] , [0.00030947247590945598, 0.0003096348236650127, 0.00030979751027189554, 0.000309958873287884, 0.00031012094889622173, 0.00031028344149058298] , [0.00061811892949696905, 0.00061736707345579317, 0.00061662033884819062, 0.00061587557389869636, 0.00061512460147717567, 0.00061437942170227669]
# 1.1.3.4  F FILTER
e1_inter_vec_f_d, e2_inter_vec_f_d, size_inter_vec_f_d= [-2.0752707496268352e-06, -2.0752800628534401e-06, -2.0782183855769848e-06, -2.0911917090416214e-06, -2.1507078781724758e-06, -2.0536454394459204e-06] ,[2.4928525090083669e-06, 2.4937093257908215e-06, 2.4920701980644248e-06, 2.4812296032911129e-06, 2.4292245507241681e-06, 2.5138258934022384e-06] , [-4.3005174845101028e-06, -4.2969317278840346e-06, -4.2742219356284308e-06, -4.292150718955279e-06, -4.3184462677059446e-06, -4.2527073956732944e-06]
e1_inter_vec_err_f_d, e2_inter_vec_err_f_d, size_inter_vec_err_f_d= [3.321850611825653e-05, 3.3221675473585945e-05, 3.3228639535228819e-05, 3.3235077359085198e-05, 3.3243825753444328e-05, 3.3235322715613729e-05] , [4.2796971506422294e-05, 4.2799999821268916e-05, 4.2805214630087663e-05, 4.2811347372082061e-05, 4.2816874770742732e-05, 4.2820392383340074e-05] , [8.0718166860218319e-05, 8.0704280200726071e-05, 8.0695083916967409e-05, 8.0694199551756924e-05, 8.0695427436305289e-05, 8.0670622203576664e-05]

f=100
                                                                              
fig=plt.figure()

ax = fig.add_subplot (331)
ax.errorbar( x_vec_b, size_inter_vec_f_b, yerr= f*np.array(size_inter_vec_err_f_b), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_j_b, yerr= f*np.array(size_inter_vec_err_j_b), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_y_b, yerr= f*np.array(size_inter_vec_err_y_b), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, size_inter_vec_h_b, yerr= f*np.array(size_inter_vec_err_h_b), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\beta$ ($\times$10$^{6}$)"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$d_{\Delta R/R}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (332)
ax.errorbar( x_vec_g, size_inter_vec_f_g, yerr= f*np.array(size_inter_vec_err_f_g), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_j_g, yerr= f*np.array(size_inter_vec_err_j_g), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_y_g, yerr= f*np.array(size_inter_vec_err_y_g), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, size_inter_vec_h_g, yerr= f*np.array(size_inter_vec_err_h_g), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
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
ax.errorbar( x_vec_d, size_inter_vec_f_d, yerr= f*np.array(size_inter_vec_err_f_g), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_j_d, yerr= f*np.array(size_inter_vec_err_j_g), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_y_d, yerr= f*np.array(size_inter_vec_err_y_g), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, size_inter_vec_h_d, yerr= f*np.array(size_inter_vec_err_h_g), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
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
ax.errorbar( x_vec_b, e1_inter_vec_f_b, yerr= f*np.array(e1_inter_vec_err_f_b), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_j_b, yerr=f*np.array( e1_inter_vec_err_j_b), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_y_b, yerr=f*np.array( e1_inter_vec_err_y_b), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e1_inter_vec_h_b, yerr= f*np.array(e1_inter_vec_err_h_b), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=False)
x1label=r"$\beta$"
lx=ax.set_xlabel(x1label, visible=False)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$d_{\Delta e_1}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (335)
ax.errorbar( x_vec_g, e1_inter_vec_f_g, yerr= f*np.array(e1_inter_vec_err_f_g), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e1_inter_vec_j_g, yerr= f*np.array(e1_inter_vec_err_j_g), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e1_inter_vec_y_g, yerr= f*np.array(e1_inter_vec_err_y_g), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e1_inter_vec_h_g, yerr= f*np.array(e1_inter_vec_err_h_g), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
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
ax.errorbar( x_vec_d, e1_inter_vec_f_d, yerr= f*np.array(e1_inter_vec_err_f_g), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e1_inter_vec_j_d, yerr= f*np.array(e1_inter_vec_err_j_g), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e1_inter_vec_y_d, yerr= f*np.array(e1_inter_vec_err_y_g), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e1_inter_vec_h_d, yerr= f*np.array(e1_inter_vec_err_h_g), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
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
ax.errorbar( x_vec_b, e2_inter_vec_f_b, yerr= f*np.array(e2_inter_vec_err_f_b), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e2_inter_vec_j_b, yerr= f*np.array(e2_inter_vec_err_j_b), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e2_inter_vec_y_b, yerr= f*np.array(e2_inter_vec_err_y_b), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_b, e2_inter_vec_h_b, yerr= f*np.array(e2_inter_vec_err_h_b), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
plt.axhline(y=0.,color='k',ls='solid')
ax.set_xticklabels([int(x) for x in ax.get_xticks()], visible=visible_x)
x1label=r"$\beta$ ($\times$10$^{6}$)"
lx=ax.set_xlabel(x1label, visible=visible_x)
ax.set_xscale('linear')
ax.set_yticklabels(ax.get_yticks(), visible= visible_y)
#y1label=r"$\Delta$e$_1/\beta$"
y1label=r"$d_{\Delta e_2}$"
#ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ly=ax.set_ylabel(y1label, visible=visible_y, size=12)
xmin, xmax=plt.xlim()
delta=(xmax-xmin)
plt.xlim ([xmin - 0.02*delta, xmax + 0.02*delta])

ax = fig.add_subplot (338)
ax.errorbar( x_vec_g, e2_inter_vec_f_g, yerr= f*np.array(e2_inter_vec_err_f_g), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e2_inter_vec_j_g, yerr= f*np.array(e2_inter_vec_err_j_g), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e2_inter_vec_y_g, yerr= f*np.array(e2_inter_vec_err_y_g), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_g, e2_inter_vec_h_g, yerr= f*np.array(e2_inter_vec_err_h_g), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
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
ax.errorbar( x_vec_d, e2_inter_vec_f_d, yerr= f*np.array(e2_inter_vec_err_f_g), ecolor = 'r', label='F184', fmt='r:+', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e2_inter_vec_j_d, yerr= f*np.array(e2_inter_vec_err_j_g), ecolor = 'b', label='J129', fmt='b--o', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e2_inter_vec_y_d, yerr= f*np.array(e2_inter_vec_err_y_g), ecolor = 'g', label='Y106', fmt='g-s', markersize=marker_size, alpha=alpha)
ax.errorbar( x_vec_d, e2_inter_vec_h_d, yerr= f*np.array(e2_inter_vec_err_h_g), ecolor = 'y', label='H158', fmt='y-.x', markersize=marker_size, alpha=alpha)
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
fig.suptitle ("AB magnitude = 20.0", size=11)
plt.subplots_adjust(top=0.925)
pp.savefig()
pp.close()




