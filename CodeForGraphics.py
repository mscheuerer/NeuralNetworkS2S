
import numpy as np
import numpy.ma as ma
import scipy as sp
import math
import os, sys
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches

import datetime
import time
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from netCDF4 import Dataset
from numpy import ma, loadtxt
from numpy.linalg import solve
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import kendalltau
from colorspace import diverging_hcl, sequential_hcl

plt.ion()

divcmp = diverging_hcl("Tropic",rev=True).cmap(name = "Diverging Color Map")
pcpcmp = sequential_hcl(h=[-220,20], c=[0,100], l=[100,30], power=1.5).cmap(name = "Precipitation Color Map")
clmcmp = sequential_hcl(h=[90,330], c=[30,100], l=[90,30],power=2.5).cmap(name = "Climatological Precipitation Color Map")

ncat = 20




###################################################################################################
#                                                                                                 #
#  Figure 1:  Illustrate discretization of analyzed precipitation amounts                         #
#                                                                                                 #
###################################################################################################


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/precip_PRISM_cal_19810101_20171231.npz")
#list(f1)
obs_precip = f1['precip']
obs_lat = f1['lat']
obs_lon = f1['lon']
obs_dates_ord = f1['dates_ord']
obs_dates = f1['dates']
f1.close()

ndts, nxy = obs_precip.shape


## Aggregate daily to weekly accumulations
obs_precip_week = np.zeros((ndts-6,nxy), dtype=np.float32)
for iday in range(7):
    obs_precip_week += obs_precip[iday:(ndts-6+iday),:]

ndts, nxy = obs_precip_week.shape

obs_dates_ord = obs_dates_ord[:ndts]
obs_dates = obs_dates[:ndts]


## Calculate day of the year ('doy') for each date in the observation data set
doy = np.zeros(ndts, dtype=np.int32)
for idt in range(ndts):
    doy[idt] = (datetime.date.fromordinal(int(obs_dates_ord[idt]))-datetime.date(obs_dates[idt,0],1,1)).days


pctl = np.zeros((nxy,99), dtype=np.float32)
pop = np.zeros(nxy, dtype=np.float32)
mean = np.zeros(nxy, dtype=np.float32)
thr = np.zeros((nxy,ncat-1), dtype=np.float32)

idd = 15

ind_doy = np.where(doy==idd)[0]
ind_doy_ext = np.append(np.append(ind_doy[0]-366,ind_doy),ind_doy[-1]+365)
wnd_ind = np.add.outer(ind_doy_ext,np.arange(-30,31)).flatten()                # data within a 61-day window around each date
imin = np.where(wnd_ind>=0)[0][0]                                              #  are considered to estimate climatological PoP
imax = np.where(wnd_ind<ndts)[0][-1]                                           #  and climatological percentiles

for ixy in range(nxy):
    y = obs_precip_week[wnd_ind[imin:(imax+1)],ixy]
    pop[ixy] = np.mean(y>0.254)
    mean[ixy] = np.mean(y)
    thr[ixy,0] = 0.254
    qtlv = 1. + pop[ixy]*((np.arange(1,ncat-1)/float(ncat-1))-1.)
    thr[ixy,1:] = np.quantile(y,qtlv)
    pctl[ixy,:] = np.percentile(y,np.arange(1,100))


itnf = np.logical_and(obs_lon==-120.625,obs_lat==39.375)     # coordinates of our example grid point in Tahoe National Forest
ilat = (obs_lat==39.375)                                     # latitude of our example transect


plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
    xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
    yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
plt.scatter(obs_lon,obs_lat,c=mean,marker='s',cmap=clmcmp,s=28,lw=.1,vmin=0,vmax=105,edgecolors=[.2,.2,.2])
cbar = plt.colorbar()
cbar.ax.set_yticklabels(['0 mm','20 mm','40 mm','60 mm','80 mm','100 mm'])
plt.plot([-124.5,-119.],[39.375,39.375],c='black',linewidth=2)
plt.scatter(obs_lon[itnf],obs_lat[itnf],c='red',marker='*',zorder=3)
plt.title('        Average 7-day precipitation amounts in January\n',fontsize=12)

plt.subplot(1, 2, 2, xlim=(-123.8,-120), \
    xticks=[-123.5,-122.5,-121.5,-120.5], xticklabels=['-123.5'+'\u00b0','-122.5'+'\u00b0','-121.5'+'\u00b0','-120.5'+'\u00b0'], \
    yticks=[0,100,200,300,400], yticklabels=['0 mm','100 mm','200 mm','300 mm','400 mm'])
plt.scatter(np.repeat(obs_lon[ilat,np.newaxis],99,axis=1),pctl[ilat,:],c='DodgerBlue',s=30)
plt.plot(np.repeat(obs_lon[ilat,np.newaxis],ncat-1,axis=1),thr[ilat,:],c='black',linewidth=1.5)
plt.title('Category boundaries along meridional transect\n',fontsize=12)

plt.tight_layout()







###################################################################################################
#                                                                                                 #
#  Figure 2:  ANN schematic and probability forecasts for case study at Tahoe National Forest     #
#                                                                                                 #
###################################################################################################


iyyyy = 2017
imm = 1
idd = 8

itnf = np.logical_and(obs_lon==-120.625,obs_lat==39.375)     # coordinates of our example grid point in Tahoe National Forest

f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
#list(f1)
obs_lat = f1['obs_lat']
obs_lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
pop_doy = f1['pop_doy']
thr_doy = f1['thr_doy']
qtev_doy = f1['qtev_doy']
obs_precip_week = f1['apcp_obs']
f1.close()

ntms, nxy = obs_precip_week.shape

for ivdate in range(ntms):
    if datetime.date.fromordinal(int(obs_dates_ord[ivdate])) == datetime.date(iyyyy,imm,idd):
        break


f2 = np.load("/home/michael/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates = f2['dates_ord']
f2.close()

ndts, nyrs, nlts = mod_dates.shape

iidate = np.zeros((3,2),dtype=np.int16)    # date and year index for selected date

for idt in range(ndts):
    for iyr in range(nyrs):
        for ilt in range(3):
            if datetime.date.fromordinal(int(mod_dates[idt,iyr,6+ilt*7])) == datetime.date(iyyyy,imm,idd):
                iidate[ilt,0] = idt
                iidate[ilt,1] = iyr


f3 = np.load("/home/michael/Desktop/CalifAPCP/stats/ensemble_stats_week2_ANN_yr"+str(iidate[0,1])+".npz")
doy_vdate = f3['doy_dts'][iidate[0,0]]
apcp_ens_pit = f3['apcp_ens_pit_verif'][iidate[0,0],:,:]
f3.close()


prob_cat_tnf = np.zeros((4,20),dtype=np.float32)   # Probability forcast for each category at TNF grid point  

for ilt in range(3):
    f5 = np.load("/home/michael/Desktop/CalifAPCP/forecasts/ANN-efi/probfcst_10-l1_week"+str(ilt+2)+"_yr"+str(iidate[ilt,1])+".npz")
    prob_cat_tnf[ilt,:] = f5['prob_fcst_cat'][iidate[ilt,0],itnf,:]
    f5.close()

prob_cat_tnf[3,:] = np.append(1.-pop_doy[doy_vdate,itnf],np.repeat(pop_doy[doy_vdate,itnf]/(ncat-1),ncat-1))   # Clim. probabilities


##  Set positions for ANN schematic

npr = 3
nhd = 5
ncl = 4

size = 450.

pcl_x = np.full(ncl,2.5,dtype=np.float32)
pcl_y = np.arange(5.5,5.5+ncl)
pcl_c = np.full(ncl,0.2,dtype=np.float32)

pred_x = np.full(npr,1,dtype=np.float32)
pred_y = np.arange(1,1+npr)
pred_c = np.full(npr,0.4,dtype=np.float32)

hid1_x =  np.full(nhd,2,dtype=np.float32)
hid1_y = np.arange(0,nhd)
hid1_c = np.full(nhd,0.6,dtype=np.float32)

hid2_x =  np.full(ncl,3,dtype=np.float32)
hid2_y = np.arange(0.5,ncl+0.5)
hid2_c = np.full(ncl,0.6,dtype=np.float32)

out_x = np.full(ncl,4.5,dtype=np.float32)
out_y = np.arange(3,3+ncl)
out_c = np.full(ncl,0.8,dtype=np.float32)

x = np.concatenate([pcl_x,pred_x,hid1_x,hid2_x,out_x-.5])
y = np.concatenate([pcl_y,pred_y,hid1_y,hid2_y,out_y])
colors = np.concatenate([pcl_c,pred_c,hid1_c,hid2_c+.1,out_c])


##  Now: actual plot

width = 0.2

plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1, xlim=[0.8,4.55])
plt.scatter(x, y, c=colors, s=size, alpha=0.5)
plt.axis('off')

for i in range(ncl):
    plt.arrow(pcl_x[i]+.15,pcl_y[i],out_x[i]-0.8-pcl_x[i],out_y[i]-pcl_y[i]+.1, head_width=.05, length_includes_head=True, color='k')
    plt.arrow(hid2_x[i]+.15,hid2_y[i],out_x[i]-0.8-hid2_x[i],out_y[i]-hid2_y[i]-.1, head_width=.05, length_includes_head=True, color='k')

for i in range(npr):
    for j in range(nhd):
        plt.arrow(pred_x[i]+.15,pred_y[i],hid1_x[j]-0.3-pred_x[i],.95*(hid1_y[j]-pred_y[i]), head_width=.05, length_includes_head=True, color='k')

for i in range(nhd):
    for j in range(ncl):
        plt.arrow(hid1_x[i]+.15,hid1_y[i],hid2_x[j]-0.3-hid1_x[i],.95*(hid2_y[j]-hid1_y[i]), head_width=.05, length_includes_head=True, color='k')

for i in range(nhd):
    plt.text(hid1_x[i],hid1_y[i],'ELU',horizontalalignment='center',verticalalignment='center',fontsize=9)

for i in range(ncl):
    plt.text(hid2_x[i],hid2_y[i],'ELU',horizontalalignment='center',verticalalignment='center',fontsize=9)

for i in range(ncl):
    plt.text(out_x[i]-.5,out_y[i],'S',horizontalalignment='center',verticalalignment='center',fontsize=9)

plt.text(1.35,7.5,'climatological\n log probabilities')
plt.text(.6,-.4,'input layer\n (predictors)')
plt.text(1.4,3.65,r'$W_1$')
plt.text(1.8,-.9,'hidden layer')
plt.text(2.5,3.9,r'$W_2$')
plt.text(2.8,-.8,'preliminary\n output layer')
plt.text(3.85,1.8,'output\n layer')
plt.text(0.7,8.6,'a)',fontsize=18)

ax = plt.subplot(1, 2, 2, xticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], xticklabels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.bar(np.arange(ncat)-3*width/2, prob_cat_tnf[0,:], width, label='week-2 probability forecast', color='orange')
ax.bar(np.arange(ncat)-width/2, prob_cat_tnf[1,:], width, label='week-3 probability forecast', color='seagreen')
ax.bar(np.arange(ncat)+width/2, prob_cat_tnf[2,:], width, label='week-4 probability forecast', color='r')
ax.bar(np.arange(ncat)+3*width/2, prob_cat_tnf[3,:], width, label='climatological probability', color='b')
ax.legend(loc=9,fontsize=11)
plt.text(-0.6,0.21,'b)',fontsize=18)

plt.tight_layout()







###################################################################################################
#                                                                                                 #
#  Figure 3:  Illustrate conversion of probability forecasts at TNF to predictive CDF             #
#                                                                                                 #
###################################################################################################

cdf_tnf = np.cumsum(prob_cat_tnf,axis=1)[:,:(ncat-1)]
chf_tnf = -np.log(1.-cdf_tnf)

xx = np.arange(315.)

cdf_ip_tnf = np.zeros((4,len(xx)),dtype=np.float32)
chf_ip_tnf = np.zeros((4,len(xx)),dtype=np.float32)

for ithr in range(4):
    itp_fct = interp1d(thr_doy[doy_vdate,itnf,:].squeeze(), chf_tnf[ithr,:], kind='linear',fill_value='extrapolate')
    chf_ip_tnf[ithr,:] = itp_fct(xx)
    cdf_ip_tnf[ithr,:] = 1.-np.exp(-itp_fct(xx))


plt.figure(figsize=(15,4))

ax = plt.subplot(1, 3, 1, xlim=[-5,320], xticks=[0,50,100,150,200,250,300], xticklabels=['0 mm','50 mm','100 mm','150 mm','200 mm','250 mm','300 mm'], yticks=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[0,:], label='week-2 cum. probabilities', color='orange')
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[1,:], label='week-3 cum. probabilities', color='seagreen')
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[2,:], label='week-4 cum. probabilities', color='r')
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[3,:], label='clim. cum. probabilities', color='b')
#ax.set_title('Cumulative probabilities\n')
ax.legend(loc=4,fontsize=10)
plt.text(10,0.91,'a)',fontsize=18)

ax = plt.subplot(1, 3, 2, xlim=[-5,320], xticks=[0,50,100,150,200,250,300], xticklabels=['0 mm','50 mm','100 mm','150 mm','200 mm','250 mm','300 mm'], yticks=[0,.5,1.,1.5,2.,2.5,3.,3.5])
ax.scatter(thr_doy[doy_vdate,itnf,:], chf_tnf[0,:], label='week-2 cum. hazard', color='orange')
ax.plot(xx, chf_ip_tnf[0,:], color='orange')
ax.scatter(thr_doy[doy_vdate,itnf,:], chf_tnf[1,:], label='week-3 cum. hazard', color='seagreen')
ax.plot(xx, chf_ip_tnf[1,:], color='seagreen')
ax.scatter(thr_doy[doy_vdate,itnf,:], chf_tnf[2,:], label='week-4 cum. hazard', color='r')
ax.plot(xx, chf_ip_tnf[2,:], color='r')
ax.scatter(thr_doy[doy_vdate,itnf,:], chf_tnf[3,:], label='clim. cum. hazard', color='b')
ax.plot(xx, chf_ip_tnf[3,:], color='b')
#ax.set_title('Cumulative hazard function\n')
ax.legend(loc=(.15,.7),fontsize=10)
plt.text(10,3.5,'b)',fontsize=18)

ax = plt.subplot(1, 3, 3, xlim=[-5,320], xticks=[0,50,100,150,200,250,300], xticklabels=['0 mm','50 mm','100 mm','150 mm','200 mm','250 mm','300 mm'], yticks=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[0,:], label='week-2 predictive CDF', color='orange')
ax.plot(xx, cdf_ip_tnf[0,:], color='orange')
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[1,:], label='week-3 predictive CDF', color='seagreen')
ax.plot(xx, cdf_ip_tnf[1,:], color='seagreen')
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[2,:], label='week-4 predictive CDF', color='r')
ax.plot(xx, cdf_ip_tnf[2,:], color='r')
ax.scatter(thr_doy[doy_vdate,itnf,:], cdf_tnf[3,:], label='clim. predictive CDF', color='b')
ax.plot(xx, cdf_ip_tnf[3,:], color='b')
#ax.set_title('Interpolated CDF\n')
ax.legend(loc=4,fontsize=10)
plt.text(10,0.91,'c)',fontsize=18)

plt.tight_layout()







###################################################################################################
#                                                                                                 #
#  Figure 4:  Illustrate construction of basis functions                                          #
#                                                                                                 #
###################################################################################################


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/precip_PRISM_cal_19810101_20171231.npz")
#list(f1)
obs_precip = f1['precip']
obs_lat = f1['lat']
obs_lon = f1['lon']
obs_dates_ord = f1['dates_ord']
obs_dates = f1['dates']
f1.close()

ndts, nxy = obs_precip.shape


r_basis = 7.
lon_ctr = np.outer(np.ones(3),np.arange(-124,-115,3.5)).reshape(9)[[0,1,4,5,8]]
lat_ctr = np.outer(np.arange(33,42,3.5)[::-1],np.ones(3)).reshape(9)[[0,1,4,5,8]]

dst_lon = np.abs(np.subtract.outer(obs_lon,lon_ctr))
dst_lat = np.abs(np.subtract.outer(obs_lat,lat_ctr))
dst = np.sqrt(dst_lon**2+dst_lat**2)
rbf = np.where(dst>r_basis,0.,(1.-(dst/r_basis)**3)**3)
basis = rbf/np.sum(rbf,axis=1)[:,None]
nbs = basis.shape[1]


plt.figure(figsize=(18.5,7.5))

for ibs in range(5):
    plt.subplot(2, 5, ibs+1, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
        xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
        yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
    plt.scatter(obs_lon,obs_lat,c=rbf[:,ibs],marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.0,vmax=1.0,edgecolors=[.2,.2,.2])
    plt.scatter(lon_ctr[ibs],lat_ctr[ibs],c='black',marker='*',zorder=3)
    #cbar = plt.colorbar()
    plt.title('   Preliminary basis function '+str(ibs+1)+'\n',fontsize=12)
    plt.subplot(2, 5, ibs+6, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
        xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
        yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
    plt.scatter(obs_lon,obs_lat,c=basis[:,ibs],marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.0,vmax=0.68,edgecolors=[.2,.2,.2])
    #plt.scatter(lon_ctr[ibs],lat_ctr[ibs],c='black',marker='*',zorder=3)
    #cbar = plt.colorbar()
    plt.title('      Basis function '+str(ibs+1)+'\n',fontsize=12)

plt.tight_layout()






###################################################################################################
#                                                                                                 #
#  Figure 5:  Schematic to explain CNN-based modeling approach                                    #
#                                                                                                 #
###################################################################################################


plt.figure(figsize=(11,4.5))

ax = plt.subplot(2, 1, 1, xlim=[.95,5.15], ylim=[-.8,1.2])

plt.text(0.88,0.9,'a)',fontsize=16)
rect = patches.Rectangle((1.15,-.43),3.3,1.6, edgecolor='r', facecolor="none")
ax.add_patch(rect)
plt.axis('off')
plt.text(1.35,.85,'CNN',color='r',fontsize=18)

plt.scatter(np.full(1,1.), np.zeros(1), marker='s', color='w', s=120., alpha=1., lw=1., edgecolors=[.01,.01,.01])
plt.scatter(np.full(1,1.)+0.02, np.zeros(1)-0.08, marker='s', color='w', s=120., alpha=1., lw=1., edgecolors=[.01,.01,.01])
plt.text(.95,.25,'ERA5',fontsize=8)
plt.arrow(1.1,0.,.15,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')

plt.text(1.29,-.1,'Conv2D',fontsize=12)
rect = patches.Rectangle((1.27,-.25),.3,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(1.6,0.,.08,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(1.78,0.,'max',fontsize=10)
plt.text(1.75,-.18,'pooling',fontsize=10)
rect = patches.Rectangle((1.72,-.25),.26,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(2.02,0.,.12,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')

plt.text(2.19,-.1,'Conv2D',fontsize=12)
rect = patches.Rectangle((2.17,-.25),.3,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(2.5,0.,.08,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(2.68,0.,'max',fontsize=10)
plt.text(2.65,-.18,'pooling',fontsize=10)
rect = patches.Rectangle((2.62,-.25),.26,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(2.92,0.,.12,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')

plt.text(3.09,0.,'Hidden',fontsize=10)
plt.text(3.1,-0.18,'Layer',fontsize=10)
rect = patches.Rectangle((3.07,-.25),.24,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(3.34,0.,.12,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')

plt.text(3.59,0.,'Basis',fontsize=10)
plt.text(3.5,-0.18,'Coefficients',fontsize=10)
rect = patches.Rectangle((3.48,-.25),.36,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(3.87,0.,.12,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')

plt.text(3.97,.9,'Basis',fontsize=10)
plt.text(3.92,0.72,'Functions',fontsize=10)
rect = patches.Rectangle((3.9,.66),.3,.42, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(4.05,.56,0.,-.35, head_width=.02, head_length=0.08, length_includes_head=True, color='k')
plt.scatter(4.05,0.0, color='w', s=180, alpha=1., lw=1., edgecolors=[.01,.01,.01])
plt.scatter(4.05,0.0, color='k', s=6)
plt.arrow(4.11,0.,.12,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')

plt.text(4.27,0.,'Preliminary',fontsize=10)
plt.text(4.31,-0.18,'Output',fontsize=10)
rect = patches.Rectangle((4.25,-.25),.35,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)

plt.text(4.6,.9,'Log. Clim.',fontsize=10)
plt.text(4.55,0.72,'Probabilities',fontsize=10)
rect = patches.Rectangle((4.53,.66),.38,.42, edgecolor='k', facecolor="none")
ax.add_patch(rect)
plt.arrow(4.64,0.,.18,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.arrow(4.73,.56,0.1,-.5, head_width=.02, head_length=0.08, length_includes_head=True, color='k')

plt.text(4.89,-.1,'Output',fontsize=12)
rect = patches.Rectangle((4.87,-.25),.27,.45, edgecolor='k', facecolor="none")
ax.add_patch(rect)


plt.subplot(2, 1, 2, xlim=[0.8,3.55])
plt.text(0.76,3.,'b)',fontsize=16)

plt.scatter(np.full(2,1.), np.arange(0,4,3), marker='s', color='w', s=120., alpha=1., lw=1., edgecolors=[.01,.01,.01])
plt.scatter(np.full(2,1.)+0.02, np.arange(0,4,3)-0.08, marker='s', color='w', s=120., alpha=1., lw=1., edgecolors=[.01,.01,.01])
plt.scatter(np.full(3,1.), np.arange(1.2,2.6,0.5), color='k', s=10)
plt.axis('off')
plt.text(.92,3.35,'IFS m1',fontsize=8)
plt.text(.92,0.35,'IFS m11',fontsize=8)

plt.arrow(1.1,0.,.2,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(1.15,0.15,'CNN',fontsize=8,color='r')
plt.arrow(1.1,3.,.2,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(1.15,3.15,'CNN',fontsize=8,color='r')
plt.arrow(1.1,1.5,.2,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(1.15,1.65,'CNN',fontsize=8,color='r')

plt.text(1.35,3.,r'$x_{s,i}^1$')
plt.text(1.35,0.,r'$x_{s,i}^{11}$')
plt.scatter(np.full(3,1.4), np.arange(1.2,2.6,0.5), color='k', s=10)

plt.arrow(1.47,0.,.2,1.2, head_width=.025, head_length=0.08, length_includes_head=True, color='k')
plt.arrow(1.47,1.5,.2,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.arrow(1.47,3.,.2,-1.2, head_width=.025, head_length=0.08, length_includes_head=True, color='k')

plt.text(1.71,1.4,r'${\widebar x}_{s,i}$')
plt.arrow(1.85,1.5,.2,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(1.86,1.65,'relaxation',fontsize=8)

plt.text(2.1,1.4,r'$\eta\/{\widebar x}_{s,i}$')
plt.arrow(2.28,1.5,.2,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(2.12,3.,r'$log(p_{cl,s,i})$',fontsize=10)
plt.arrow(2.28,2.8,.2,-1.1, head_width=.025, head_length=0.08, length_includes_head=True, color='k')

plt.text(2.5,1.4,r'$z_{s,i}(\eta)$')
plt.arrow(2.7,1.5,.2,0., head_width=.08, head_length=0.02, length_includes_head=True, color='k')
plt.text(2.95,1.4,r'$p_{s,i}(\eta)$')

plt.tight_layout()





###################################################################################################
#                                                                                                 #
#  Figure 6:  Plots of scores for the discussion of tuning parameters                             #
#                                                                                                 #
###################################################################################################


x_wk2 = np.repeat(np.arange(20)[:,np.newaxis],5,axis=1)
x_wk3 = np.repeat(np.arange(22,42)[:,np.newaxis],5,axis=1)
x_wk4 = np.repeat(np.arange(44,64)[:,np.newaxis],5,axis=1)

crps_10cl_m0 = np.zeros((20,5,3),dtype=np.float32)
crps_20cl_m0 = np.zeros((20,5,3),dtype=np.float32)
crps_30cl_m0 = np.zeros((20,5,3),dtype=np.float32)
ccces_20cl_m0 = np.zeros((20,5,3),dtype=np.float32)
ccces_20cl_m1 = np.zeros((20,5,3),dtype=np.float32)
ccces_20cl_m2 = np.zeros((20,5,3),dtype=np.float32)


for ilead in range(3):
    clead = ['week2','week3','week4'][ilead]
    f1 = np.load("/home/michael/Desktop/CalifAPCP/tuning/efi-10cl-m0-l1_"+clead+".npz")
    crps_10cl_m0[:,:,ilead] = f1['opt_valid_crps']
    f1.close()
    f2 = np.load("/home/michael/Desktop/CalifAPCP/tuning/efi-20cl-m0-l1_"+clead+".npz")
    crps_20cl_m0[:,:,ilead] = f2['opt_valid_crps']
    f2.close()
    f3 = np.load("/home/michael/Desktop/CalifAPCP/tuning/efi-30cl-m0-l1_"+clead+".npz")
    crps_30cl_m0[:,:,ilead] = f3['opt_valid_crps']
    f3.close()
    f4 = np.load("/home/michael/Desktop/CalifAPCP/tuning/efi-20cl-m0-l1_"+clead+".npz")
    ccces_20cl_m0[:,:,ilead] = f4['opt_valid_scores']
    f4.close()
    f5 = np.load("/home/michael/Desktop/CalifAPCP/tuning/efi-20cl-m1-l1_"+clead+".npz")
    ccces_20cl_m1[:,:,ilead] = f5['opt_valid_scores']
    f5.close()
    f6 = np.load("/home/michael/Desktop/CalifAPCP/tuning/efi-20cl-m2-l1_"+clead+".npz")
    ccces_20cl_m2[:,:,ilead] = f6['opt_valid_scores']
    f6.close()


y1c_wk2 = 1.-np.sort(crps_10cl_m0[:,:,0]/crps_20cl_m0[:,:,0])
y2c_wk2 = 1.-np.sort(crps_30cl_m0[:,:,0]/crps_20cl_m0[:,:,0])

y1c_wk3 = 1.-np.sort(crps_10cl_m0[:,:,1]/crps_20cl_m0[:,:,1])
y2c_wk3 = 1.-np.sort(crps_30cl_m0[:,:,1]/crps_20cl_m0[:,:,1])

y1c_wk4 = 1.-np.sort(crps_10cl_m0[:,:,2]/crps_20cl_m0[:,:,2])
y2c_wk4 = 1.-np.sort(crps_30cl_m0[:,:,2]/crps_20cl_m0[:,:,2])

y1m_wk2 = 1.-np.sort(ccces_20cl_m1[:,:,0]/ccces_20cl_m0[:,:,0])
y2m_wk2 = 1.-np.sort(ccces_20cl_m2[:,:,0]/ccces_20cl_m0[:,:,0])

y1m_wk3 = 1.-np.sort(ccces_20cl_m1[:,:,1]/ccces_20cl_m0[:,:,1])
y2m_wk3 = 1.-np.sort(ccces_20cl_m2[:,:,1]/ccces_20cl_m0[:,:,1])

y1m_wk4 = 1.-np.sort(ccces_20cl_m1[:,:,2]/ccces_20cl_m0[:,:,2])
y2m_wk4 = 1.-np.sort(ccces_20cl_m2[:,:,2]/ccces_20cl_m0[:,:,2])



f1 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m0-drpt-f48.npz")
ccces_m0f48 = f1['opt_valid_scores']
f1.close()

f2 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m1-drpt-f48.npz")
ccces_m1f48 = f2['opt_valid_scores']
f2.close()

f3 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m2-drpt-f48.npz")
ccces_m2f48 = f3['opt_valid_scores']
f3.close()

f4 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m0-drpt-f44.npz")
ccces_m0f44 = f4['opt_valid_scores']
f4.close()

f5 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m0-drpt-f88.npz")
ccces_m0f88 = f5['opt_valid_scores']
f5.close()

f6 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m0-drpt-f816.npz")
ccces_m0f816 = f6['opt_valid_scores']
f6.close()

f7 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m0-l1-f48.npz")
ccces_m0f48_l1 = f7['opt_valid_scores']
f7.close()


y1m = 1.-np.sort(ccces_m1f48/ccces_m0f48)
y2m = 1.-np.sort(ccces_m2f48/ccces_m0f48)
y3r = 1.-np.sort(ccces_m0f48_l1/ccces_m0f48)

y1f = 1.-np.sort(ccces_m0f44/ccces_m0f48)
y2f = 1.-np.sort(ccces_m0f88/ccces_m0f48)
y3f = 1.-np.sort(ccces_m0f816/ccces_m0f48)



plt.figure(figsize=(16,12))

plt.subplot(3,2,1, ylim=[-0.0077,0.0077])
plt.scatter(x_wk2,y1c_wk2,c='orange',label='week-2')
plt.scatter(x_wk3,y1c_wk3,c='seagreen',label='week-3')
plt.scatter(x_wk4,y1c_wk4,c='r',label='week-4')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend(loc=(0.52,0.68),fontsize=12)
plt.title('CRPSS: 10 vs 20 categories, 1 x 10 nodes',fontsize=14)
plt.text(-1,0.0063,'a)',fontsize=16)
plt.axhline(y=0)
for i in range(20):
    plt.plot(x_wk2[i,::4],y1c_wk2[i,::4],c='orange')
    plt.plot(x_wk3[i,::4],y1c_wk3[i,::4],c='seagreen')
    plt.plot(x_wk4[i,::4],y1c_wk4[i,::4],c='r')

plt.subplot(3,2,2, ylim=[-0.0077,0.0077])
plt.scatter(x_wk2,y2c_wk2,c='orange',label='week-2')
plt.scatter(x_wk3,y2c_wk3,c='seagreen',label='week-3')
plt.scatter(x_wk4,y2c_wk4,c='r',label='week-4')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend(loc=(0.52,0.68),fontsize=12)
plt.title('CRPSS: 30 vs 20 categories, 1 x 10 nodes',fontsize=14)
plt.text(-1,0.0063,'b)',fontsize=16)
plt.axhline(y=0)
for i in range(20):
    plt.plot(x_wk2[i,::4],y2c_wk2[i,::4],c='orange')
    plt.plot(x_wk3[i,::4],y2c_wk3[i,::4],c='seagreen')
    plt.plot(x_wk4[i,::4],y2c_wk4[i,::4],c='r')

plt.subplot(3,2,3, ylim=[-0.0042,0.0042])
plt.scatter(x_wk2,y1m_wk2,c='orange',label='week-2')
plt.scatter(x_wk3,y1m_wk3,c='seagreen',label='week-3')
plt.scatter(x_wk4,y1m_wk4,c='r',label='week-4')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend(loc=(0.52,0.68),fontsize=12)
plt.title('MCCESS: 1 x 10 vs 1 x 20 nodes, 20 categories',fontsize=14)
plt.text(-1,0.0034,'c)',fontsize=16)
plt.axhline(y=0)
for i in range(20):
    plt.plot(x_wk2[i,::4],y1m_wk2[i,::4],c='orange')
    plt.plot(x_wk3[i,::4],y1m_wk3[i,::4],c='seagreen')
    plt.plot(x_wk4[i,::4],y1m_wk4[i,::4],c='r')

plt.subplot(3,2,4, ylim=[-0.0042,0.0042])
plt.scatter(x_wk2,y2m_wk2,c='orange',label='week-2')
plt.scatter(x_wk3,y2m_wk3,c='seagreen',label='week-3')
plt.scatter(x_wk4,y2m_wk4,c='r',label='week-4')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend(loc=(0.52,0.68),fontsize=12)
plt.title('MCCESS: 1 x 10 vs 2 x 10 nodes, 20 categories',fontsize=14)
plt.text(-1,0.0034,'d)',fontsize=16)
plt.axhline(y=0)
for i in range(20):
    plt.plot(x_wk2[i,::4],y2m_wk2[i,::4],c='orange')
    plt.plot(x_wk3[i,::4],y2m_wk3[i,::4],c='seagreen')
    plt.plot(x_wk4[i,::4],y2m_wk4[i,::4],c='r')

plt.subplot(3,2,5, ylim=[-0.042,0.042])
plt.scatter(x_wk2,y1m,c='royalblue',label='1 x 10 vs. 1 x 20 nodes, dropout')
plt.scatter(x_wk3,y2m,c='navy',label='1 x 10 vs. 2 x 10 nodes, dropout')
plt.scatter(x_wk4,y3r,c='darkturquoise',label='1 x 10 nodes, dropout vs. l1')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend(loc=(0.58,0.67),fontsize=12)
plt.title('MCCESS: CNN with 4/8 filters',fontsize=14)
plt.text(-1,0.034,'e)',fontsize=16)
plt.axhline(y=0)
for i in range(20):
    plt.plot(x_wk2[i,::4],y1m[i,::4],c='royalblue')
    plt.plot(x_wk3[i,::4],y2m[i,::4],c='navy')
    plt.plot(x_wk4[i,::4],y3r[i,::4],c='darkturquoise')

plt.subplot(3,2,6, ylim=[-0.023,0.023])
plt.scatter(x_wk2,y1f,c='blueviolet',label='4/8 vs. 4/4 filters')
plt.scatter(x_wk3,y2f,c='lightskyblue',label='4/8 vs. 8/8 filters')
plt.scatter(x_wk4,y3f,c='midnightblue',label='4/8 vs. 8/16 filters')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend(loc=(0.51,0.02),fontsize=12)
plt.title('MCCESS: CNN for 1 x 10 hidden nodes, dropout',fontsize=14)
plt.text(-1,0.018,'f)',fontsize=16)
plt.axhline(y=0)
for i in range(20):
    plt.plot(x_wk2[i,::4],y1f[i,::4],c='blueviolet')
    plt.plot(x_wk3[i,::4],y2f[i,::4],c='lightskyblue')
    plt.plot(x_wk4[i,::4],y3f[i,::4],c='midnightblue')

plt.tight_layout()







###################################################################################################
#                                                                                                 #
#  Figure 7:  Maps of RPSS (highlighting statistically significant grid points)                   #
#                                                                                                 #
###################################################################################################


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/precip_PRISM_cal_19810101_20171231.npz")
obs_lat = f1['lat']
obs_lon = f1['lon']
f1.close()

nxy = len(obs_lon)

ndts = 61
nyrs = 20


acfANN = np.zeros((3,15),dtype=np.float32)
acfCNN = np.zeros((3,15),dtype=np.float32)
pvalANN = np.zeros((3,nxy),dtype=np.float32)
pvalCNN = np.zeros((3,nxy),dtype=np.float32)
alphaFDR_ANN = np.zeros(3,dtype=np.float32)
alphaFDR_CNN = np.zeros(3,dtype=np.float32)

rpssMapANN = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)
rpssMapCSGD = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)
rpssMapCNN = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)

rpssAvgANN = ma.array(np.zeros(3,dtype=np.float32),mask=True)
rpssAvgCSGD = ma.array(np.zeros(3,dtype=np.float32),mask=True)
rpssAvgCNN = ma.array(np.zeros(3,dtype=np.float32),mask=True)

for ilead in range(3):
    f1 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-ann_week"+str(ilead+2)+".npz")
    Bs33Clm = f1['Bs33pClm']
    Bs33ANN = f1['Bs33pANN']
    Bs33CSGD = f1['Bs33pCSGD']
    Bs67Clm = f1['Bs67pClm']
    Bs67ANN = f1['Bs67pANN']
    Bs67CSGD = f1['Bs67pCSGD']
    Bs85Clm = f1['Bs85pClm']
    Bs85ANN = f1['Bs85pANN']
    Bs85CSGD = f1['Bs85pCSGD']
    f1.close()
    f2 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-cnn_week"+str(ilead+2)+".npz")
    Bs33CNN = f2['Bs33pCNN']
    Bs67CNN = f2['Bs67pCNN']
    Bs85CNN = f2['Bs85pCNN']
    f2.close()
    rpsClm = Bs33Clm + Bs67Clm + Bs85Clm       # calculate ranked probability score
    rpsANN = Bs33ANN + Bs67ANN + Bs85ANN
    rpsCSGD = Bs33CSGD + Bs67CSGD + Bs85CSGD
    rpsCNN = Bs33CNN + Bs67CNN + Bs85CNN
    rpssMapANN[ilead,:] = 1.-np.sum(rpsANN,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssMapCSGD[ilead,:] = 1.-np.sum(rpsCSGD,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssMapCNN[ilead,:] = 1.-np.sum(rpsCNN,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssAvgANN[ilead] = 1.-np.sum(rpsANN)/np.sum(rpsClm)
    rpssAvgCSGD[ilead] = 1.-np.sum(rpsCSGD)/np.sum(rpsClm)
    rpssAvgCNN[ilead] = 1.-np.sum(rpsCNN)/np.sum(rpsClm)
    rpsDiffANN = rpsCSGD-rpsANN
    rpsDiffCNN = rpsCSGD-rpsCNN
    rpsDiffStdzANN = (rpsDiffANN-np.mean(rpsDiffANN,axis=(0,1))[None,None,:])/np.std(rpsDiffANN,axis=(0,1))[None,None,:]
    rpsDiffStdzCNN = (rpsDiffCNN-np.mean(rpsDiffCNN,axis=(0,1))[None,None,:])/np.std(rpsDiffCNN,axis=(0,1))[None,None,:]
    for lg in range(15):
        acfANN[ilead,lg] = np.mean(rpsDiffStdzANN[lg:,:,:]*rpsDiffStdzANN[:(ndts-lg),:,:])         # Estimate temporal autocorrelation
        acfCNN[ilead,lg] = np.mean(rpsDiffStdzCNN[lg:,:,:]*rpsDiffStdzCNN[:(ndts-lg),:,:])
    rhoANN = acfANN[ilead,1]/acfANN[ilead,0]
    rhoCNN = acfCNN[ilead,1]/acfCNN[ilead,0]
    print(rhoANN,rhoCNN)
    nANN = round(ndts*nyrs*(1-rhoANN)/(1+rhoANN))
    nCNN = round(ndts*nyrs*(1-rhoCNN)/(1+rhoCNN))
    #print(nANN,nCNN)
    for ixy in range(nxy):
        smplANN = rpsCSGD[:,:,ixy].flatten()-rpsANN[:,:,ixy].flatten()
        smplCNN = rpsCSGD[:,:,ixy].flatten()-rpsCNN[:,:,ixy].flatten()
        tstatANN = np.mean(smplANN)/np.sqrt(np.var(smplANN)/nANN)        # test statistic for paired t-test
        tstatCNN = np.mean(smplCNN)/np.sqrt(np.var(smplCNN)/nCNN)
        pvalANN[ilead,ixy] = 1.-sp.stats.t.cdf(tstatANN,df=nANN-1)       # p-value for one-sided test
        pvalCNN[ilead,ixy] = 1.-sp.stats.t.cdf(tstatCNN,df=nCNN-1)
        #pval[ilead,ixy] = 2*min(1.-sp.stats.t.cdf(tstat,df=n-1),sp.stats.t.cdf(tstat,df=n-1))
    pvalANN_srt = np.sort(pvalANN[ilead,:])
    iANN = np.where(pvalANN_srt<=0.1*np.arange(1,nxy+1)/nxy)[0]
    if len(iANN)>0:
        alphaFDR_ANN[ilead] = pvalANN_srt[iANN[-1]]
    pvalCNN_srt = np.sort(pvalCNN[ilead,:])
    iCNN = np.where(pvalCNN_srt<=0.1*np.arange(1,nxy+1)/nxy)[0]
    if len(iCNN)>0:
        alphaFDR_CNN[ilead] = pvalCNN_srt[iCNN[-1]]
    plt.figure(); plt.scatter(np.arange(663),0.1*np.arange(1,664)/663); plt.scatter(np.arange(663),pvalANN_srt); plt.scatter(np.arange(663),pvalCNN_srt)



##  First figure depicts distribution of RPS differences and autocorrelation function

fig = plt.figure(figsize=(15,9))

for ilead in range(3):
    ax1 = fig.add_subplot(2,3,ilead+1)
    sp.stats.probplot(rpsDiffStdzANN.flatten(),plot=plt)
    plt.title("Q-Q Plot of RPS differences (week "+str(ilead+2)+")",fontsize=14)
    ax2 = fig.add_subplot(2,3,ilead+4)
    plt.scatter(np.arange(15),acfANN[ilead,:])
    plt.axhline(y=0)
    plt.axhline(y=0.05,ls='--')
    plt.axhline(y=-0.05,ls='--')
    plt.plot(np.arange(15),acfANN[ilead,1]**np.arange(15),c='red')
    plt.title("ACF of RPS differences (week "+str(ilead+2)+")",fontsize=14)

plt.tight_layout()


fig = plt.figure(figsize=(11.3,9.))

for ilead in range(3):
    ylim = np.array([0.26,0.052,0.026])[ilead]
    #ylim = np.amax(abs(rpssMapCSGD[ilead,:]))
    indSgnfANN = (pvalANN[ilead,:]<alphaFDR_ANN[ilead])
    indSgnfCNN = (pvalCNN[ilead,:]<alphaFDR_CNN[ilead])
    ax1 = fig.add_subplot(3,3,ilead+1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapCSGD[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgCSGD[ilead].round(3),fontsize=12)
    plt.title("RPSS - CSGD (week "+str(ilead+2)+")",fontsize=14)
    ax2 = fig.add_subplot(3,3,ilead+4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapANN[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.scatter(obs_lon[indSgnfANN],obs_lat[indSgnfANN],c=rpssMapANN[ilead,indSgnfANN],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=19.75,lw=0.8,edgecolors=[.2,.2,.2])
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgANN[ilead].round(3),fontsize=12)
    plt.title("RPSS - ANN (week "+str(ilead+2)+")",fontsize=14)
    ax3 = fig.add_subplot(3,3,ilead+7)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapCNN[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.scatter(obs_lon[indSgnfCNN],obs_lat[indSgnfCNN],c=rpssMapCNN[ilead,indSgnfCNN],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=19.75,lw=0.8,edgecolors=[.2,.2,.2])
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgCNN[ilead].round(3),fontsize=12)
    plt.title("RPSS - CNN (week "+str(ilead+2)+")",fontsize=14)

plt.tight_layout()







###################################################################################################
#                                                                                                 #
#  Figure 8:  Maps of EFI skill and RPSS change due to removing geographic predictors             #
#                                                                                                 #
###################################################################################################


taucmp = sequential_hcl("Purples 2").cmap(name = "Correlation Color Map")


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
obs_lat = f1['obs_lat']
obs_lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
apcp_obs_cat = f1['apcp_obs_cat']
f1.close()

ttt, nxy, ncat = apcp_obs_cat.shape


rpssDiffANN = np.zeros((3,nxy),dtype=np.float32)
tauEFI = np.zeros((3,nxy),dtype=np.float32)

for ilead in range(3):
    ylim = np.array([0.085,0.016,0.012])[ilead]
    f1 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-ann_week"+str(ilead+2)+".npz")
    Bs33Clm = f1['Bs33pClm']
    Bs67Clm = f1['Bs67pClm']
    Bs85Clm = f1['Bs85pClm']
    Bs33ANN = f1['Bs33pANN']
    Bs67ANN = f1['Bs67pANN']
    Bs85ANN = f1['Bs85pANN']
    f1.close()
    rpsClm = Bs33Clm + Bs67Clm + Bs85Clm
    rpsANN = Bs33ANN + Bs67ANN + Bs85ANN
    f2 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-ann_no-coords_week"+str(ilead+2)+".npz")
    Bs33ANN = f2['Bs33pANN']
    Bs67ANN = f2['Bs67pANN']
    Bs85ANN = f2['Bs85pANN']
    f2.close()
    rpsANNnc = Bs33ANN + Bs67ANN + Bs85ANN
    rpssDiffANN[ilead,:] = (np.mean(rpsANNnc,axis=(0,1))-np.mean(rpsANN,axis=(0,1))) / np.mean(rpsClm,axis=(0,1))


for ilead in range(3):
    apcp_efi_verif = np.zeros((61,20,nxy),dtype=np.float32)
    apcp_verif_cat = np.zeros((61,20,nxy),dtype=np.float32)
    for iyr in range(20):
        f2 = np.load("/home/michael/Desktop/CalifAPCP/stats/ensemble_stats_week"+str(ilead+2)+"_ANN_yr"+str(iyr)+".npz")
        apcp_obs_ind_verif = f2['apcp_obs_ind_verif']
        apcp_ens_pit_verif = f2['apcp_ens_pit_verif']
        f2.close()
        apcp_efi_verif[:,iyr,:] = -1.+(2./np.pi)*np.mean(np.arccos(1.-2.*apcp_ens_pit_verif),axis=2)
        apcp_verif_cat[:,iyr,:] = np.argmax(apcp_obs_cat[apcp_obs_ind_verif,:,:],axis=2)
    for ixy in range(nxy):
        tauEFI[ilead,ixy] = kendalltau(apcp_efi_verif[:,:,ixy].flatten(),apcp_verif_cat[:,:,ixy].flatten())[0]


fig = plt.figure(figsize=(12.,6.))

for ilead in range(3):
    yliml = np.array([0.24,0.11,0.055])[ilead]
    ylimu = np.array([0.46,0.21,0.165])[ilead]
    ylim = np.array([0.065,0.0108,0.0108])[ilead]
    ax1 = fig.add_subplot(2,3,ilead+1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=tauEFI[ilead,:],marker='s',cmap=taucmp,vmin=yliml,vmax=ylimu,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.title("Kendall's tau for EFI (week "+str(ilead+2)+")",fontsize=14)
    ax2 = fig.add_subplot(2,3,ilead+4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssDiffANN[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.title("RPSS difference (week "+str(ilead+2)+")",fontsize=14)

plt.tight_layout()







###################################################################################################
#                                                                                                 #
#  Figure 9:  Map of Z500 and TCW analyses for highest/lowest P(>85th pctl) at Eureka/San Diego   #
#                                                                                                 #
###################################################################################################


divcmp = diverging_hcl("Green-Brown",rev=True).cmap(name = "Diverging Color Map")

states_us = np.load('/home/michael/Desktop/CalifAPCP/data/states_us.npz',allow_pickle=True)['polygons'].tolist()
states_mexico = np.load('/home/michael/Desktop/CalifAPCP/data/states_mexico.npz',allow_pickle=True)['polygons'].tolist()

f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
lat = f1['obs_lat']
lon = f1['obs_lon']
f1.close()


inc = np.logical_and(lon==-124.125,lat==40.875)   # Eureka
isc = np.logical_and(lon==-117.125,lat==32.875)   # San Diego
lcns = [np.argmax(inc),np.argmax(isc)]

iyr = np.array([[19,7],[12,3]],dtype=np.int32)    # date and year index for lowest/highest P(>85th pctl)
idt = np.array([[3,4],[16,25]],dtype=np.int32)    #  at Eureka and San Diego, set manually here


##  Load ERA5 z500 and tcw fields, subset to 22 x 18 image

ixl = 10
ixu = -6
jyl = 6
jyu = -6

f2 = np.load("/home/michael/Desktop/CalifAPCP/data/z500_tcw_predictors_era5.npz")
era5_dates_ord = f2['dates_ord']
era5_lon = f2['longitude'][ixl:ixu]
era5_lat = f2['latitude'][jyl:jyu]
z500 = f2['z500_1wk'][:,:,jyl:jyu,ixl:ixu]
tcw = f2['tcw_1wk'][:,:,jyl:jyu,ixl:ixu]
f2.close()

ndts, nyrs, ny, nx = z500.shape


##  Normalize tcw to 10th/90th climatological percentiles at each grid point

tcw_q10 = np.percentile(tcw,10,axis=1)
tcw_q90 = np.percentile(tcw,90,axis=1)
tcw_q10_sm = np.zeros(tcw_q10.shape, dtype=np.float32)
tcw_q90_sm = np.zeros(tcw_q90.shape, dtype=np.float32)

X = np.ones((ndts,3), dtype=np.float32)                  # Fit harmonic function to annual cycle of tcw climatology
X[:,1] = np.sin(2.*np.pi*era5_dates_ord[:,0]/365.25)
X[:,2] = np.cos(2.*np.pi*era5_dates_ord[:,0]/365.25)

for ix in range(nx):
    for jy in range(ny):
        coef_q10 = solve(np.matmul(X.T,X),np.matmul(X.T,tcw_q10[:,jy,ix]))
        tcw_q10_sm[:,jy,ix] = np.matmul(X,coef_q10)
        coef_q90 = solve(np.matmul(X.T,X),np.matmul(X.T,tcw_q90[:,jy,ix]))
        tcw_q90_sm[:,jy,ix] = np.matmul(X,coef_q90)

tcw_ano = -1.+2.*(tcw-tcw_q10_sm[:,None,:,:])/(tcw_q90_sm-tcw_q10_sm)[:,None,:,:]


##  Normalize z500 to 1st/99th climatological percentiles across all grid points

z500_q01 = np.percentile(z500,1,axis=(1,2,3))
z500_q99 = np.percentile(z500,99,axis=(1,2,3))

coef_q01 = solve(np.matmul(X.T,X),np.matmul(X.T,z500_q01))
z500_q01_sm = np.matmul(X,coef_q01)
coef_q99 = solve(np.matmul(X.T,X),np.matmul(X.T,z500_q99))
z500_q99_sm = np.matmul(X,coef_q99)

z500_ano = -1.+2.*(z500-z500_q01_sm[:,None,None,None])/(z500_q99_sm-z500_q01_sm)[:,None,None,None]


##  Make plots

contour_levels_tcw = np.arange(-2.,2.25,0.25)
x, y = np.meshgrid(era5_lon,era5_lat)

title_str = ['Lowest P(>85th percentile) at Eureka','Highest  P(>85th percentile) at Eureka','Lowest  P(>85th percentile) at San Diego','Highest  P(>85th percentile) at San Diego']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.,6.5))
fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.95, hspace=0.15, wspace=0.05)
for ilc in range(2):
    for iwd in range(2):
        ax = axes.flat[2*ilc+(1-iwd)]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title_str[2*ilc+(1-iwd)])
        C1 = ax.contour(x,y,z500_ano[idt[iwd,ilc],iyr[iwd,ilc],:,:],linewidths=0.35,colors='k',zorder=2)
        C2 = ax.contourf(x,y,tcw_ano[idt[iwd,ilc],iyr[iwd,ilc],:,:],levels=contour_levels_tcw,alpha=1,cmap=divcmp,extend='both',zorder=1,corner_mask=True)
        plt.clabel(C1,fontsize=10,inline=1,fmt='%.2f')
        ax.scatter(lon[lcns[ilc]],lat[lcns[ilc]],c='red',marker='*',zorder=3)
        for k in range(len(states_us)):
            pathPolygon = path.Path(states_us[str(k)])
            ax.add_patch(patches.PathPatch(pathPolygon, facecolor='none', lw=1.))
        for k in range(len(states_mexico)):
            pathPolygon = path.Path(np.squeeze(states_mexico[k]))
            ax.add_patch(patches.PathPatch(pathPolygon, facecolor='none', lw=1.))

cbar = fig.colorbar(C2,ax=axes.ravel().tolist())
cbar.set_label('\n normalized TCW anomalies', fontsize=12)









###################################################################################################
#                                                                                                 #
#  Figure for presentations: Examples of resulting exceedance probabilities                       #
#                                                                                                 #
###################################################################################################


iyyyy = 2017
imm = 1
idd = 8

itnf = np.logical_and(obs_lon==-120.625,obs_lat==39.375)     # coordinates of our example grid point in Tahoe National Forest

f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
#list(f1)
obs_lat = f1['obs_lat']
obs_lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
pop_doy = f1['pop_doy']
thr_doy = f1['thr_doy']
qtev_doy = f1['qtev_doy']
obs_precip_week = f1['apcp_obs']
f1.close()

ntms, nxy = obs_precip_week.shape

for ivdate in range(ntms):
    if datetime.date.fromordinal(int(obs_dates_ord[ivdate])) == datetime.date(iyyyy,imm,idd):
        break


f2 = np.load("/home/michael/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates = f2['dates_ord']
f2.close()

ndts, nyrs, nlts = mod_dates.shape

iidate = np.zeros((3,2),dtype=np.int16)    # date and year index for selected date

for idt in range(ndts):
    for iyr in range(nyrs):
        for ilt in range(3):
            if datetime.date.fromordinal(int(mod_dates[idt,iyr,6+ilt*7])) == datetime.date(iyyyy,imm,idd):
                iidate[ilt,0] = idt
                iidate[ilt,1] = iyr


f3 = np.load("/home/michael/Desktop/CalifAPCP/stats/ensemble_stats_week2_ANN_yr"+str(iidate[0,1])+".npz")
doy_vdate = f3['doy_dts'][iidate[0,0]]
apcp_ens_pit = f3['apcp_ens_pit_verif'][iidate[0,0],:,:]
f3.close()


ilt = 0

f5 = np.load("/home/michael/Desktop/CalifAPCP/forecasts/ANN-efi/probfcst_10-l1_week"+str(ilt+2)+"_yr"+str(iidate[ilt,1])+".npz")
prob_fcst_cat = f5['prob_fcst_cat'][iidate[ilt,0],:,:]
f5.close()

prob_fcst_chf = -np.log(1.-np.cumsum(prob_fcst_cat,axis=1)[:,:(ncat-1)])

prob_clm_cat = np.concatenate((1.-pop_doy[doy_vdate,:,np.newaxis],np.repeat(pop_doy[doy_vdate,:,np.newaxis]/(ncat-1),ncat-1,axis=1)),axis=1)
prob_clm_chf = -np.log(1.-np.cumsum(prob_clm_cat,axis=1)[:,:(ncat-1)])

pot6in = np.zeros(nxy,dtype=np.float32)
pot85p = np.zeros(nxy,dtype=np.float32)
pot6in_cl = np.zeros(nxy,dtype=np.float32)
pot85p_cl = np.zeros(nxy,dtype=np.float32)

for ixy in range(nxy):
    itp_fct = interp1d(thr_doy[doy_vdate,ixy,:], prob_fcst_chf[ixy,:], kind='linear',fill_value='extrapolate')
    pot6in[ixy] = np.exp(-itp_fct(152.4))
    pot85p[ixy] = np.exp(-itp_fct(qtev_doy[doy_vdate,ixy,2]))
    itp_fct = interp1d(thr_doy[doy_vdate,ixy,:], prob_clm_chf[ixy,:], kind='linear',fill_value='extrapolate')
    pot6in_cl[ixy] = np.exp(-itp_fct(152.4))
    pot85p_cl[ixy] = np.exp(-itp_fct(qtev_doy[doy_vdate,ixy,2]))


plt.figure(figsize=(10,4))

plt.subplot(1, 2, 2, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
    xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
    yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
plt.scatter(obs_lon,obs_lat,c=pot6in,marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.0,vmax=0.64,edgecolors=[.2,.2,.2])
#plt.scatter(obs_lon,obs_lat,c=pot6in,marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.0,vmax=0.32,edgecolors=[.2,.2,.2])
cbar = plt.colorbar()
plt.title('      Probability for exceeding 6 inches of precipitation\n',fontsize=12)

plt.subplot(1, 2, 1, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
    xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
    yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
plt.scatter(obs_lon,obs_lat,c=pot85p,marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.,vmax=1.,edgecolors=[.2,.2,.2])
#plt.scatter(obs_lon,obs_lat,c=pot85p,marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.13,vmax=0.23,edgecolors=[.2,.2,.2])
cbar = plt.colorbar()
plt.title('      Probability for exceeding 85th climat. percentile\n',fontsize=12)

plt.tight_layout()


plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
    xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
    yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
plt.scatter(obs_lon,obs_lat,c=pot6in,marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.0,vmax=0.65,edgecolors=[.2,.2,.2])
cbar = plt.colorbar()
plt.title('      Probability for exceeding 6 inches of precipitation\n',fontsize=12)

plt.subplot(1, 2, 2, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
    xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
    yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
plt.scatter(obs_lon,obs_lat,c=np.log10(pot6in/pot6in_cl),marker='s',cmap=divcmp,s=28,lw=.1,vmin=-2.2,vmax=2.2,edgecolors=[.2,.2,.2])
cbar = plt.colorbar(ticks=[-2,-1,0,1,2])
cbar.ax.set_yticklabels(['0.01','0.1','1','10','100'])
plt.title('       Ratio of forecast probability to climat. probability\n',fontsize=12)

plt.tight_layout()


plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
    xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
    yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
plt.scatter(obs_lon,obs_lat,c=pot85p,marker='s',cmap=pcpcmp,s=28,lw=.1,vmin=0.0,vmax=1.0,edgecolors=[.2,.2,.2])
cbar = plt.colorbar()
plt.title('      Probability for exceeding 85th climat. percentile\n',fontsize=12)

plt.subplot(1, 2, 2, xlim=(-124.9,-113.8), ylim=(31.9,42.5), \
    xticks=[-124,-122,-120,-118,-116,-114], xticklabels=['-124'+'\u00b0','-122'+'\u00b0','-120'+'\u00b0','-118'+'\u00b0','-116'+'\u00b0','-114'+'\u00b0'], \
    yticks=[32,34,36,38,40,42], yticklabels=['32'+'\u00b0','34'+'\u00b0','36'+'\u00b0','38'+'\u00b0','40'+'\u00b0','42'+'\u00b0'])
plt.scatter(obs_lon,obs_lat,c=np.log10(pot85p/pot85p_cl),marker='s',cmap=divcmp,s=28,lw=.1,vmin=-0.7,vmax=0.7,edgecolors=[.2,.2,.2])
cbar = plt.colorbar(ticks=[np.log10(0.25),np.log10(0.5),0.,np.log10(2.),np.log10(4.)])
cbar.ax.set_yticklabels(['0.25','0.5','1','2','4'])
plt.title('       Ratio of forecast probability to climat. probability\n',fontsize=12)

plt.tight_layout()




###################################################################################################
#                                                                                                 #
#  Figure for presentations: Reliability diagrams                                                 #
#                                                                                                 #
###################################################################################################


#fct = 8
#p = 0.8
nmin = 50

#cat33u = np.arange(np.round(-fct*0.67**p),np.round(fct*0.33**p))
#cat67u = np.arange(np.round(-fct*0.33**p),np.round(fct*0.67**p))
#cat85u = np.arange(np.round(-fct*0.15**p),np.round(fct*0.85**p))

cat33u = np.arange(11)
cat67u = np.arange(11)
cat85u = np.arange(11)

x33 = ma.array(np.zeros((3,3,len(cat33u)),dtype=np.float32),mask=True)
x67 = ma.array(np.zeros((3,3,len(cat67u)),dtype=np.float32),mask=True)
x85 = ma.array(np.zeros((3,3,len(cat85u)),dtype=np.float32),mask=True)
y33 = ma.array(np.zeros((3,3,len(cat33u)),dtype=np.float32),mask=True)
y67 = ma.array(np.zeros((3,3,len(cat67u)),dtype=np.float32),mask=True)
y85 = ma.array(np.zeros((3,3,len(cat85u)),dtype=np.float32),mask=True)
freq33 = ma.array(np.zeros((3,3,len(cat33u)),dtype=np.float32),mask=True)
freq67 = ma.array(np.zeros((3,3,len(cat67u)),dtype=np.float32),mask=True)
freq85 = ma.array(np.zeros((3,3,len(cat85u)),dtype=np.float32),mask=True)

for ilead in range(3):
    f1 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-ann_week"+str(ilead+2)+".npz")
    exc33p = f1['exc33p']
    exc67p = f1['exc67p']
    exc85p = f1['exc85p']
    pot33pCSGD = f1['pot33pCSGD']
    pot67pCSGD = f1['pot67pCSGD']
    pot85pCSGD = f1['pot85pCSGD']
    pot33pANN = f1['pot33pANN']
    pot67pANN = f1['pot67pANN']
    pot85pANN = f1['pot85pANN']
    f1.close()
    cat33csgd = np.round(pot33pCSGD*10).flatten()
    cat67csgd = np.round(pot67pCSGD*10).flatten()
    cat85csgd = np.round(pot85pCSGD*10).flatten()
    cat33ann = np.round(pot33pANN*10).flatten()
    cat67ann = np.round(pot67pANN*10).flatten()
    cat85ann = np.round(pot85pANN*10).flatten()
    #cat33 = np.round(fct*np.sign(pot33pANN-0.67)*abs(pot33pANN-0.67)**p).flatten()
    #cat67 = np.round(fct*np.sign(pot67pANN-0.33)*abs(pot67pANN-0.33)**p).flatten()
    #cat85 = np.round(fct*np.sign(pot85pANN-0.15)*abs(pot85pANN-0.15)**p).flatten()
    f2 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-cnn_week"+str(ilead+2)+".npz")
    pot33pCNN = f2['pot33pCNN']
    pot67pCNN = f2['pot67pCNN']
    pot85pCNN = f2['pot85pCNN']
    f2.close()
    cat33cnn = np.round(pot33pCNN*10).flatten()
    cat67cnn = np.round(pot67pCNN*10).flatten()
    cat85cnn = np.round(pot85pCNN*10).flatten()
    for i in range(len(cat33u)):
        freq33[0,ilead,i] = np.sum(cat33csgd==cat33u[i])
        if freq33[0,ilead,i]>nmin:
            x33[0,ilead,i] = np.mean(pot33pCSGD.flatten()[cat33csgd==cat33u[i]])
            y33[0,ilead,i] = np.mean(exc33p.flatten()[cat33csgd==cat33u[i]])
        freq33[1,ilead,i] = np.sum(cat33ann==cat33u[i])
        if freq33[1,ilead,i]>nmin:
            x33[1,ilead,i] = np.mean(pot33pANN.flatten()[cat33ann==cat33u[i]])
            y33[1,ilead,i] = np.mean(exc33p.flatten()[cat33ann==cat33u[i]])
        freq33[2,ilead,i] = np.sum(cat33cnn==cat33u[i])
        if freq33[2,ilead,i]>nmin:
            x33[2,ilead,i] = np.mean(pot33pCNN.flatten()[cat33cnn==cat33u[i]])
            y33[2,ilead,i] = np.mean(exc33p.flatten()[cat33cnn==cat33u[i]])
        freq67[0,ilead,i] = np.sum(cat67csgd==cat67u[i])
        if freq67[0,ilead,i]>nmin:
            x67[0,ilead,i] = np.mean(pot67pCSGD.flatten()[cat67csgd==cat67u[i]])
            y67[0,ilead,i] = np.mean(exc67p.flatten()[cat67csgd==cat67u[i]])
        freq67[1,ilead,i] = np.sum(cat67ann==cat67u[i])
        if freq67[1,ilead,i]>nmin:
            x67[1,ilead,i] = np.mean(pot67pANN.flatten()[cat67ann==cat67u[i]])
            y67[1,ilead,i] = np.mean(exc67p.flatten()[cat67ann==cat67u[i]])
        freq67[2,ilead,i] = np.sum(cat67cnn==cat67u[i])
        if freq67[2,ilead,i]>nmin:
            x67[2,ilead,i] = np.mean(pot67pCNN.flatten()[cat67cnn==cat67u[i]])
            y67[2,ilead,i] = np.mean(exc67p.flatten()[cat67cnn==cat67u[i]])
        freq85[0,ilead,i] = np.sum(cat85cnn==cat85u[i])
        if freq85[0,ilead,i]>nmin:
            x85[0,ilead,i] = np.mean(pot85pCSGD.flatten()[cat85csgd==cat85u[i]])
            y85[0,ilead,i] = np.mean(exc85p.flatten()[cat85csgd==cat85u[i]])
        freq85[1,ilead,i] = np.sum(cat85ann==cat85u[i])
        if freq85[1,ilead,i]>nmin:
            x85[1,ilead,i] = np.mean(pot85pANN.flatten()[cat85ann==cat85u[i]])
            y85[1,ilead,i] = np.mean(exc85p.flatten()[cat85ann==cat85u[i]])
        freq85[2,ilead,i] = np.sum(cat85cnn==cat85u[i])
        if freq85[2,ilead,i]>nmin:
            x85[2,ilead,i] = np.mean(pot85pCNN.flatten()[cat85cnn==cat85u[i]])
            y85[2,ilead,i] = np.mean(exc85p.flatten()[cat85cnn==cat85u[i]])


fig = plt.figure(figsize=(14,9))

for ilt in range(3):
    ax1 = fig.add_subplot(2,3,1+ilt)
    relCSGD = plt.plot(x33[0,ilt,:],y33[0,ilt,:],'-o',c='blueviolet')
    relANN = plt.plot(x33[1,ilt,:],y33[1,ilt,:],'-o',c='royalblue')
    relCNN = plt.plot(x33[2,ilt,:],y33[2,ilt,:],'-o',c='indigo')
    plt.plot([0,1],[0,1],c='k')
    plt.axvline(0.67,c='k',ls=':',lw=1,ymin=0.05,ymax=0.95)
    plt.title("Reliability for P(> 33th pctl), week-"+str(ilt+2)+"\n",fontsize=14)
    plt.legend((relCSGD[0],relANN[0],relCNN[0]),('CSGD','ANN','CNN'),loc=4,fontsize=12)
    ins1 = ax1.inset_axes([0.03,0.68,0.4,0.3])
    ins1.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)
    ins1.set_xlabel('Frequency of usage',fontsize=11)
    ins1.bar(cat33u-0.25,freq33[0,ilt,:],0.23,color='blueviolet')
    ins1.bar(cat33u-0.0,freq33[1,ilt,:],0.23,color='royalblue')
    ins1.bar(cat33u+0.25,freq33[2,ilt,:],0.23,color='indigo')
    #ins1.axvline(0.0,c='k',ls=':',lw=1)
    ins1.axvline(6.67,c='k',ls=':',lw=1)
    ax2 = fig.add_subplot(2,3,4+ilt)
    relCSGD = plt.plot(x67[0,ilt,:],y67[0,ilt,:],'-o',c='blueviolet')
    relANN = plt.plot(x67[1,ilt,:],y67[1,ilt,:],'-o',c='royalblue')
    relCNN = plt.plot(x67[2,ilt,:],y67[2,ilt,:],'-o',c='indigo')
    plt.plot([0,1],[0,1],c='k')
    plt.axvline(0.33,c='k',ls=':',lw=1,ymin=0.05,ymax=0.5)
    plt.title("Reliability for P(> 67th pctl), week-"+str(ilt+2)+"\n",fontsize=14)
    plt.legend((relCSGD[0],relANN[0],relCNN[0]),('CSGD','ANN','CNN'),loc=4,fontsize=12)
    ins2 = ax2.inset_axes([0.03,0.68,0.4,0.3])
    ins2.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)
    ins2.set_xlabel('Frequency of usage',fontsize=11)
    ins2.bar(cat67u-0.25,freq67[0,ilt,:],0.23,color='blueviolet')
    ins2.bar(cat67u-0.0,freq67[1,ilt,:],0.23,color='royalblue')
    ins2.bar(cat67u+0.25,freq67[2,ilt,:],0.23,color='indigo')
    #ins2.axvline(0.0,c='k',ls=':',lw=1)
    ins2.axvline(3.33,c='k',ls=':',lw=1)

plt.tight_layout()




