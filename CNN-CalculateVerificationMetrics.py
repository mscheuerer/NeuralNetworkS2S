
import numpy as np
import numpy.ma as ma
import scipy.stats as stats
import math
import os, sys
import matplotlib.pyplot as plt
import datetime
import time
import pdb
import pickle

from scipy.stats import gamma
from scipy.interpolate import interp1d

plt.ion()



leadDay = 20         # d works out to being a d+0.5 day forecast
accumulation = 7     # Precipitation accumulation period

clead = 'week'+str((leadDay+8)//7)


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
#list(f1)
obs_lat = f1['obs_lat']
obs_lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
thr_doy = f1['thr_doy']
qtev_doy = f1['qtev_doy']
apcp_obs_cat = f1['apcp_obs_cat']
obs_precip_week = f1['apcp_obs']
f1.close()

ndts, nxy, ncat = apcp_obs_cat.shape


#f2 = np.load("/home/michael/Desktop/CalifAPCP/data/z500_tcw_predictors_era5.npz")
#mod_dates = f2['dates_ord']
#f2.close()

f2 = np.load("/home/michael/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates = f2['dates_ord'][:,:,leadDay]
f2.close()

ndts, nyrs = mod_dates.shape


doy_dts = np.zeros(ndts,dtype=np.int32)
obs_precip_vdate = np.zeros((ndts,nyrs,nxy),dtype=np.float32)
for idt in range(ndts):
    for iyr in range(nyrs):
        fnd = np.nonzero(obs_dates_ord==mod_dates[idt,iyr])[0][0]
        obs_precip_vdate[idt,iyr,:] = obs_precip_week[fnd,:]
    date_ord = int(mod_dates[idt,-1]-0.5)
    doy_dts[idt] = min(364,(datetime.date.fromordinal(date_ord)-datetime.date(datetime.date.fromordinal(date_ord).year,1,1)).days)


### Calculate skill scores

exc33p = np.zeros(obs_precip_vdate.shape)
brier33pClm = np.zeros(obs_precip_vdate.shape)
pot33pCNN = np.zeros(obs_precip_vdate.shape)
brier33pCNN = np.zeros(obs_precip_vdate.shape)

exc67p = np.zeros(obs_precip_vdate.shape)
brier67pClm = np.zeros(obs_precip_vdate.shape)
pot67pCNN = np.zeros(obs_precip_vdate.shape)
brier67pCNN = np.zeros(obs_precip_vdate.shape)

exc85p = np.zeros(obs_precip_vdate.shape)
brier85pClm = np.zeros(obs_precip_vdate.shape)
pot85pCNN = np.zeros(obs_precip_vdate.shape)
brier85pCNN = np.zeros(obs_precip_vdate.shape)

rpsClm = np.zeros(obs_precip_vdate.shape)
rpsCNN = np.zeros(obs_precip_vdate.shape)

crpsClm = np.zeros(obs_precip_vdate.shape)
crpsCNN = np.zeros(obs_precip_vdate.shape)


wwCl = 15

x = (np.arange(0,101)/5)**2      # evaluation points for numerical approximation of the CRPS
dx = np.diff(x)


imod = 0

for iyr in range(nyrs):
    print(iyr)
    #f5 = np.load("/home/michael/Desktop/CalifAPCP/forecasts/CNN/probfcst_cnn-m"+str(imod)+"-drpt-f48_yr"+str(iyr)+".npz")
    #prob_fcst_cat = f5['prob_fcst_cat_era5']
    #f5.close()
    f5 = np.load("/home/michael/Desktop/CalifAPCP/forecasts/CNN/probfcst_cnn-m"+str(imod)+"-drpt-f48_"+clead+"_yr"+str(iyr)+".npz")
    prob_fcst_cat = f5['prob_fcst_cat']
    f5.close()
    prob_fcst_chf = -np.log(1.-np.cumsum(prob_fcst_cat,axis=2)[:,:,:(ncat-1)])
    prob_over_thr = np.zeros((ndts,nxy,qtev_doy.shape[2]),dtype=np.float32)
    for idt in range(ndts):
        windowClm = np.argsort(np.abs(idt-np.arange(ndts)))[:wwCl]
        ### Calculate exceedance ANN probabilities from interpolated cumulative hazard function
        for ixy in range(nxy):
            itp_fct = interp1d(thr_doy[doy_dts[idt],ixy,:], prob_fcst_chf[idt,ixy,:], kind='linear',fill_value='extrapolate')
            prob_over_thr = np.exp(-itp_fct(qtev_doy[doy_dts[idt],ixy,:]))
            pot33pCNN[idt,iyr,ixy] = prob_over_thr[0]
            pot67pCNN[idt,iyr,ixy] = prob_over_thr[1]
            pot85pCNN[idt,iyr,ixy] = prob_over_thr[2]
            ## Calculate CRPS for CNN
            bs = (1.-np.exp(-itp_fct(x))-1.*(obs_precip_vdate[idt,iyr,ixy]<=x))**2
            crpsCNN[idt,iyr,ixy] = 0.5*np.sum((bs[1:]+bs[:len(dx)])*dx)
        ### Get current year and julian day to use to select climatological percentiles
        currentYear = datetime.date.fromordinal(int(mod_dates[idt,iyr])).year
        currentDay = (datetime.date.fromordinal(int(mod_dates[idt,iyr]))-datetime.date(currentYear,1,1)).days
        obsClm = obs_precip_vdate[windowClm,:,:].reshape((wwCl*nyrs,nxy))
        crps_exc = 1.*np.less_equal.outer(obs_precip_vdate[idt,iyr,:],x)
        ## Calculate CRPS for Clm
        clm_cdf = np.mean(obsClm[:,:,None]<=x[None,None,:],axis=0)
        bs = (clm_cdf-crps_exc)**2
        crpsClm[idt,iyr,:] = 0.5*np.sum((bs[:,1:]+bs[:,:len(dx)])*dx[None,:],axis=1)
        ## Calculate Brier scores for different thresholds
        p33 = qtev_doy[doy_dts[idt],:,0]
        exc33p[idt,iyr,:] = (obs_precip_vdate[idt,iyr,:]>p33)
        brier33pClm[idt,iyr,:] = (exc33p[idt,iyr,:]-np.mean(obsClm>p33[None,:],axis=0))**2
        brier33pCNN[idt,iyr,:] = (exc33p[idt,iyr,:]-pot33pCNN[idt,iyr,:])**2
        p67 = qtev_doy[doy_dts[idt],:,1]
        exc67p[idt,iyr,:] = (obs_precip_vdate[idt,iyr,:]>p67)
        brier67pClm[idt,iyr,:] = (exc67p[idt,iyr,:]-np.mean(obsClm>p67[None,:],axis=0))**2
        brier67pCNN[idt,iyr,:] = (exc67p[idt,iyr,:]-pot67pCNN[idt,iyr,:])**2
        p85 = qtev_doy[doy_dts[idt],:,2]
        exc85p[idt,iyr,:] = (obs_precip_vdate[idt,iyr,:]>p85)
        brier85pClm[idt,iyr,:] = (exc85p[idt,iyr,:]-np.mean(obsClm>p85[None,:],axis=0))**2
        brier85pCNN[idt,iyr,:] = (exc85p[idt,iyr,:]-pot85pCNN[idt,iyr,:])**2


outfilename = "/home/michael/Desktop/CalifAPCP/results/scores-cnn_"+clead
np.savez(outfilename, crpsClm=crpsClm, crpsCNN=crpsCNN, \
     exc33p=exc33p, pot33pCNN=pot33pCNN, Bs33pClm=brier33pClm, Bs33pCNN=brier33pCNN, \
     exc67p=exc67p, pot67pCNN=pot67pCNN, Bs67pClm=brier67pClm, Bs67pCNN=brier67pCNN, \
     exc85p=exc85p, pot85pCNN=pot85pCNN, Bs85pClm=brier85pClm, Bs85pCNN=brier85pCNN)



# calculate ranked probability score
rpsClm = brier33pClm + brier67pClm + brier85pClm
rpsCNN = brier33pCNN + brier67pCNN + brier85pCNN


# rpssAvgCNN
round(1.-np.sum(rpsCNN)/np.sum(rpsClm),4)


