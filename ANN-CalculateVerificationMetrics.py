
import numpy as np
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


##  Load PRISM data

f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
#list(f1)
obs_lat = f1['obs_lat']
obs_lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
pop_doy = f1['pop_doy']
thr_doy = f1['thr_doy']
qtev_doy = f1['qtev_doy']
apcp_obs_cat = f1['apcp_obs_cat']
obs_precip_week = f1['apcp_obs']
f1.close()

ndts, nxy, ncat = apcp_obs_cat.shape


##  Load IFS ensemble forecasts

f2 = np.load("/home/michael/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates = f2['dates_ord'][:,:,leadDay]
mod_lon = f2['lon']
mod_lat = f2['lat']
mod_precip = np.sum(f2['precip'][:,:,leadDay:leadDay+accumulation,:,:],axis=2)
f2.close()

ndts, nyrs = mod_dates.shape

ecmwf_ind = np.zeros(len(mod_lon),dtype=np.bool)

for ixy in range(nxy):
    dst = abs(mod_lon-obs_lon[ixy]) + abs(mod_lat-obs_lat[ixy])    # Nearest neighbor interpolation
    ecmwf_ind[np.argmin(dst)] = True

ecmwf_precip = mod_precip[:,:,:,ecmwf_ind]


obs_precip_vdate = np.zeros((ndts,nyrs,nxy),dtype=np.float32)
for idt in range(ndts):
    for iyr in range(nyrs):
        fnd = np.nonzero(obs_dates_ord==mod_dates[idt,iyr])[0][0]
        obs_precip_vdate[idt,iyr,:] = obs_precip_week[fnd,:]         # PRISM data on the verification days



### Calculate skill scores

exc33p = np.zeros(obs_precip_vdate.shape)
brier33pClm = np.zeros(obs_precip_vdate.shape)
pot33pANN = np.zeros(obs_precip_vdate.shape)
brier33pANN = np.zeros(obs_precip_vdate.shape)
pot33pCSGD = np.zeros(obs_precip_vdate.shape)
brier33pCSGD = np.zeros(obs_precip_vdate.shape)
pot33pENS = np.zeros(obs_precip_vdate.shape)
brier33pENS = np.zeros(obs_precip_vdate.shape)

exc67p = np.zeros(obs_precip_vdate.shape)
brier67pClm = np.zeros(obs_precip_vdate.shape)
pot67pANN = np.zeros(obs_precip_vdate.shape)
brier67pANN = np.zeros(obs_precip_vdate.shape)
pot67pCSGD = np.zeros(obs_precip_vdate.shape)
brier67pCSGD = np.zeros(obs_precip_vdate.shape)
pot67pENS = np.zeros(obs_precip_vdate.shape)
brier67pENS = np.zeros(obs_precip_vdate.shape)

exc85p = np.zeros(obs_precip_vdate.shape)
brier85pClm = np.zeros(obs_precip_vdate.shape)
pot85pANN = np.zeros(obs_precip_vdate.shape)
brier85pANN = np.zeros(obs_precip_vdate.shape)
pot85pCSGD = np.zeros(obs_precip_vdate.shape)
brier85pCSGD = np.zeros(obs_precip_vdate.shape)
pot85pENS = np.zeros(obs_precip_vdate.shape)
brier85pENS = np.zeros(obs_precip_vdate.shape)

rpsClm = np.zeros(obs_precip_vdate.shape)
rpsANN = np.zeros(obs_precip_vdate.shape)
rpsCSGD = np.zeros(obs_precip_vdate.shape)
rpsENS = np.zeros(obs_precip_vdate.shape)

crpsClm = np.zeros(obs_precip_vdate.shape)
crpsANN = np.zeros(obs_precip_vdate.shape)
crpsCSGD = np.zeros(obs_precip_vdate.shape)


f3 = np.load("/home/michael/Desktop/CalifAPCP/forecasts/csgd_fcsts_params_"+clead+".npz")
csgd_pars_fcst = f3['csgd_pars_fcst']
f3.close()


wwCl = 15   # number of dates around the date of interest used to create a sample representing climatology 


x = (np.arange(0,101)/5)**2      # evaluation points for numerical approximation of the CRPS
dx = np.diff(x)

for iyr in range(nyrs):
    print(iyr)
    f4 = np.load("/home/michael/Desktop/CalifAPCP/stats/ensemble_stats_"+clead+"_ANN_yr"+str(iyr)+".npz")
    doy_dts = f4['doy_dts']
    f4.close()
    f5 = np.load("/home/michael/Desktop/CalifAPCP/forecasts/ANN-efi/probfcst_10-l1_"+clead+"_yr"+str(iyr)+".npz")
    prob_fcst_cat = f5['prob_fcst_cat']
    f5.close()
    prob_fcst_chf = -np.log(1.-np.cumsum(prob_fcst_cat,axis=2)[:,:,:(ncat-1)])
    prob_over_thr = np.zeros((ndts,nxy,qtev_doy.shape[2]),dtype=np.float32)
    for idt in range(ndts):
        ### Calculate exceedance ANN probabilities from interpolated cumulative hazard function
        for ixy in range(nxy):
            itp_fct = interp1d(thr_doy[doy_dts[idt],ixy,:], prob_fcst_chf[idt,ixy,:], kind='linear',fill_value='extrapolate')
            prob_over_thr = np.exp(-itp_fct(qtev_doy[doy_dts[idt],ixy,:]))
            pot33pANN[idt,iyr,ixy] = prob_over_thr[0]
            pot67pANN[idt,iyr,ixy] = prob_over_thr[1]
            pot85pANN[idt,iyr,ixy] = prob_over_thr[2]
            ## Calculate CRPS for ANN
            bs = (1.-np.exp(-itp_fct(x))-1.*(obs_precip_vdate[idt,iyr,ixy]<=x))**2
            crpsANN[idt,iyr,ixy] = 0.5*np.sum((bs[1:]+bs[:len(dx)])*dx)
        ### Calculate climatological exceedances based on analyzed values within a time window around the forecast date
        windowClm = np.argsort(np.abs(idt-np.arange(ndts)))[:wwCl]
        obsClm = obs_precip_vdate[windowClm,:,:].reshape((wwCl*nyrs,nxy))
        ### Calculate threshold exceedances for the Brier scores used to approximate the CRPS
        crps_exc = 1.*np.less_equal.outer(obs_precip_vdate[idt,iyr,:],x)
        ### Compose sample to represent ECMWF model climatology
        modClm = ecmwf_precip[windowClm,:,:,:].reshape((wwCl*nyrs*11,nxy))
        ## Calculate CRPS for Clm
        clm_cdf = np.mean(obsClm[:,:,None]<=x[None,None,:],axis=0)
        bs = (clm_cdf-crps_exc)**2
        crpsClm[idt,iyr,:] = 0.5*np.sum((bs[:,1:]+bs[:,:len(dx)])*dx[None,:],axis=1)
        ## Calculate CRPS for CSGD
        shape = np.square(csgd_pars_fcst[idt,iyr,:,0]/csgd_pars_fcst[idt,iyr,:,1])
        scale = np.square(csgd_pars_fcst[idt,iyr,:,1])/csgd_pars_fcst[idt,iyr,:,0]
        shift = csgd_pars_fcst[idt,iyr,:,2]
        csgd_cdf = gamma.cdf((x[None,:]-shift[:,None])/scale[:,None],shape[:,None])
        bs = (csgd_cdf-crps_exc)**2
        crpsCSGD[idt,iyr,:] = 0.5*np.sum((bs[:,1:]+bs[:,:len(dx)])*dx[None,:],axis=1)
        ## Calculate Brier scores for different thresholds
        p33 = qtev_doy[doy_dts[idt],:,0]
        exc33p[idt,iyr,:] = (obs_precip_vdate[idt,iyr,:]>p33)
        brier33pClm[idt,iyr,:] = (exc33p[idt,iyr,:]-np.mean(obsClm>p33[None,:],axis=0))**2
        brier33pANN[idt,iyr,:] = (exc33p[idt,iyr,:]-pot33pANN[idt,iyr,:])**2
        pot33pCSGD[idt,iyr,:] = 1.-gamma.cdf((p33-shift)/scale,shape)
        brier33pCSGD[idt,iyr,:] = (exc33p[idt,iyr,:]-pot33pCSGD[idt,iyr,:])**2
        p33mod = np.maximum(0.254,np.quantile(modClm,0.333,axis=0))
        #pot33pENS[idt,iyr,:] = np.mean(ecmwf_precip[idt,iyr,:,:]>p33mod[None,:],axis=0)
        pot33pENS[idt,iyr,:] = np.mean(ecmwf_precip[idt,iyr,:,:]>p33[None,:],axis=0)
        brier33pENS[idt,iyr,:] = (exc33p[idt,iyr,:]-pot33pENS[idt,iyr,:])**2
        p67 = qtev_doy[doy_dts[idt],:,1]
        exc67p[idt,iyr,:] = (obs_precip_vdate[idt,iyr,:]>p67)
        brier67pClm[idt,iyr,:] = (exc67p[idt,iyr,:]-np.mean(obsClm>p67[None,:],axis=0))**2
        brier67pANN[idt,iyr,:] = (exc67p[idt,iyr,:]-pot67pANN[idt,iyr,:])**2
        pot67pCSGD[idt,iyr,:] = 1.-gamma.cdf((p67-shift)/scale,shape)
        brier67pCSGD[idt,iyr,:] = (exc67p[idt,iyr,:]-pot67pCSGD[idt,iyr,:])**2
        p67mod = np.maximum(0.254,np.quantile(modClm,0.667,axis=0))
        #pot67pENS[idt,iyr,:] = np.mean(ecmwf_precip[idt,iyr,:,:]>p67mod[None,:],axis=0)
        pot67pENS[idt,iyr,:] = np.mean(ecmwf_precip[idt,iyr,:,:]>p67[None,:],axis=0)
        brier67pENS[idt,iyr,:] = (exc67p[idt,iyr,:]-pot67pENS[idt,iyr,:])**2
        p85 = qtev_doy[doy_dts[idt],:,2]
        exc85p[idt,iyr,:] = (obs_precip_vdate[idt,iyr,:]>p85)
        brier85pClm[idt,iyr,:] = (exc85p[idt,iyr,:]-np.mean(obsClm>p85[None,:],axis=0))**2
        brier85pANN[idt,iyr,:] = (exc85p[idt,iyr,:]-pot85pANN[idt,iyr,:])**2
        pot85pCSGD[idt,iyr,:] = 1.-gamma.cdf((p85-shift)/scale,shape)
        brier85pCSGD[idt,iyr,:] = (exc85p[idt,iyr,:]-pot85pCSGD[idt,iyr,:])**2
        p85mod = np.maximum(0.254,np.quantile(modClm,0.85,axis=0))
        #pot85pENS[idt,iyr,:] = np.mean(ecmwf_precip[idt,iyr,:,:]>p85mod[None,:],axis=0)
        pot85pENS[idt,iyr,:] = np.mean(ecmwf_precip[idt,iyr,:,:]>p85[None,:],axis=0)
        brier85pENS[idt,iyr,:] = (exc85p[idt,iyr,:]-pot85pENS[idt,iyr,:])**2


outfilename = "/home/michael/Desktop/CalifAPCP/results/scores-ann_"+clead
np.savez(outfilename, crpsClm=crpsClm, crpsANN=crpsANN, crpsCSGD=crpsCSGD, \
     exc33p=exc33p, pot33pANN=pot33pANN, pot33pCSGD=pot33pCSGD, pot33pENS=pot33pENS, Bs33pClm=brier33pClm, Bs33pANN=brier33pANN, Bs33pCSGD=brier33pCSGD, Bs33pENS=brier33pENS, \
     exc67p=exc67p, pot67pANN=pot67pANN, pot67pCSGD=pot67pCSGD, pot67pENS=pot67pENS, Bs67pClm=brier67pClm, Bs67pANN=brier67pANN, Bs67pCSGD=brier67pCSGD, Bs67pENS=brier67pENS, \
     exc85p=exc85p, pot85pANN=pot85pANN, pot85pCSGD=pot85pCSGD, pot85pENS=pot85pENS, Bs85pClm=brier85pClm, Bs85pANN=brier85pANN, Bs85pCSGD=brier85pCSGD, Bs85pENS=brier85pENS)





# calculate ranked probability score
rpsClm = brier33pClm + brier67pClm + brier85pClm
rpsANN = brier33pANN + brier67pANN + brier85pANN
rpsCSGD = brier33pCSGD + brier67pCSGD + brier85pCSGD
rpsENS = brier33pENS + brier67pENS + brier85pENS

# rpssAvgANN
round(1.-np.sum(rpsANN)/np.sum(rpsClm),4)

# rpssAvgCSGD
round(1.-np.sum(rpsCSGD)/np.sum(rpsClm),4)

# rpssAvgENS
round(1.-np.sum(rpsENS)/np.sum(rpsClm),4)



