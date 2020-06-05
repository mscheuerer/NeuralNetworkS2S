
import numpy as np
import scipy as sp
import math
import os, sys
import datetime
import time
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from numpy import ma
from numpy.random import random_sample
from numpy.linalg import solve
from scipy import stats
from scipy.stats import kendalltau
from scipy.stats import gamma
from scipy.special import beta
from scipy.optimize import minimize
from scipy.interpolate import *



#plt.ion()

r = 300.   # neighborhood radius (kilometers)
R = 6373.   # earth radius (kilometers)


leadDay = 20       # Start of the forecast period
accumulation = 7  # Precipitation accumulation period


f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/precip_PRISM_cal_19810101_20171231.npz")
#list(f1)
obs_precip = f1['precip']
obs_lat = f1['lat']
obs_lon = f1['lon']
obs_dates_ord = f1['dates_ord']
obs_dates = f1['dates']
f1.close()

ndays, nxy = obs_precip.shape

obs_precip_week = np.zeros((ndays-6,nxy), dtype=np.float32)
for iday in range(7):
    obs_precip_week += obs_precip[iday:(ndays-6+iday),:]

nwks, nxy = obs_precip_week.shape

obs_precip_week[obs_precip_week<0.254] = 0.
obs_dates_ord = obs_dates_ord[:nwks]
obs_dates = obs_dates[:nwks]



f2 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/precip_climatological_csgd.npz")
pop_cl = f2['pop_cl_doy']
mean_cl = f2['mean_cl_doy']
shape_cl = f2['shape_cl_doy']
scale_cl = f2['scale_cl_doy']
shift_cl = f2['shift_cl_doy']
f2.close()



f3 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/mod_precip_calplus.npz")
### Modeled precip is (reforecast time, member, year, lead time, lat, lon)
mod_precip = f3['precip']
#mod_dates_ord = f3['datesOrd']
mod_lon = f3['lon']
mod_lat = f3['lat']
f3.close()

f3 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates_ord = f3['dates_ord']                                                 # Need to load dates from other file since dates in 'mod_precip_calplus.npz' are incorrect
f3.close()

ndts, nmem, nyrs, nlts, nlat, nlon = mod_precip.shape

### Modeled precip accumulated over forecast period (reforecast time, year, ensembles, space)
mod_precip_fcstperiod = np.sum(mod_precip[:,:,:,leadDay:leadDay+accumulation,:,:],axis=3).reshape((ndts,nmem,nyrs,nlon*nlat))
mod_dates_fcstperiod = mod_dates_ord[:,:,leadDay]



### Calculate day of the year ('doy') for each reforecast date
doy = np.zeros(ndts,dtype=np.int32)
for idt in range(ndts):
    yyyy = datetime.date.fromordinal(int(mod_dates_fcstperiod[idt,0])).year
    doy[idt] = (datetime.date.fromordinal(int(mod_dates_fcstperiod[idt,0]))-datetime.date(yyyy,1,1)).days


### Calculate spatially smoothed ensemble foreasts at analysis grid locations
mod_precip_fcstperiod_sm = np.zeros((ndts,nmem,nyrs,nxy),dtype=np.float32)
for ixy in range(nxy):
    lat1 = np.deg2rad(obs_lat[ixy])
    lon1 = np.deg2rad(obs_lon[ixy])
    lat2 = np.deg2rad(mod_lat)
    lon2 = np.deg2rad(mod_lon)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2)**2)[:,None] + np.cos(lat1) * np.outer(np.cos(lat2),np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    gcdst = (R*c).reshape(nlat*nlon)                    # great circle distances between forecast and analysis grid points
    uselocs = np.nonzero(gcdst<r)[0]
    wgt = (1.-(gcdst[uselocs]/r)**2) / sum(1.-(gcdst[uselocs]/r)**2)
    mod_precip_fcstperiod_sm[:,:,:,ixy] = np.average(mod_precip_fcstperiod[:,:,:,uselocs],axis=3,weights=wgt)



### Calculate ensemble mean and l-scale
## (note: we ended up not using l-scale as a predictor for uncertainty)

def l_scale(ensfcst,axis):
    # L-scale of n ensemble forecasts with m member
    shp = ensfcst.shape
    m = shp[axis]
    bcst = np.repeat(1,len(shp))
    bcst[axis] = m
    term1 = 2.*np.sum(np.sort(ensfcst,axis=axis)*np.arange(1,m+1).reshape((tuple(bcst))),axis=axis)/(m**2)
    term2 = (m+1)*np.mean(ensfcst,axis=axis)/m
    return term1-term2

ensmean_precip = np.mean(mod_precip_fcstperiod_sm,axis=1)
ensmeandiff_precip = 2.*l_scale(mod_precip_fcstperiod_sm,axis=1)


def crpsCondCSGD(par,obs,ensmeanano,muc,sigmac,shiftc):
    # average CRPS for CSGD conditional on the ensemble statistics
    logarg = par[1] + par[2]*ensmeanano
    mu = muc * np.log1p(np.expm1(par[0])*logarg) / par[0]
    sigma = sigmac * (par[3]*np.sqrt(mu/muc))
    shape = np.square(mu/sigma)
    scale = np.square(sigma)/mu
    shift = shiftc
    betaf = beta(0.5,shape+0.5)
    cstd = (0.254-shift)/scale
    ystd = np.maximum(obs-shift,0.0)/scale
    Fyk = sp.stats.gamma.cdf(ystd,shape,scale=1)
    Fck = sp.stats.gamma.cdf(cstd,shape,scale=1)
    FykP1 = sp.stats.gamma.cdf(ystd,shape+1,scale=1)
    FckP1 = sp.stats.gamma.cdf(cstd,shape+1,scale=1)
    F2c2k = sp.stats.gamma.cdf(2*cstd,2*shape,scale=1)
    crps = ystd*(2.*Fyk-1.) - cstd*np.square(Fck) + shape*(1.+2.*Fck*FckP1-np.square(Fck)-2*FykP1) - (shape/float(math.pi))*betaf*(1.-F2c2k)
    return ma.mean(scale*crps)


param_initial = [0.05,0.5,0.5,0.9]
param_ranges = ((0.001,1.0), (0.1,1.0), (0.0,3.0), (0.1,1.0))

par_reg = np.zeros((nyrs,nxy,4), dtype=np.float32)
csgd_pars_fcst = np.zeros((ndts,nyrs,nxy,3), dtype=np.float32)

for iyr in range(nyrs):
    print(iyr)
    ### Split data into training and verification data, save day index of observational data
    doy_train = np.outer(doy,np.ones(19,dtype=np.int32)).flatten()
    apcp_obs_ind_train = np.zeros((ndts,nyrs),dtype=np.int32)
    apcp_obs_ind_verif = np.zeros(ndts,dtype=np.int32)
    for idt in range(ndts):
        apcp_obs_ind_verif[idt] = np.nonzero(mod_dates_fcstperiod[idt,iyr]==obs_dates_ord)[0][0]
        for jyr in range(0,nyrs):
            apcp_obs_ind_train[idt,jyr] = np.nonzero(mod_dates_fcstperiod[idt,jyr]==obs_dates_ord)[0][0]
    apcp_obs_ind_train = np.delete(apcp_obs_ind_train,iyr,axis=1)
    ensmean_train = np.delete(ensmean_precip,iyr,axis=1)
    ensmean_clavg = np.mean(ensmean_train,axis=1)
    ensmean_clavg_sm = np.zeros((ndts,nxy), dtype=np.float32)
    for idt in range(ndts):
        wnd_ind = np.minimum(np.minimum(abs(doy[idt]-doy),abs(doy[idt]-365-doy)),abs(doy[idt]+365-doy))<31
        ensmean_clavg_sm[idt,:] = np.mean(ensmean_clavg[wnd_ind,:],axis=0)
    ensmean_ano_train = ensmean_train / ensmean_clavg_sm[:,None,:]
    ensmean_ano_verif = ensmean_precip[:,iyr,:] / ensmean_clavg_sm
    apcp_obs_train = obs_precip_week[apcp_obs_ind_train.flatten(),:]
    for ixy in range(nxy):
        obs = apcp_obs_train[:,ixy].astype(np.float64)
        ensmeanano = ensmean_ano_train[:,:,ixy].flatten().astype(np.float64)
        muc = (shape_cl[doy_train,ixy]*scale_cl[doy_train,ixy]).astype(np.float64)
        sigmac = (np.sqrt(shape_cl[doy_train,ixy])*scale_cl[doy_train,ixy]).astype(np.float64)
        shiftc = (shift_cl[doy_train,ixy]).astype(np.float64)
        par_reg[iyr,ixy,:] = minimize(crpsCondCSGD, param_initial, args=(obs,ensmeanano,muc,sigmac,shiftc), \
                                      method='L-BFGS-B', bounds=param_ranges, tol=1e-6).x
    ### Get mu, sigma and shift for each training day
    mu_cl_verif = shape_cl[doy,:]*scale_cl[doy,:]
    sigma_cl_verif = np.sqrt(shape_cl[doy,:])*scale_cl[doy,:]
    shift_cl_verif = shift_cl[doy,:]
    logarg = par_reg[iyr,:,1] + par_reg[iyr,:,2]*ensmean_ano_verif
    csgd_pars_fcst[:,iyr,:,0] = mu_cl_verif * np.log1p(np.expm1(par_reg[iyr,:,0])*logarg) / par_reg[iyr,:,0]
    csgd_pars_fcst[:,iyr,:,1] = sigma_cl_verif * (par_reg[iyr,:,3]*np.sqrt(csgd_pars_fcst[:,iyr,:,0]/mu_cl_verif))
    csgd_pars_fcst[:,iyr,:,2] = shift_cl_verif


### Save out to file
outfilename = "/Users/mscheuerer/Desktop/CalifAPCP/forecasts/csgd_fcsts_params_week4"
np.savez(outfilename, par_reg= par_reg, csgd_pars_fcst=csgd_pars_fcst)




