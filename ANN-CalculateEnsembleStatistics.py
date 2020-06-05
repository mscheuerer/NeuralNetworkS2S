
import numpy as np
import scipy as sp
import math
import os, sys
import datetime
import time
import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from numpy import ma
from numpy import loadtxt
from scipy.interpolate import interp1d



#plt.ion()

r = 300.   # neighborhood radius (kilometers)
R = 6373.   # earth radius (kilometers)


leadDay = 6      # leadDay=d works out to being a d+0.5 day forecast
accumulation = 7  # Precipitation accumulation period

clead = 'week'+str((leadDay+8)//7)


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
obs_lat = f1['obs_lat']
obs_lon = f1['obs_lon']
obs_1week_dates_ord = f1['obs_dates_ord']
obs_1week_dates = f1['obs_dates']    
f1.close()

nxy = len(obs_lat)


f3 = np.load("/home/michael/Desktop/CalifAPCP/data/mod_precip_calplus.npz")
### Modeled precip is (reforecast time, member, year, lead time, lat, lon)
mod_precip = f3['precip']
#mod_dates_ord = f3['datesOrd']
mod_lon = f3['lon']
mod_lat = f3['lat']
f3.close()

f3 = np.load("/home/michael/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates_ord = f3['dates_ord']
f3.close()

ndts, nmem, nyrs, nlts, nlat, nlon = mod_precip.shape

### Modeled precip 7-day accumulation is (reforecast time, year, ensembles, space)
mod_precip_week = np.sum(mod_precip[:,:,:,leadDay:leadDay+accumulation,:,:],axis=3).reshape((ndts,nmem,nyrs,nlon*nlat))
mod_dates_week = mod_dates_ord[:,:,leadDay]


### Calculate day of the year ('doy') for each reforecast date
doy = np.zeros(ndts,dtype=np.int32)
for idt in range(ndts):
    yyyy = datetime.date.fromordinal(int(mod_dates_week[idt,0])).year
    doy[idt] = min(364,(datetime.date.fromordinal(int(mod_dates_week[idt,0]))-datetime.date(yyyy,1,1)).days)


### Calculate spatially smoothed ensemble foreasts at analysis grid locations
mod_precip_week_sm = np.zeros((ndts,nmem,nyrs,nxy),dtype=np.float32)
for ixy in range(0,nxy):
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
    mod_precip_week_sm[:,:,:,ixy] = np.average(mod_precip_week[:,:,:,uselocs],axis=3,weights=wgt)


qt_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
chf = -np.log(1.-(np.array(qt_levels)))

for iyr in range(0,nyrs):
    print(iyr)
    ### Split data into training and verification data, save day index of observational data
    apcp_obs_ind_train = np.zeros((ndts,nyrs),dtype=np.int32)
    apcp_obs_ind_verif = np.zeros(ndts,dtype=np.int32)
    for idt in range(ndts):
        apcp_obs_ind_verif[idt] = np.nonzero(mod_dates_week[idt,iyr]==obs_1week_dates_ord)[0][0]
        for jyr in range(0,nyrs):
            apcp_obs_ind_train[idt,jyr] = np.nonzero(mod_dates_week[idt,jyr]==obs_1week_dates_ord)[0][0]
    apcp_obs_ind_train = np.delete(apcp_obs_ind_train,iyr,axis=1)
    fcst_train = np.delete(mod_precip_week_sm,iyr,axis=2)
    fcst_verif = mod_precip_week_sm[:,:,iyr,:]
    ### Transform smoothed ensemble forecasts to uniform distribution
    apcp_ens_pit_train = np.zeros((ndts,nyrs-1,nxy,nmem),dtype=np.float32)
    apcp_ens_pit_verif = np.zeros((ndts,nxy,nmem),dtype=np.float32)
    apcp_fcst_p0_cl = np.zeros((ndts,nxy),dtype=np.float32)
    for idt in range(ndts):
        doy_diff = np.minimum(abs(doy-doy[idt]-365),np.minimum(abs(doy-doy[idt]),abs(doy-doy[idt]+365)))
        wnd_ind = np.where(doy_diff<31)[0]
        for ixy in range(nxy):
            x = np.quantile(fcst_train[wnd_ind,:,:,ixy].flatten(),q=qt_levels)
            izmax = np.where(x>0.0)[0][0] - 1
            if izmax<0:
                itp_fct = interp1d(np.append(0.0,x), np.append(0.0,chf), kind='linear',fill_value='extrapolate')
            else:
                itp_fct = interp1d(x[izmax:], chf[izmax:], kind='linear',fill_value='extrapolate')
            apcp_ens_pit_train[idt,:,ixy,:] = np.transpose(1.-np.exp(-itp_fct(fcst_train[idt,:,:,ixy])))
            apcp_ens_pit_verif[idt,ixy,:] = 1.-np.exp(-itp_fct(fcst_verif[idt,:,ixy]))
            apcp_fcst_p0_cl[idt,ixy] = np.mean(fcst_train[wnd_ind,:,:,ixy]==0.0)
    ### Save out to file
    outfilename = "/home/michael/Desktop/CalifAPCP/stats/ensemble_stats_"+clead+"_ANN_yr"+str(iyr)
    np.savez(outfilename, doy_dts=doy, \
        apcp_obs_ind_train=apcp_obs_ind_train, \
        apcp_obs_ind_verif=apcp_obs_ind_verif, \
        apcp_ens_pit_train=apcp_ens_pit_train, \
        apcp_ens_pit_verif=apcp_ens_pit_verif, \
        apcp_fcst_p0_cl=apcp_fcst_p0_cl)







