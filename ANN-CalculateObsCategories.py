
import numpy as np
import scipy as sp
import math
import os, sys
#import matplotlib.pyplot as plt
import matplotlib.path as path
import datetime
import time
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from netCDF4 import Dataset
from numpy import ma
from numpy import loadtxt
from scipy import stats

#plt.ion()

ncat = 30
qtlv_eval = [.333,.667,.85,.95]



#==============================================================================    
# Load PRISM data set, aggregate to 1-week average and calculate doy
#==============================================================================

f1 = np.load("/home/michael/Desktop/CalifAPCP/data/precip_PRISM_cal_19810101_20171231.npz")
#list(f1)
obs_precip = f1['precip']
obs_lat = f1['lat']
obs_lon = f1['lon']
obs_dates_ord = f1['dates_ord']
obs_dates = f1['dates']
f1.close()

ndts, nxy = obs_precip.shape

obs_precip_week = np.zeros((ndts-6,nxy), dtype=np.float32)
for iday in range(7):
    obs_precip_week += obs_precip[iday:(ndts-6+iday),:]

ndts, nxy = obs_precip_week.shape

obs_dates_ord = obs_dates_ord[:ndts]
obs_dates = obs_dates[:ndts]


doy = np.zeros(ndts, dtype=np.int32)
for idt in range(ndts):
    doy[idt] = (datetime.date.fromordinal(int(obs_dates_ord[idt]))-datetime.date(obs_dates[idt,0],1,1)).days
    if datetime.date(obs_dates[idt,0],1,1).year%4==0 and  doy[idt]>58:
        doy[idt] -= 1                                                     # in leap year, do not count Feb 29



#==============================================================================    
# Estimate climatological PoP and 'hybrid' quantiles using a moving window
#==============================================================================

pop_doy = np.zeros((365,nxy), dtype=np.float32)
thr_doy = np.zeros((365,nxy,ncat-1), dtype=np.float32)
qtev_doy = np.zeros((365,nxy,len(qtlv_eval)), dtype=np.float32)

for idd in range(365):
    print(idd)
    ind_doy = np.where(doy==idd)[0]
    ind_doy_ext = np.append(np.append(ind_doy[0]-366,ind_doy),ind_doy[-1]+365)
    wnd_ind = np.add.outer(ind_doy_ext,np.arange(-30,31)).flatten()
    imin = np.where(wnd_ind>=0)[0][0]
    imax = np.where(wnd_ind<ndts)[0][-1]
    for ixy in range(nxy):
        y = obs_precip_week[wnd_ind[imin:(imax+1)],ixy]
        pop_doy[idd,ixy] = np.mean(y>0.254)
        thr_doy[idd,ixy,0] = 0.254
        qtlv = 1. + pop_doy[idd,ixy]*((np.arange(1,ncat-1)/float(ncat-1))-1.)
        thr_doy[idd,ixy,1:] = np.quantile(y,qtlv)
        qtev_doy[idd,ixy,:] = np.maximum(0.254,np.quantile(y,qtlv_eval))



#==============================================================================    
# Assign observations to classes (multiple assignments allowed if ambiguous)
#==============================================================================

apcp_obs_cat = np.zeros((ndts,nxy,ncat),dtype=np.bool_)

for idt in range(ndts):
    for ixy in range(0,nxy):
        lower = np.append(-np.Inf,thr_doy[doy[idt],ixy,:])
        upper = np.append(thr_doy[doy[idt],ixy,:],np.Inf)
        apcp_obs_cat[idt,ixy,:] = np.logical_and(obs_precip_week[idt,ixy]>=lower,obs_precip_week[idt,ixy]<=upper)

np.savez("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_"+str(ncat)+"cl",
    obs_lat = obs_lat,
    obs_lon = obs_lon,
    obs_dates_ord = obs_dates_ord,
    obs_dates = obs_dates,    
    apcp_obs_cat = apcp_obs_cat,
    apcp_obs = obs_precip_week,
    pop_doy = pop_doy,
    thr_doy = thr_doy,
    qtev_doy = qtev_doy)







