
import numpy as np
import scipy as sp
import math
import os, sys
import matplotlib.pyplot as plt
import matplotlib.path as path
import datetime
import time

from netCDF4 import Dataset
from numpy import ma
from numpy import loadtxt
from scipy import stats
from scipy.stats import gamma
from scipy.special import beta
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

plt.ion()


def crpsClimoCSGD(shape,obs,mean,pop):
    # average CRPS for climatological CSGD as a function of shape (pop and mean fixed)
    crps = np.zeros(len(obs),dtype='float64')
    Fck = 1.-pop
    cstd = gamma.ppf(Fck,shape)
    fkp1q0 = gamma.pdf(cstd,shape+1.,scale=1.)
    scale = (mean-0.254*pop) / (shape*(pop+fkp1q0)-pop*cstd)    # assumes that precipitation amounts < 0.254 mm are considered zero
    shift = 0.254-cstd*scale
    penalty = max(0.005-shape*scale-shift,0.0)    # penalize shifts that would move most of the PDF below zero 
    betaf = beta(0.5,shape+0.5)
    FckP1 = gamma.cdf(cstd,shape+1,scale=1)
    F2c2k = gamma.cdf(2*cstd,2*shape,scale=1)
    indz = np.less(obs,0.254)
    indp = np.greater_equal(obs,0.254)
    ystd = (obs[indp]-shift)/scale
    Fyk = gamma.cdf(ystd,shape,scale=1)
    FykP1 = gamma.cdf(ystd,shape+1,scale=1)
    crps[indz] = cstd*(2.*Fck-1.) - cstd*np.square(Fck) \
      + shape*(1.+2.*Fck*FckP1-np.square(Fck)-2*FckP1) \
      - (shape/float(math.pi))*betaf*(1.-F2c2k)
    crps[indp] = ystd*(2.*Fyk-1.) - cstd*np.square(Fck) \
      + shape*(1.+2.*Fck*FckP1-np.square(Fck)-2*FykP1) \
      - (shape/float(math.pi))*betaf*(1.-F2c2k)
    return scale*ma.mean(crps) + penalty



#==============================================================================    
# Load the PRISM gridded precipitation data and fit monthly CSGD distribution
#==============================================================================

f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/precip_PRISM_cal_19810101_20171231.npz")
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

obs_precip_week[obs_precip_week<0.254] = 0.
obs_dates_ord = obs_dates_ord[:ndts]
obs_dates = obs_dates[:ndts]


pop_month = np.zeros((12,nxy), dtype=np.float32)
mean_month = np.zeros((12,nxy), dtype=np.float32)
shape_month = np.zeros((12,nxy), dtype=np.float32)

mid_mon = [14,45,73,104,134,165,195,226,257,287,318,348]

for imonth in range(0,12):
    date2 = datetime.datetime(2001,1,1)+datetime.timedelta(mid_mon[imonth])
    fnd_month = np.nonzero(obs_dates[:,1]==date2.month)[0]
    fnd_day = np.nonzero(obs_dates[fnd_month,2]==date2.day)[0]
    day_array = []
    for windowval in range(-30,31):
        day_array.extend(fnd_month[fnd_day]+windowval)
    day_array = np.sort(np.array(day_array))
    day_array = day_array[day_array>=0]
    day_array = day_array[day_array<len(obs_dates)]
    for ixy in range(0,nxy):
        obs = obs_precip_week[day_array,ixy]
        obs = obs[np.isnan(obs)==False]
        pop_month[imonth,ixy] = np.mean(np.greater(obs,.1))
        mean_month[imonth,ixy] = np.mean(obs)
        if pop_month[imonth,ixy]<0.002:                                   # very dry location, use fixed (very dry) CSGD
            pop_month[imonth,ixy] = 1.-gamma.cdf(0.254,0.0016,scale=1.25)
            mean_month[imonth,ixy] = 0.002*(1-gamma.cdf(0.254,1.0016,scale=1.25))
            shape_month[imonth,ixy] = 0.0016
            continue
        shape_month[imonth,ixy] = minimize_scalar(crpsClimoCSGD, args=(obs,mean_month[imonth,ixy],pop_month[imonth,ixy]), method='bounded', bounds=(.0016,1.5)).x
        if ixy%100==0:
            print(imonth+1,ixy+1)



#==============================================================================    
# Interpolate parameters from mid-month to each day of the year
#==============================================================================

pop_doy = np.zeros((366,nxy), dtype=np.float32)
mean_doy = np.zeros((366,nxy), dtype=np.float32)
shape_doy = np.zeros((366,nxy), dtype=np.float32)
scale_doy = np.zeros((366,nxy), dtype=np.float32)
shift_doy = np.zeros((366,nxy), dtype=np.float32)


mid_ind = np.array((-17,14,45,73,104,134,165,195,226,257,287,318,348,379), dtype=np.float32)

for idd in range(366):
    print( 'Processing doy '+str(idd+1))
    iup = np.where(idd<mid_ind)[0][0]
    wlw = (mid_ind[iup]-idd) / (mid_ind[iup]-mid_ind[iup-1])
    wup = (idd-mid_ind[iup-1]) / (mid_ind[iup]-mid_ind[iup-1])
    imlw = (11,0,1,2,3,4,5,6,7,8,9,10,11,0)[iup-1]
    imup = (11,0,1,2,3,4,5,6,7,8,9,10,11,0)[iup]
    for ixy in range(0,nxy):
        pop_doy[idd,ixy] = wlw*pop_month[imlw,ixy] + wup*pop_month[imup,ixy]
        mean_doy[idd,ixy] = wlw*mean_month[imlw,ixy] + wup*mean_month[imup,ixy]
        shape_doy[idd,ixy] = wlw*shape_month[imlw,ixy] + wup*shape_month[imup,ixy]
        q0 = gamma.ppf(1.-pop_doy[idd,ixy],shape_doy[idd,ixy])
        scale_doy[idd,ixy] = (mean_doy[idd,ixy]-0.254*pop_doy[idd,ixy])/(shape_doy[idd,ixy]*(1.-gamma.cdf(q0,shape_doy[idd,ixy]+1.))-pop_doy[idd,ixy]*q0)
        shift_doy[idd,ixy] = 0.254-scale_doy[idd,ixy]*q0


np.savez("/Users/mscheuerer/Desktop/CalifAPCP/data/precip_climatological_csgd",
    pop_cl_doy=pop_doy, mean_cl_doy = mean_doy, shape_cl_doy = shape_doy, scale_cl_doy = scale_doy, shift_cl_doy = shift_doy)




