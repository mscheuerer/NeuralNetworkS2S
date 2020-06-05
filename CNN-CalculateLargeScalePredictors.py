
import numpy as np
import scipy as sp
import math
import os, sys
import datetime
import time
#import matplotlib.path as path
#import matplotlib.patches as patches
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from numpy import ma
from numpy.linalg import solve
from numpy.linalg import svd

#plt.ion()

data_path = '/Volumes/ExtMichael/Michael/ECMWF-subseasonal/'



###  Load geopotential height forecast fields and aggregate to week-2, week-3, and week-4 averages

f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates_ord = f1['dates_ord']
f1.close()

ndts, nyrs, nlts = mod_dates_ord.shape

ixl = 71    # -144
ixu = 147   # -107
jyl = 55    #   52
jyu = 115   #   23

nxf = len(range(ixl,ixu+1))
nyf = len(range(jyl,jyu+1))

nens = 11

z500_week2 = np.zeros((ndts,nyrs,nens,nyf,nxf), dtype=np.float32)
z500_week3 = np.zeros((ndts,nyrs,nens,nyf,nxf), dtype=np.float32)
z500_week4 = np.zeros((ndts,nyrs,nens,nyf,nxf), dtype=np.float32)

wgt12h = np.r_[0.5,np.ones(13,dtype=np.float32),0.5]

for idt in range(ndts):
    date_init = datetime.date.fromordinal(int(mod_dates_ord[idt,-1,0]-1.5))    # Initialization date of ECMWF reforecast
    cyear = format(date_init.year+1)
    cmonth = format(date_init.month,'02')
    cday = format(date_init.day,'02')
    infile = data_path+'ControlLargeDomain/geopotential/'+cyear+'-'+cmonth+'-'+cday+'cntrl_12hrpress_start0hr.nc'
    nc = Dataset(infile)
    z = nc.variables['z'][:,:,:,jyl:(jyu+1),ixl:(ixu+1)]
    nc.close()
    z500_week2[idt,:,0,:,:] = np.average(z[:,13:28,0,:,:],axis=1,weights=wgt12h)
    z500_week3[idt,:,0,:,:] = np.average(z[:,27:42,0,:,:],axis=1,weights=wgt12h)
    z500_week4[idt,:,0,:,:] = np.average(z[:,41:56,0,:,:],axis=1,weights=wgt12h)
    print(infile)
    infile = data_path+'EnsembleLargeDomain/geopotential/'+cyear+'-'+cmonth+'-'+cday+'ens_12hrpress_start0hr.z.nc'
    nc = Dataset(infile)
    z = nc.variables['z'][:,:,:,:,jyl:(jyu+1),ixl:(ixu+1)]
    nc.close()
    z500_week2[idt,:,1:,:,:] = np.average(z[:,13:28,:,0,:,:],axis=1,weights=wgt12h)
    z500_week3[idt,:,1:,:,:] = np.average(z[:,27:42,:,0,:,:],axis=1,weights=wgt12h)
    z500_week4[idt,:,1:,:,:] = np.average(z[:,41:56,:,0,:,:],axis=1,weights=wgt12h)
    print(infile)


#  Upscale to 1-deg grid

nxfu = (nxf-1)//2
nyfu = (nyf-1)//2

z500_week2_1deg = np.zeros((ndts,nyrs,nens,nyfu,nxfu), dtype=np.float32)
z500_week3_1deg = np.zeros((ndts,nyrs,nens,nyfu,nxfu), dtype=np.float32)
z500_week4_1deg = np.zeros((ndts,nyrs,nens,nyfu,nxfu), dtype=np.float32)

for ixd in range(-1,2):
    wx = 0.5**(1+abs(ixd))
    for jyd in range(-1,2):
        wy = 0.5**(1+abs(jyd))
        w = wx*wy
        z500_week2_1deg += z500_week2[:,:,:,(1+jyd):(nyf-1+jyd):2,(1+ixd):(nxf-1+ixd):2]*w
        z500_week3_1deg += z500_week3[:,:,:,(1+jyd):(nyf-1+jyd):2,(1+ixd):(nxf-1+ixd):2]*w
        z500_week4_1deg += z500_week4[:,:,:,(1+jyd):(nyf-1+jyd):2,(1+ixd):(nxf-1+ixd):2]*w


### Save out to file
outfilename = "/Users/mscheuerer/Desktop/CalifAPCP/data/z500_predictor_cnn"
np.savez(outfilename, mod_dates_ord=mod_dates_ord,
             longitude=lon.data[(ixl+1):ixu:2],
             latitude=lat.data[(jyl+1):jyu:2],
             z500_week2=z500_week2_1deg,
             z500_week3=z500_week3_1deg,
             z500_week4=z500_week4_1deg)




###  Load total column water forecast fields and aggregate to week-2, week-3, and week-4 averages

f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates_ord = f1['dates_ord']
f1.close()

ndts, nyrs, nlts = mod_dates_ord.shape

ixl = 71    # -144
ixu = 147   # -107
jyl = 55    #   52
jyu = 115   #   23

nxf = len(range(ixl,ixu+1))
nyf = len(range(jyl,jyu+1))

nens = 11

tcw_week2 = np.zeros((ndts,nyrs,nens,nyf,nxf), dtype=np.float32)
tcw_week3 = np.zeros((ndts,nyrs,nens,nyf,nxf), dtype=np.float32)
tcw_week4 = np.zeros((ndts,nyrs,nens,nyf,nxf), dtype=np.float32)

wgt6h = np.r_[0.5,np.ones(27,dtype=np.float32),0.5]

for idt in range(ndts):
    date_init = datetime.date.fromordinal(int(mod_dates_ord[idt,-1,0]-1.5))    # Initialization date of ECMWF reforecast
    cyear = format(date_init.year+1)
    cmonth = format(date_init.month,'02')
    cday = format(date_init.day,'02')
    infile = data_path+'ControlLargeDomain/tcw/'+cyear+'-'+cmonth+'-'+cday+'cntrl_6hrsfc_start0hr.nc'
    nc = Dataset(infile)
    twc = nc.variables['tcw'][:,:,jyl:(jyu+1),ixl:(ixu+1)]
    nc.close()
    tcw_week2[idt,:,0,:,:] = np.average(twc[:,26:55,:,:],axis=1,weights=wgt6h)
    tcw_week3[idt,:,0,:,:] = np.average(twc[:,54:83,:,:],axis=1,weights=wgt6h)
    tcw_week4[idt,:,0,:,:] = np.average(twc[:,82:111,:,:],axis=1,weights=wgt6h)
    print(infile)
    infile = data_path+'EnsembleLargeDomain/tcw/'+cyear+'-'+cmonth+'-'+cday+'ens_6hrsfc_start0hr.tcw.nc'
    nc = Dataset(infile)
    twc = nc.variables['tcw'][:,:,:,jyl:(jyu+1),ixl:(ixu+1)]
    nc.close()
    tcw_week2[idt,:,1:,:,:] = np.average(twc[:,26:55,:,:],axis=1,weights=wgt6h)
    tcw_week3[idt,:,1:,:,:] = np.average(twc[:,54:83,:,:],axis=1,weights=wgt6h)
    tcw_week4[idt,:,1:,:,:] = np.average(twc[:,82:111,:,:],axis=1,weights=wgt6h)
    print(infile)

#nc = Dataset(infile)
#lons = nc.variables['longitude'][ixl:(ixu+1)]
#lats = nc.variables['latitude'][jyl:(jyu+1)]
#nc.close()



#  Upscale to 1-deg grid

nxfu = (nxf-1)//2
nyfu = (nyf-1)//2

tcw_week2_1deg = np.zeros((ndts,nyrs,nens,nyfu,nxfu), dtype=np.float32)
tcw_week3_1deg = np.zeros((ndts,nyrs,nens,nyfu,nxfu), dtype=np.float32)
tcw_week4_1deg = np.zeros((ndts,nyrs,nens,nyfu,nxfu), dtype=np.float32)

for ixd in range(-1,2):
    wx = 0.5**(1+abs(ixd))
    for jyd in range(-1,2):
        wy = 0.5**(1+abs(jyd))
        w = wx*wy
        tcw_week2_1deg += tcw_week2[:,:,:,(1+jyd):(nyf-1+jyd):2,(1+ixd):(nxf-1+ixd):2]*w
        tcw_week3_1deg += tcw_week3[:,:,:,(1+jyd):(nyf-1+jyd):2,(1+ixd):(nxf-1+ixd):2]*w
        tcw_week4_1deg += tcw_week4[:,:,:,(1+jyd):(nyf-1+jyd):2,(1+ixd):(nxf-1+ixd):2]*w


### Save out to file
outfilename = "/Users/mscheuerer/Desktop/CalifAPCP/data/tcw_predictor_cnn"
np.savez(outfilename, mod_dates_ord=mod_dates_ord, tcw_week2=tcw_week2_1deg, tcw_week3=tcw_week3_1deg, tcw_week4=tcw_week4_1deg)


f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/tcw_predictor.npz")
tcw_week2 = f1['tcw_week2']
tcw_week3 = f1['tcw_week3']
tcw_week4 = f1['tcw_week4']
f1.close()



# Load ERA-5 reanalyses for z500

data_path = '/Projects/ClimateAnalysis/OBS/ERA5/'

infile = data_path+'GEOPOT500.1981.4x.nc'
nc = Dataset(infile)
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
nc.close()

lon = np.where(lon>180,lon-360,lon)

idx_lon = np.logical_and(np.greater_equal(lon,-144.5),np.less_equal(lon,-106.5))
idx_lat = np.logical_and(np.greater_equal(lat,22.5),np.less_equal(lat,52.5))

nx = sum(idx_lon)
ny = sum(idx_lat)
ntimes = 4*(737059-723181+1)

dates_ord = np.zeros(ntimes,dtype=np.float32)
z500 = np.zeros((ntimes,ny,nx),dtype=np.float32)

idtb = 0

for iyr in range(38):
    infile = data_path+'GEOPOT500.'+str(1981+iyr)+'.4x.nc'
    print(infile)
    nc = Dataset(infile)
    ntyr = len(nc.dimensions['time'])
    idte = idtb + ntyr
    dates_ord[idtb:idte] = 657072 + nc.variables['time'][:]/24.
    z500[idtb:idte,:,:] = nc.variables['GEOPOT'][:,0,idx_lat,idx_lon]/9.806
    nc.close()
    idtb = idte

#  Upscale to 1-deg grid

nxu = (nx-1)//4
nyu = (ny-1)//4

z500_1deg = np.zeros((ntimes,nyu,nxu), dtype=np.float32)

for ixd in range(-2,3):
    wx = 0.125*min(3-abs(ixd),2)
    for jyd in range(-2,3):
        wy = 0.125*min(3-abs(jyd),2)
        w = wx*wy
        z500_1deg += z500[:,(2+jyd):(ny-2+jyd):4,(2+ixd):(nx-2+ixd):4]*w


#  Accumulate to 7-day averages

f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates_ord = f1['dates_ord']
f1.close()

ndts, nyrs, nlts = mod_dates_ord.shape

wgt6h = np.r_[0.5,np.ones(27,dtype=np.float32),0.5]

z500_acc1wk = np.zeros((ndts,nyrs,nyu,nxu), dtype=np.float32)

for idt in range(ndts):
    for iyr in range(nyrs):
        date_init_ord = mod_dates_ord[idt,iyr,0]-1.            # Initialization date of ECMWF reforecast
        era5_ind = np.where(dates_ord==date_init_ord)[0]
        if len(era5_ind)<1:
            print('Waring! No match found for idt='+str(idt)+', iyr='+str(iyr)+'.\n')
            continue
        idtl = era5_ind[0]
        idtu = era5_ind[0] + 29
        if idtu>ntimes:
            print('Waring! Aggregation period outside the data range for idt='+str(idt)+', iyr='+str(iyr)+'.\n')
            continue
        z500_acc1wk[idt,iyr,:,:] = np.average(z500_1deg[idtl:idtu,:,:],axis=0,weights=wgt6h)



# Load ERA-5 reanalyses for tcw

data_path = '/Projects/ClimateAnalysis/OBS/ERA5/'

infile = data_path+'TCW.1981.nc'
nc = Dataset(infile)
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
nc.close()

lon = np.where(lon>180,lon-360,lon)

idx_lon = np.logical_and(np.greater_equal(lon,-144.5),np.less_equal(lon,-106.5))
idx_lat = np.logical_and(np.greater_equal(lat,22.5),np.less_equal(lat,52.5))

nx = sum(idx_lon)
ny = sum(idx_lat)
ntimes = 4*(737059-723181+1)

dates_ord = np.zeros(ntimes,dtype=np.float32)
tcw = np.zeros((ntimes,ny,nx),dtype=np.float32)

idtb = 0

for iyr in range(38):
    infile = data_path+'TCW.'+str(1981+iyr)+'.nc'
    print(infile)
    nc = Dataset(infile)
    ntyr = len(nc.dimensions['time'])
    idte = idtb + ntyr
    dates_ord[(idtb//6):(idte//6)] = 657072 + nc.variables['time'][::6]/24.
    tcw[(idtb//6):(idte//6),:,:] = nc.variables['TCW'][::6,idx_lat,idx_lon]/9.806
    nc.close()
    idtb = idte


#  Upscale to 1-deg grid

nxu = (nx-1)//4
nyu = (ny-1)//4

tcw_1deg = np.zeros((ntimes,nyu,nxu), dtype=np.float32)

for ixd in range(-2,3):
    wx = 0.125*min(3-abs(ixd),2)
    for jyd in range(-2,3):
        wy = 0.125*min(3-abs(jyd),2)
        w = wx*wy
        tcw_1deg += tcw[:,(2+jyd):(ny-2+jyd):4,(2+ixd):(nx-2+ixd):4]*w

lon_1deg = lon[idx_lon][2:nx-2:4]
lat_1deg = lat[idx_lat][2:ny-2:4]


#  Accumulate to 7-day averages

f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/mod_precip_cal.npz")
mod_dates_ord = f1['dates_ord']
f1.close()

ndts, nyrs, nlts = mod_dates_ord.shape

era5_dates_ord = mod_dates_ord[:,:,0]-1.
wgt6h = np.r_[0.5,np.ones(27,dtype=np.float32),0.5]

tcw_acc1wk = np.zeros((ndts,nyrs,nyu,nxu), dtype=np.float32)

for idt in range(ndts):
    for iyr in range(nyrs):
        date_init_ord = era5_dates_ord[idt,iyr]               # Initialization date of ECMWF reforecast
        era5_ind = np.where(dates_ord==date_init_ord)[0]
        if len(era5_ind)<1:
            print('Waring! No match found for idt='+str(idt)+', iyr='+str(iyr)+'.\n')
            continue
        idtl = era5_ind[0]
        idtu = era5_ind[0] + 29
        if idtu>ntimes:
            print('Waring! Aggregation period outside the data range for idt='+str(idt)+', iyr='+str(iyr)+'.\n')
            continue
        tcw_acc1wk[idt,iyr,:,:] = np.average(tcw_1deg[idtl:idtu,:,:],axis=0,weights=wgt6h)


### Save out to file
outfilename = "/Users/mscheuerer/Desktop/CalifAPCP/data/z500_tcw_predictors_era5"
np.savez(outfilename, dates_ord=era5_dates_ord,
             longitude=lon_1deg,
             latitude=lat_1deg.data,
             z500_1wk=z500_acc1wk,
             tcw_1wk=tcw_acc1wk)







