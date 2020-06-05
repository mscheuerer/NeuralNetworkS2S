
import numpy as np
import scipy as sp
import math
import os, sys
#import matplotlib.pyplot as plt
import datetime
import time
import keras
import keras.backend as K

from netCDF4 import Dataset
from numpy import ma
from numpy.linalg import solve
from scipy import stats
from scipy.interpolate import interp1d

from keras import models
from keras import layers
from keras import regularizers

from keras.layers import Input, Dense, Dot, Add, Activation, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from keras.models import Model
from keras.optimizers import Adam

#plt.ion()


##  Load categorical analysis data

f1 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
lat = f1['obs_lat']
lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
pop_doy = f1['pop_doy']
thr_doy = f1['thr_doy']
qtev_doy = f1['qtev_doy']
apcp_obs_cat = f1['apcp_obs_cat']
apcp_obs = f1['apcp_obs']
f1.close()

ndts, nxy, ncat = apcp_obs_cat.shape



##  Load ERA5 z500 and tcw fields, subset to 22 x 18 image, same for the ensemble forecast fields

ixl = 10
ixu = -6
jyl = 6
jyu = -6

f2 = np.load("/Users/mscheuerer/Desktop/CalifAPCP/data/z500_tcw_predictors_era5.npz")
era5_dates_ord = f2['dates_ord']
era5_lon = f2['longitude'][ixl:ixu]
era5_lat = f2['latitude'][jyl:jyu]
z500 = f2['z500_1wk'][:,:,jyl:jyu,ixl:ixu]
tcw = f2['tcw_1wk'][:,:,jyl:jyu,ixl:ixu]
f2.close()

ndts, nyrs, ny, nx = z500.shape



##########################################################################################################################################################################
#
#  Upscale to 2 degrees
#
z500_1deg = z500
tcw_1deg = tcw
z500 = (z500_1deg[:,:,0:ny:2,0:nx:2]+z500_1deg[:,:,1:ny:2,0:nx:2]+z500_1deg[:,:,0:ny:2,1:nx:2]+z500_1deg[:,:,1:ny:2,1:nx:2])/4.
tcw = (tcw_1deg[:,:,0:ny:2,0:nx:2]+tcw_1deg[:,:,1:ny:2,0:nx:2]+tcw_1deg[:,:,0:ny:2,1:nx:2]+tcw_1deg[:,:,1:ny:2,1:nx:2])/4.
ny = ny//2
nx = nx//2
#
##########################################################################################################################################################################


## Calculate doy for each analysis date

doy_dts = np.zeros(ndts,dtype=np.int32)
apcp_obs_ind = np.zeros((ndts,nyrs),dtype=np.int32)
for idt in range(ndts):
    for iyr in range(nyrs):
        apcp_obs_ind[idt,iyr] = np.where(obs_dates_ord==era5_dates_ord[idt,iyr])[0][0]
    date_ord = int(era5_dates_ord[idt,0]-0.5)
    doy_dts[idt] = min(364,(datetime.date.fromordinal(date_ord)-datetime.date(datetime.date.fromordinal(date_ord).year,1,1)).days)



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



# Define basis functions

r_basis = 7.
lon_ctr = np.outer(np.arange(-124,-115,3.5),np.ones(3)).reshape(9)[[2,4,5,6,7]]
lat_ctr = np.outer(np.ones(3),np.arange(33,42,3.5)).reshape(9)[[2,4,5,6,7]]

dst_lon = np.abs(np.subtract.outer(lon,lon_ctr))
dst_lat = np.abs(np.subtract.outer(lat,lat_ctr))
dst = np.sqrt(dst_lon**2+dst_lat**2)
basis = np.where(dst>r_basis,0.,(1.-(dst/r_basis)**3)**3)
basis = basis/np.sum(basis,axis=1)[:,None]
nbs = basis.shape[1]



##  Define functions for building a CNN

def build_cat_model(n_xy, n_bins, n_basis, hidden_nodes, dropout_rate):
    #inp_imgs = Input(shape=(18,22,2,))
    inp_imgs = Input(shape=(9,11,2,))
    inp_basis = Input(shape=(n_xy,n_basis,))
    inp_cl = Input(shape=(n_xy,n_bins,))
    c = Conv2D(4, (3,3), activation='elu')(inp_imgs)
    #c = MaxPooling2D((2,2))(c)
    c = Conv2D(8, (3,3), activation='elu')(c)
    #c = MaxPooling2D((2,2))(c)
    x = Flatten()(c)
    for h in hidden_nodes:
        x = Dropout(dropout_rate)(x)
        x = Dense(h, activation='elu')(x)
    x = Dense(n_bins*n_basis, activation='elu')(x)
    x = Reshape((n_bins,n_basis))(x)
    z = Dot(axes=2)([inp_basis, x])
    z = Add()([z, inp_cl])
    out = Activation('softmax')(z)
    return Model(inputs=[inp_imgs, inp_basis, inp_cl], outputs=out)


def modified_categorical_crossentropy(y_mat, prob_fcst):
    prob_obs_cat = K.sum(y_mat*prob_fcst,axis=2)
    return -K.mean(K.log(prob_obs_cat))


nyrs = 20

#reg = 10.**np.arange(-6,-2)
reg = np.arange(0.1,0.6,0.1)
nreg = len(reg)

imod = 0

mod = [[10],[20],[10,10]]


x = (np.arange(0,101)/5)**2      # evaluation points for numerical calculation of the CRPS
dx = np.diff(x)

opt_reg_param = np.zeros(nyrs, dtype=np.float32)
opt_valid_scores = np.zeros((nyrs,5), dtype=np.float32)
opt_valid_crps = np.zeros((nyrs,5), dtype=np.float32)


for iyr in range(nyrs):
    print('year: ',iyr)
    # Calculate image predictors and basis functions
    apcp_obs_ind_cv = np.delete(apcp_obs_ind,iyr,axis=1)
    z500_pred_cv = np.delete(z500_ano,iyr,axis=1).reshape((ndts*(nyrs-1),ny,nx,1))
    tcw_pred_cv = np.delete(tcw_ano,iyr,axis=1).reshape((ndts*(nyrs-1),ny,nx,1))
    pred_imgs_cv = np.concatenate((z500_pred_cv,tcw_pred_cv),axis=3)
    basis_cv = np.repeat(basis[np.newaxis,:,:],ndts*(nyrs-1),axis=0)
    # Calculate climatological log probabilities for each class
    apcp_pop_cl = np.repeat(pop_doy[doy_dts,np.newaxis,:],nyrs-1,axis=1).reshape((ndts*(nyrs-1),nxy,1))
    logp_cl_cv = np.concatenate((np.log(1.-apcp_pop_cl),np.repeat(np.log(apcp_pop_cl),ncat-1,axis=2)-np.log(ncat-1)),axis=2)
    # perform 5-fold cross validation to find optimal regularization
    date_order = np.arange(ndts*(nyrs-1)).reshape(ndts,nyrs-1).T.flatten()
    cv_ind = date_order[np.arange(ndts*(nyrs-1))%232<231]                        # remove the date between the 5 cross-validated blocks
    valid_score = np.zeros((nreg,5), dtype=np.float32)
    valid_crps = np.zeros((nreg,5), dtype=np.float32)
    for cvi in range(5):
        train_ind = cv_ind[np.arange(len(cv_ind))//(len(cv_ind)//5)!=cvi]
        valid_ind = cv_ind[np.arange(len(cv_ind))//(len(cv_ind)//5)==cvi]
        pred_imgs_train = pred_imgs_cv[train_ind,:,:,:]
        basis_train = basis_cv[train_ind,:,:]
        logp_cl_train = logp_cl_cv[train_ind,:,:]
        cat_targets_train = apcp_obs_cat[apcp_obs_ind_cv.flatten()[train_ind],:,:].astype(float)
        pred_imgs_valid = pred_imgs_cv[valid_ind,:,:]
        basis_valid = basis_cv[valid_ind,:,:]
        logp_cl_valid = logp_cl_cv[valid_ind,:,:]
        cat_targets_valid = apcp_obs_cat[apcp_obs_ind_cv.flatten()[valid_ind],:,:].astype(float)
        doy_valid = np.repeat(doy_dts[:,np.newaxis],nyrs-1,axis=1).flatten()[valid_ind]
        for ireg in range(nreg):
            # Define and fit ANN model (using batch gradient descent)
            keras.backend.clear_session()
            model = model = build_cat_model(nxy, ncat, nbs, mod[imod], reg[ireg])
            model.compile(optimizer=Adam(0.01), loss=modified_categorical_crossentropy)
            model.fit([pred_imgs_train,basis_train,logp_cl_train], cat_targets_train, epochs=150, batch_size=len(train_ind), verbose=0)
            valid_score[ireg,cvi] = model.evaluate([pred_imgs_valid,basis_valid,logp_cl_valid], cat_targets_valid, batch_size=len(valid_ind), verbose=0)
            # Calculate CRPS for each cross-validation fold
            prob_fcst_cat = model.predict([pred_imgs_valid,basis_valid,logp_cl_valid])
            prob_fcst_chf = -np.log(np.maximum(1.-np.cumsum(prob_fcst_cat,axis=2)[:,:,:(ncat-1)],1.e-10))
            crps_fold = np.zeros((len(valid_ind),nxy),dtype=np.float32)
            for ivdt in range(len(valid_ind)):
                for ixy in range(nxy):
                    itp_fct = interp1d(thr_doy[doy_valid[ivdt],ixy,:], prob_fcst_chf[ivdt,ixy,:], kind='linear',fill_value='extrapolate')
                    bs = (1.-np.exp(-itp_fct(x))-1.*(apcp_obs[apcp_obs_ind_cv.flatten()[valid_ind[ivdt]],ixy]<=x))**2
                    crps_fold[ivdt,ixy] = 0.5*np.sum((bs[1:]+bs[:len(dx)])*dx)
            valid_crps[ireg,cvi] = np.mean(crps_fold)
    opt_reg_ind = np.argmin(np.mean(valid_score,axis=1))
    opt_reg_param[iyr] = reg[opt_reg_ind]
    opt_valid_scores[iyr,:] = valid_score[opt_reg_ind,:]
    opt_valid_crps[iyr,:] =  valid_crps[opt_reg_ind,:]
    print(np.mean(valid_score,axis=1).round(3))
    print(np.mean(valid_crps,axis=1).round(2))
    print(opt_reg_param[iyr])

### Save out to file
outfilename = "/Users/mscheuerer/Desktop/CalifAPCP/tuning/cnn-2deg-m"+str(imod)+"-drpt-f48"
np.savez(outfilename, opt_reg_param=opt_reg_param, opt_valid_scores=opt_valid_scores, opt_valid_crps=opt_valid_crps)





