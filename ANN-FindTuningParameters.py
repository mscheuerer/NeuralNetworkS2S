
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

from scipy import stats
from scipy.interpolate import interp1d

from keras import models
from keras import layers
from keras import regularizers

from keras.layers import Input, Dense, Add, Activation
from keras.models import Model
from keras.optimizers import Adam

#plt.ion()


ncl = '20'
clead = 'week2'
imod = 0


def build_cat_model(n_features, hidden_nodes, n_bins, par_reg):
    inp1 = Input(shape=(n_features,))
    inp2 = Input(shape=(n_bins,))
    x = Dense(hidden_nodes[0], activation='elu', kernel_regularizer=regularizers.l1(par_reg))(inp1)
    if len(hidden_nodes) > 1:
        for h in hidden_nodes[1:]:
            x = Dense(h, activation='elu', kernel_regularizer=regularizers.l1(par_reg))(x)
    x = Dense(n_bins, activation='elu', kernel_regularizer=regularizers.l1(par_reg))(x)
    x = Add()([x, inp2])
    out = Activation('softmax')(x)
    return Model(inputs=[inp1, inp2], outputs=out)


def modified_categorical_crossentropy(y_mat, prob_fcst):
    prob_obs_cat = K.sum(y_mat*prob_fcst,axis=1)
    return -K.mean(K.log(prob_obs_cat))



f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_"+ncl+"cl.npz")
#list(f1)
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


nyrs = 20

reg = 10.**np.arange(-6,-2)
nreg = len(reg)

mod = [[10],[20],[10,10]]


x = (np.arange(0,101)/5)**2      # evaluation points for numerical approximation of the CRPS
dx = np.diff(x)

opt_reg_param = np.zeros(nyrs, dtype=np.float32)
opt_valid_scores = np.zeros((nyrs,5), dtype=np.float32)
opt_valid_crps = np.zeros((nyrs,5), dtype=np.float32)


for iyr in range(nyrs):
    print('year: ',iyr)
    # Load smoothed ensemble forecast  PIT values
    f4 = np.load("/home/michael/Desktop/CalifAPCP/stats/ensemble_stats_"+clead+"_ANN_yr"+str(iyr)+".npz")
    doy_dts = f4['doy_dts']
    apcp_obs_ind = f4['apcp_obs_ind_train']
    apcp_ens_pit = f4['apcp_ens_pit_train']
    f4.close()
    ndts, nyrs_cv, nxy, nmem = apcp_ens_pit.shape
    # Calculate normalized coordinates, cosine/sine of day of the year, and climatological probability of precipitation
    lon_nml = np.repeat(-1.+2.*(lon[np.newaxis,:]-min(lon))/(max(lon)-min(lon)),ndts*nyrs_cv,axis=0).reshape((ndts*nyrs_cv,nxy,1))
    lat_nml = np.repeat(-1.+2.*(lat[np.newaxis,:]-min(lat))/(max(lat)-min(lat)),ndts*nyrs_cv,axis=0).reshape((ndts*nyrs_cv,nxy,1))
    apcp_pop_cl = np.repeat(pop_doy[doy_dts,np.newaxis,:],nyrs_cv,axis=1).reshape((ndts*nyrs_cv,nxy,1))
    # Calculate predictors and classification targets
    apcp_efi = -1.+(2./np.pi)*np.mean(np.arccos(1.-2.*apcp_ens_pit),axis=3).reshape((ndts*nyrs_cv,nxy,1))
    predictors = np.concatenate((lon_nml,lat_nml,-1.+2.*apcp_pop_cl,apcp_efi),axis=2)
    logp_cl = np.concatenate((np.log(1.-apcp_pop_cl),np.repeat(np.log(apcp_pop_cl),ncat-1,axis=2)-np.log(ncat-1)),axis=2)
    # perform 5-fold cross validation to find optimal regularization
    date_order = np.arange(ndts*nyrs_cv).reshape(ndts,nyrs_cv).T.flatten()
    cv_ind = date_order[np.arange(ndts*nyrs_cv)%232<231]                        # remove the date between the 5 cross-validated blocks
    valid_score = np.zeros((nreg,5), dtype=np.float32)
    valid_crps = np.zeros((nreg,5), dtype=np.float32)
    for cvi in range(5):
        train_ind = cv_ind[np.arange(len(cv_ind))//(len(cv_ind)//5)!=cvi]
        valid_ind = cv_ind[np.arange(len(cv_ind))//(len(cv_ind)//5)==cvi]
        predictors_train = predictors[train_ind,:,:].reshape((-1,predictors.shape[-1]))
        logp_cl_train = logp_cl[train_ind,:,:].reshape((-1,ncat))
        cat_targets_train = apcp_obs_cat[apcp_obs_ind.flatten()[train_ind],:,:].reshape((-1,ncat)).astype(float)
        predictors_valid = predictors[valid_ind,:,:].reshape((-1,predictors.shape[-1]))
        logp_cl_valid = logp_cl[valid_ind,:,:].reshape((-1,ncat))
        cat_targets_valid = apcp_obs_cat[apcp_obs_ind.flatten()[valid_ind],:,:].reshape((-1,ncat)).astype(float)
        doy_valid = np.repeat(doy_dts[:,np.newaxis],nyrs_cv,axis=1).flatten()[valid_ind]
        for ireg in range(nreg):
            # Define and fit ANN model (using batch gradient descent)
            keras.backend.clear_session()
            model = build_cat_model(predictors.shape[-1], mod[imod], ncat, reg[ireg])
            model.compile(optimizer=Adam(0.05), loss=modified_categorical_crossentropy)
            model.fit([predictors_train,logp_cl_train], cat_targets_train, epochs=100, batch_size=len(train_ind)*nxy, verbose=0)
            valid_score[ireg,cvi] = model.evaluate([predictors_valid,logp_cl_valid], cat_targets_valid, batch_size=len(train_ind)*nxy, verbose=0)
            # Calculate CRPS for each cross-validation fold
            prob_fcst_cat = model.predict([predictors_valid,logp_cl_valid]).reshape((len(valid_ind),nxy,ncat))
            prob_fcst_chf = -np.log(1.-np.cumsum(prob_fcst_cat,axis=2)[:,:,:(ncat-1)])
            crps_fold = np.zeros((len(valid_ind),nxy),dtype=np.float32)
            for ivdt in range(len(valid_ind)):
                for ixy in range(nxy):
                    itp_fct = interp1d(thr_doy[doy_valid[ivdt],ixy,:], prob_fcst_chf[ivdt,ixy,:], kind='linear',fill_value='extrapolate')
                    bs = (1.-np.exp(-itp_fct(x))-1.*(apcp_obs[apcp_obs_ind.flatten()[valid_ind[ivdt]],ixy]<=x))**2
                    crps_fold[ivdt,ixy] = 0.5*np.sum((bs[1:]+bs[:len(dx)])*dx)
            valid_crps[ireg,cvi] = np.mean(crps_fold)
    opt_reg_ind = np.argmin(np.mean(valid_score,axis=1))
    opt_reg_param[iyr] = reg[opt_reg_ind]
    opt_valid_scores[iyr,:] = valid_score[opt_reg_ind,:]
    opt_valid_crps[iyr,:] =  valid_crps[opt_reg_ind,:]


### Save out to file
outfilename = "/home/michael/Desktop/CalifAPCP/tuning/efi-"+ncl+"cl-m"+str(imod)+"-l1_"+clead
np.savez(outfilename, opt_reg_param=opt_reg_param, opt_valid_scores=opt_valid_scores, opt_valid_crps=opt_valid_crps)





