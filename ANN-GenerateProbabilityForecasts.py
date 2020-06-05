
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

from keras import models
from keras import layers
from keras import regularizers

from keras.layers import Input, Dense, Add, Activation, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam

#plt.ion()



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


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
#list(f1)
lat = f1['obs_lat']
lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
pop_doy = f1['pop_doy']
thr_doy = f1['thr_doy']
qtev_doy = f1['qtev_doy']
apcp_obs_cat = f1['apcp_obs_cat']
f1.close()

ncat = apcp_obs_cat.shape[2]



clead = 'week2'

f3 = np.load("/home/michael/Desktop/CalifAPCP/tuning/efi-20cl-m0-l1_"+clead+".npz")
opt_reg_param = f3['opt_reg_param']
f3.close()


for iyr in range(20):
    print(iyr)
    # Load smoothed ensemble forecast PIT values
    f2 = np.load("/home/michael/Desktop/CalifAPCP/stats/ensemble_stats_"+clead+"_ANN_yr"+str(iyr)+".npz")
    doy_dts = f2['doy_dts']
    apcp_obs_ind_train = f2['apcp_obs_ind_train']
    apcp_obs_ind_verif = f2['apcp_obs_ind_verif']
    apcp_ens_pit_train = f2['apcp_ens_pit_train']
    apcp_ens_pit_verif = f2['apcp_ens_pit_verif']
    f2.close()
    ndts, nyrs_tr, nxy, nmem = apcp_ens_pit_train.shape
    # Calculate normalized coordinates and climatological probability of precipitation
    lon_train = np.repeat(-1.+2.*(lon[np.newaxis,:]-lon[0])/(lon[-1]-lon[0]),ndts*nyrs_tr,axis=0).reshape((ndts,nyrs_tr,nxy,1))
    lon_verif = np.repeat(-1.+2.*(lon[np.newaxis,:]-lon[0])/(lon[-1]-lon[0]),ndts,axis=0).reshape((ndts,nxy,1))
    lat_train = np.repeat(-1.+2.*(lat[np.newaxis,:]-lat[-1])/(lat[0]-lat[-1]),ndts*nyrs_tr,axis=0).reshape((ndts,nyrs_tr,nxy,1))
    lat_verif = np.repeat(-1.+2.*(lat[np.newaxis,:]-lat[-1])/(lat[0]-lat[-1]),ndts,axis=0).reshape((ndts,nxy,1))
    apcp_pop_cl_train = np.repeat(pop_doy[doy_dts,np.newaxis,:,None],nyrs_tr,axis=1)
    apcp_pop_cl_verif = pop_doy[doy_dts,:,None]
    # Calculate predictors and classification targets
    apcp_efi_train = -1.+(2./np.pi)*np.mean(np.arccos(1.-2.*apcp_ens_pit_train),axis=3)[:,:,:,None]
    apcp_efi_verif = -1.+(2./np.pi)*np.mean(np.arccos(1.-2.*apcp_ens_pit_verif),axis=2)[:,:,None]
    train_predictors = np.concatenate((lon_train,lat_train,apcp_efi_train),axis=3).reshape((-1,3))
    train_logp_cl = np.concatenate((np.log(1.-apcp_pop_cl_train),np.repeat(np.log(apcp_pop_cl_train),ncat-1,axis=3)-np.log(ncat-1)),axis=3).reshape((-1,ncat))
    train_cat_targets = apcp_obs_cat[apcp_obs_ind_train.flatten(),:,:].reshape((-1,ncat)).astype(float)
    # Define and fit ANN model
    keras.backend.clear_session()
    model = build_cat_model(train_predictors.shape[-1], [10], ncat, opt_reg_param[iyr])
    model.compile(optimizer=Adam(0.05), loss=modified_categorical_crossentropy)
    model.fit([train_predictors,train_logp_cl], train_cat_targets, epochs=100, batch_size=ndts*nyrs_tr*nxy, verbose=0)
    # Calculate probability forecasts
    verif_predictors = np.concatenate((lon_verif,lat_verif,apcp_efi_verif),axis=2).reshape((-1,3))
    verif_logp_cl = np.concatenate((np.log(1.-apcp_pop_cl_verif),np.repeat(np.log(apcp_pop_cl_verif),ncat-1,axis=2)-np.log(ncat-1)),axis=2).reshape((-1,ncat))
    prob_fcst_cat = model.predict([verif_predictors,verif_logp_cl]).reshape((ndts,nxy,ncat))
    ### Save out to file
    outfilename = "/home/michael/Desktop/CalifAPCP/forecasts/ANN-efi/probfcst_10-l1_"+clead+"_yr"+str(iyr)
    np.savez(outfilename, prob_fcst_cat=prob_fcst_cat)




