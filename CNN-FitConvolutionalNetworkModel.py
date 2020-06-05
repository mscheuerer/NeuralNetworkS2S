
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

from keras import models
from keras import layers
from keras import regularizers

from keras.layers import Input, Dense, Dot, Add, Activation, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from keras.models import Model
from keras.optimizers import Adam

#plt.ion()


##  Load categorical analysis data

f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
#list(f1)
lat = f1['obs_lat']
lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
pop_doy = f1['pop_doy']
apcp_obs_cat = f1['apcp_obs_cat']
f1.close()

ndts, nxy, ncat = apcp_obs_cat.shape



##  Load ERA5 z500 and tcw fields, subset to 22 x 18 image, same for the ensemble forecast fields

ixl = 10
ixu = -6
jyl = 6
jyu = -6

f2 = np.load("/home/michael/Desktop/CalifAPCP/data/z500_tcw_predictors_era5.npz")
era5_dates_ord = f2['dates_ord']
era5_lon = f2['longitude'][ixl:ixu]
era5_lat = f2['latitude'][jyl:jyu]
z500 = f2['z500_1wk'][:,:,jyl:jyu,ixl:ixu]
tcw = f2['tcw_1wk'][:,:,jyl:jyu,ixl:ixu]
f2.close()

ndts, nyrs, ny, nx = z500.shape


z500_fcst = np.zeros((3,ndts,nyrs,11,ny,nx),dtype=np.float32)
tcw_fcst = np.zeros((3,ndts,nyrs,11,ny,nx),dtype=np.float32)

f3 = np.load("/home/michael/Desktop/CalifAPCP/data/z500_predictor_cnn.npz")
mod_dates_ord = f3['mod_dates_ord'][:,:,6:21:7]

f4 = np.load("/home/michael/Desktop/CalifAPCP/data/tcw_predictor_cnn.npz")

for ilt in range(3):
    clead = 'week'+str(ilt+2)
    z500_fcst[ilt,:,:,:,:,:] = f3['z500_'+clead][:,:,:,jyl:jyu,ixl:ixu]      # subset to 22 x 18 image
    tcw_fcst[ilt,:,:,:,:,:] = f4['tcw_'+clead][:,:,:,jyl:jyu,ixl:ixu]

f3.close()
f4.close()



## Calculate doy for each analysis date and for each forecast valid date

doy_dts = np.zeros(ndts,dtype=np.int32)
apcp_obs_ind = np.zeros((ndts,nyrs),dtype=np.int32)
for idt in range(ndts):
    for iyr in range(nyrs):
        apcp_obs_ind[idt,iyr] = np.where(obs_dates_ord==era5_dates_ord[idt,iyr])[0][0]
    date_ord = int(era5_dates_ord[idt,0]-0.5)
    doy_dts[idt] = min(364,(datetime.date.fromordinal(date_ord)-datetime.date(datetime.date.fromordinal(date_ord).year,1,1)).days)

doy_fcst = np.zeros((3,ndts),dtype=np.int32)
for idt in range(ndts):
    for ilt in range(3):
        date_ord = int(int(mod_dates_ord[idt,0,ilt])-0.5)
        doy_fcst[ilt,idt] = min(364,(datetime.date.fromordinal(date_ord)-datetime.date(datetime.date.fromordinal(date_ord).year,1,1)).days)



##  Normalize tcw to 10th/90th climatological percentiles at each grid point

tcw_q10 = np.percentile(tcw,10,axis=1)
tcw_q90 = np.percentile(tcw,90,axis=1)
tcw_q10_sm = np.zeros(tcw_q10.shape, dtype=np.float32)
tcw_q90_sm = np.zeros(tcw_q90.shape, dtype=np.float32)

tcw_fcst_q10 = np.percentile(tcw_fcst,10,axis=(2,3))
tcw_fcst_q90 = np.percentile(tcw_fcst,90,axis=(2,3))
tcw_fcst_q10_sm = np.zeros(tcw_fcst_q10.shape, dtype=np.float32)
tcw_fcst_q90_sm = np.zeros(tcw_fcst_q90.shape, dtype=np.float32)

X = np.ones((ndts,3), dtype=np.float32)                  # Fit harmonic function to annual cycle of tcw climatology
X[:,1] = np.sin(2.*np.pi*era5_dates_ord[:,0]/365.25)
X[:,2] = np.cos(2.*np.pi*era5_dates_ord[:,0]/365.25)

for ix in range(nx):
    for jy in range(ny):
        coef_q10 = solve(np.matmul(X.T,X),np.matmul(X.T,tcw_q10[:,jy,ix]))
        tcw_q10_sm[:,jy,ix] = np.matmul(X,coef_q10)
        coef_q90 = solve(np.matmul(X.T,X),np.matmul(X.T,tcw_q90[:,jy,ix]))
        tcw_q90_sm[:,jy,ix] = np.matmul(X,coef_q90)
        for ilt in range(3):
            coef_q10 = solve(np.matmul(X.T,X),np.matmul(X.T,tcw_fcst_q10[ilt,:,jy,ix]))
            tcw_fcst_q10_sm[ilt,:,jy,ix] = np.matmul(X,coef_q10)
            coef_q90 = solve(np.matmul(X.T,X),np.matmul(X.T,tcw_fcst_q90[ilt,:,jy,ix]))
            tcw_fcst_q90_sm[ilt,:,jy,ix] = np.matmul(X,coef_q90)

tcw_ano = -1.+2.*(tcw-tcw_q10_sm[:,None,:,:])/(tcw_q90_sm-tcw_q10_sm)[:,None,:,:]
tcw_fcst_ano = -1.+2.*(tcw_fcst-tcw_fcst_q10_sm[:,:,None,None,:,:])/(tcw_fcst_q90_sm-tcw_fcst_q10_sm)[:,:,None,None,:,:]



##  Normalize z500 to 1st/99th climatological percentiles across all grid points

z500_q01 = np.percentile(z500,1,axis=(1,2,3))
z500_q99 = np.percentile(z500,99,axis=(1,2,3))
z500_fcst_q01 = np.percentile(z500_fcst,1,axis=(2,3,4,5))
z500_fcst_q99 = np.percentile(z500_fcst,99,axis=(2,3,4,5))

coef_q01 = solve(np.matmul(X.T,X),np.matmul(X.T,z500_q01))
z500_q01_sm = np.matmul(X,coef_q01)
coef_q99 = solve(np.matmul(X.T,X),np.matmul(X.T,z500_q99))
z500_q99_sm = np.matmul(X,coef_q99)

z500_fcst_q01_sm = np.zeros(z500_fcst_q01.shape, dtype=np.float32)
z500_fcst_q99_sm = np.zeros(z500_fcst_q99.shape, dtype=np.float32)

for ilt in range(3):
    coef_q01 = solve(np.matmul(X.T,X),np.matmul(X.T,z500_fcst_q01[ilt,:]))
    z500_fcst_q01_sm[ilt,:] = np.matmul(X,coef_q01)
    coef_q99 = solve(np.matmul(X.T,X),np.matmul(X.T,z500_fcst_q99[ilt,:]))
    z500_fcst_q99_sm[ilt,:] = np.matmul(X,coef_q99)

z500_ano = -1.+2.*(z500-z500_q01_sm[:,None,None,None])/(z500_q99_sm-z500_q01_sm)[:,None,None,None]
z500_fcst_ano = -1.+2.*(z500_fcst-z500_fcst_q01_sm[:,:,None,None,None,None])/(z500_fcst_q99_sm-z500_fcst_q01_sm)[:,:,None,None,None,None]


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
    inp_imgs = Input(shape=(18,22,2,))
    #inp_imgs = Input(shape=(18,22,1,))
    inp_basis = Input(shape=(n_xy,n_basis,))
    inp_cl = Input(shape=(n_xy,n_bins,))
    c = Conv2D(4, (3,3), activation='elu')(inp_imgs)
    c = MaxPooling2D((2,2))(c)
    c = Conv2D(8, (3,3), activation='elu')(c)
    c = MaxPooling2D((2,2))(c)
    x = Flatten()(c)
    for h in hidden_nodes:
        x = Dropout(dropout_rate)(x)
        x = Dense(h, activation='elu')(x)
    x = Dense(n_bins*n_basis, activation='elu')(x)
    x = Reshape((n_bins,n_basis))(x)
    z = Dot(axes=2)([inp_basis, x])     # Tensor product with basis functions
    z = Add()([z, inp_cl])              # Add (log) probability anomalies to log climatological probabilities 
    out = Activation('softmax')(z)
    return Model(inputs=[inp_imgs, inp_basis, inp_cl], outputs=out)


def modified_categorical_crossentropy(y_mat, prob_fcst):
    prob_obs_cat = K.sum(y_mat*prob_fcst,axis=2)
    return -K.mean(K.log(prob_obs_cat))



imod = 0

mod = [[10],[20],[10,10]]

f5 = np.load("/home/michael/Desktop/CalifAPCP/tuning/cnn-m"+str(imod)+"-drpt-f48.npz")
opt_reg_param = f5['opt_reg_param']
f5.close()


for iyr in range(0,20):
    print(iyr)
    # Split data into training and verification data set
    apcp_obs_ind_train = np.delete(apcp_obs_ind,iyr,axis=1)
    apcp_obs_ind_verif = apcp_obs_ind[:,iyr]
    z500_pred_train = np.delete(z500_ano,iyr,axis=1).reshape((ndts*(nyrs-1),ny,nx,1))
    z500_pred_verif = z500_ano[:,iyr,:,:,None]
    z500_pred_fcst_train = np.delete(z500_fcst_ano,iyr,axis=2).reshape((3,ndts*(nyrs-1),11,ny,nx,1))
    z500_pred_fcst_verif = z500_fcst_ano[:,:,iyr,:,:,:,None]
    tcw_pred_train = np.delete(tcw_ano,iyr,axis=1).reshape((ndts*(nyrs-1),ny,nx,1))
    tcw_pred_verif = tcw_ano[:,iyr,:,:,None]
    tcw_pred_fcst_train = np.delete(tcw_fcst_ano,iyr,axis=2).reshape((3,ndts*(nyrs-1),11,ny,nx,1))
    tcw_pred_fcst_verif = tcw_fcst_ano[:,:,iyr,:,:,:,None]
    # Calculate climatological log probabilities for each class
    apcp_lgp0_cl_train = np.repeat(np.log(1.-pop_doy[doy_dts,np.newaxis,:]),nyrs-1,axis=1).reshape((ndts*(nyrs-1),nxy,1))
    apcp_lgp0_cl_verif = np.log(1.-pop_doy[doy_dts,:])[:,:,None]
    apcp_lgpop_cl_train = np.repeat(np.log(pop_doy[doy_dts,np.newaxis,:]),nyrs-1,axis=1).reshape((ndts*(nyrs-1),nxy,1))
    apcp_lgpop_cl_verif = np.log(pop_doy[doy_dts,:])[:,:,None]
    apcp_lgp0_cl_fcst_train = np.zeros((3,ndts*(nyrs-1),nxy,1), dtype=np.float32)
    apcp_lgp0_cl_fcst_verif = np.zeros((3,ndts,nxy,1), dtype=np.float32)
    apcp_lgpop_cl_fcst_train = np.zeros((3,ndts*(nyrs-1),nxy,1), dtype=np.float32)
    apcp_lgpop_cl_fcst_verif = np.zeros((3,ndts,nxy,1), dtype=np.float32)
    for ilt in range(3):
        apcp_lgp0_cl_fcst_train[ilt,:,:,0] = np.repeat(np.log(1.-pop_doy[doy_fcst[ilt,:],np.newaxis,:]),nyrs-1,axis=1).reshape((ndts*(nyrs-1),nxy))
        apcp_lgp0_cl_fcst_verif[ilt,:,:,0] = np.log(1.-pop_doy[doy_fcst[ilt,:],:])
        apcp_lgpop_cl_fcst_train[ilt,:,:,0] = np.repeat(np.log(pop_doy[doy_fcst[ilt,:],np.newaxis,:]),nyrs-1,axis=1).reshape((ndts*(nyrs-1),nxy))
        apcp_lgpop_cl_fcst_verif[ilt,:,:,0] = np.log(pop_doy[doy_fcst[ilt,:],:])
    # Compose training data (large-scale predictors, auxiliary predictors, climatological probabilities, observed categories)
    train_pred_imgs = np.concatenate((z500_pred_train,tcw_pred_train),axis=3)
    #train_pred_imgs = tcw_pred_train
    train_basis = np.repeat(basis[np.newaxis,:,:],ndts*(nyrs-1),axis=0)
    train_logp_cl = np.concatenate((apcp_lgp0_cl_train,np.repeat(apcp_lgpop_cl_train,ncat-1,axis=2)-np.log(ncat-1)),axis=2)
    train_cat_targets = apcp_obs_cat[apcp_obs_ind_train.flatten(),:,:].astype(float)
    # Define and fit CNN model
    keras.backend.clear_session()
    model = build_cat_model(nxy, ncat, nbs, mod[imod], opt_reg_param[iyr])
    model.compile(optimizer=Adam(0.01), loss=modified_categorical_crossentropy)
    model.fit([train_pred_imgs,train_basis,train_logp_cl], train_cat_targets, epochs=150, batch_size=ndts*(nyrs-1), verbose=1)
    # Calculate ERA-5 probability forecasts
    verif_pred_imgs = np.concatenate((z500_pred_verif,tcw_pred_verif),axis=3)
    #verif_pred_imgs = tcw_pred_verif
    verif_basis = np.repeat(basis[np.newaxis,:,:],ndts,axis=0)
    verif_logp_cl = np.concatenate((apcp_lgp0_cl_verif,np.repeat(apcp_lgpop_cl_verif,ncat-1,axis=2)-np.log(ncat-1)),axis=2)
    prob_fcst_cat_era5 = model.predict([verif_pred_imgs,verif_basis,verif_logp_cl])
    # Calculate ensemble-based, mean probability forecasts
    logp_ano_ensmean_train = np.zeros((3,ndts*(nyrs-1),nxy,ncat), dtype=np.float32)
    logp_ano_ensmean_verif = np.zeros((3,ndts,nxy,ncat), dtype=np.float32)
    for ilt in range(3):
        train_logp_cl = np.concatenate((apcp_lgp0_cl_fcst_train[ilt,:,:,:],np.repeat(apcp_lgpop_cl_fcst_train[ilt,:,:,:],ncat-1,axis=2)-np.log(ncat-1)),axis=2)
        verif_logp_cl = np.concatenate((apcp_lgp0_cl_fcst_verif[ilt,:,:,:],np.repeat(apcp_lgpop_cl_fcst_verif[ilt,:,:,:],ncat-1,axis=2)-np.log(ncat-1)),axis=2)
        prob_fcst_cat_ens_train = np.zeros((11,ndts*(nyrs-1),nxy,ncat), dtype=np.float32)
        prob_fcst_cat_ens_verif = np.zeros((11,ndts,nxy,ncat), dtype=np.float32)
        for imem in range(11):
            train_pred_imgs = np.concatenate((z500_pred_fcst_train[ilt,:,imem,:,:,:],tcw_pred_fcst_train[ilt,:,imem,:,:,:]),axis=3)
            #train_pred_imgs = tcw_pred_fcst_train[ilt,:,imem,:,:,:]
            prob_fcst_cat_ens_train[imem,:,:,:] = model.predict([train_pred_imgs,train_basis,train_logp_cl])
            verif_pred_imgs = np.concatenate((z500_pred_fcst_verif[ilt,:,imem,:,:,:],tcw_pred_fcst_verif[ilt,:,imem,:,:,:]),axis=3)
            #verif_pred_imgs = tcw_pred_fcst_verif[ilt,:,imem,:,:,:]
            prob_fcst_cat_ens_verif[imem,:,:,:] = model.predict([verif_pred_imgs,verif_basis,verif_logp_cl])
        logp_ano_ensmean_train[ilt,:,:,:] = np.mean(np.log(prob_fcst_cat_ens_train),axis=0) - train_logp_cl     # Reconstruct the log probability anomalies
        logp_ano_ensmean_verif[ilt,:,:,:] = np.mean(np.log(prob_fcst_cat_ens_verif),axis=0) - verif_logp_cl     #  for each ensemble member and calculate mean
    ### Save out to file
    outfilename = "/home/michael/Desktop/CalifAPCP/forecasts/CNN/probfcst_cnn-m"+str(imod)+"-drpt-f48_yr"+str(iyr)
    np.savez(outfilename, prob_fcst_cat_era5=prob_fcst_cat_era5, \
                 logp_ano_ensmean_train=logp_ano_ensmean_train, \
                 logp_ano_ensmean_verif=logp_ano_ensmean_verif, \
                 apcp_lgp0_cl_fcst_train=apcp_lgp0_cl_fcst_train, \
                 apcp_lgp0_cl_fcst_verif=apcp_lgp0_cl_fcst_verif, \
                 apcp_lgpop_cl_fcst_train=apcp_lgpop_cl_fcst_train, \
                 apcp_lgpop_cl_fcst_verif=apcp_lgpop_cl_fcst_verif)












