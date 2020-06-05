
import numpy as np
import scipy as sp
import math
import os, sys
#import matplotlib.pyplot as plt
import datetime
import time

from netCDF4 import Dataset
from numpy import ma
from scipy import stats

from scipy.optimize import minimize_scalar


#plt.ion()

def adjustment_factor_target(par, y_mat, logp_ensmeanano, logp_cl):
    # average modified categorical crossentropy for relaxed perfect prog probabilities
    prob_fcst_cat_cmb = np.exp(par*logp_ensmeanano+logp_cl)
    prob_fcst = prob_fcst_cat_cmb / np.sum(prob_fcst_cat_cmb,axis=2)[:,:,None]
    prob_obs_cat = np.sum(y_mat*prob_fcst,axis=2)
    return -np.mean(np.log(prob_obs_cat))


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/categorical_precip_obs_20cl.npz")
#list(f1)
lat = f1['obs_lat']
lon = f1['obs_lon']
obs_dates_ord = f1['obs_dates_ord']
apcp_obs_cat = f1['apcp_obs_cat']
f1.close()

ndts, nxy, ncat = apcp_obs_cat.shape


f2 = np.load("/home/michael/Desktop/CalifAPCP/data/z500_predictor_cnn.npz")
mod_dates_ord = f2['mod_dates_ord'][:,:,6:21:7]
f2.close()

ndts, nyrs, nlt = mod_dates_ord.shape

apcp_obs_ind = np.zeros((ndts,nyrs,nlt),dtype=np.int32)
for idt in range(ndts):
    for iyr in range(nyrs):
        for ilt in range(3):
            apcp_obs_ind[idt,iyr,ilt] = np.where(obs_dates_ord==mod_dates_ord[idt,iyr,ilt])[0][0]


imod = 0

for iyr in range(0,20):
    print(iyr)
    # Load smoothed ensemble forecast anomalies 
    f3 = np.load("/home/michael/Desktop/CalifAPCP/forecasts/CNN/probfcst_cnn-m"+str(imod)+"-drpt-f48_yr"+str(iyr)+".npz")
    logp_ano_ensmean_train = f3['logp_ano_ensmean_train']
    logp_ano_ensmean_verif = f3['logp_ano_ensmean_verif']
    apcp_lgp0_cl_fcst_train = f3['apcp_lgp0_cl_fcst_train']
    apcp_lgp0_cl_fcst_verif = f3['apcp_lgp0_cl_fcst_verif']
    apcp_lgpop_cl_fcst_train = f3['apcp_lgpop_cl_fcst_train']
    apcp_lgpop_cl_fcst_verif = f3['apcp_lgpop_cl_fcst_verif']
    f3.close()
    for ilt in range(3):
        # Calculate index for training observations
        apcp_obs_ind_train = np.delete(apcp_obs_ind[:,:,ilt],iyr,axis=1)
        train_cat_targets = apcp_obs_cat[apcp_obs_ind_train.flatten(),:,:].astype(float)
        train_logp_cl = np.concatenate((apcp_lgp0_cl_fcst_train[ilt,:,:,:],np.repeat(apcp_lgpop_cl_fcst_train[ilt,:,:,:],ncat-1,axis=2)-np.log(ncat-1)),axis=2)
        verif_logp_cl = np.concatenate((apcp_lgp0_cl_fcst_verif[ilt,:,:,:],np.repeat(apcp_lgpop_cl_fcst_verif[ilt,:,:,:],ncat-1,axis=2)-np.log(ncat-1)),axis=2)
        train_logp_ensmeanano = logp_ano_ensmean_train[ilt,:,:,:]
        verif_logp_ensmeanano = logp_ano_ensmean_verif[ilt,:,:,:]
        a = minimize_scalar(adjustment_factor_target, args=(train_cat_targets,train_logp_ensmeanano,train_logp_cl), method='bounded', bounds=(0.,1.)).x
        print(a)
        prob_fcst_cat_cmb = np.exp(a*verif_logp_ensmeanano+verif_logp_cl)
        prob_fcst_cat = prob_fcst_cat_cmb / np.sum(prob_fcst_cat_cmb,axis=2)[:,:,None]
        ### Save out to file
        outfilename = "/home/michael/Desktop/CalifAPCP/forecasts/CNN/probfcst_cnn-m"+str(imod)+"-drpt-f48_week"+str(2+ilt)+"_yr"+str(iyr)
        np.savez(outfilename, prob_fcst_cat=prob_fcst_cat)





