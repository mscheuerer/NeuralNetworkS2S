
import numpy as np
import numpy.ma as ma
import scipy as sp
import math
import os, sys
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches

import datetime
import time
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from netCDF4 import Dataset
from numpy import ma, loadtxt
from numpy.linalg import solve
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import kendalltau
from colorspace import diverging_hcl, sequential_hcl

plt.ion()

divcmp = diverging_hcl("Tropic",rev=True).cmap(name = "Diverging Color Map")


f1 = np.load("/home/michael/Desktop/CalifAPCP/data/precip_PRISM_cal_19810101_20171231.npz")
obs_lat = f1['lat']
obs_lon = f1['lon']
f1.close()

nxy = len(obs_lon)

ndts = 61
nyrs = 20


###################################################################################################
#                                                                                                 #
#  Figure S1:  Maps of RPSS comparing different CSGD implementations                              #
#                                                                                                 #
###################################################################################################


acfRv1 = np.zeros((3,15),dtype=np.float32)
acfRv2 = np.zeros((3,15),dtype=np.float32)
pvalRv1 = np.zeros((3,nxy),dtype=np.float32)
pvalRv2 = np.zeros((3,nxy),dtype=np.float32)
alphaFDRrv1 = np.zeros(3,dtype=np.float32)
alphaFDRrv2 = np.zeros(3,dtype=np.float32)

rpssMapCSGD = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)
rpssMapCSGDrv1 = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)
rpssMapCSGDrv2 = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)

rpssAvgCSGD = ma.array(np.zeros(3,dtype=np.float32),mask=True)
rpssAvgCSGDrv1 = ma.array(np.zeros(3,dtype=np.float32),mask=True)
rpssAvgCSGDrv2 = ma.array(np.zeros(3,dtype=np.float32),mask=True)

for ilead in range(3):
    f1 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-ann_week"+str(ilead+2)+".npz")
    Bs33Clm = f1['Bs33pClm']
    Bs33CSGD = f1['Bs33pCSGD']
    Bs67Clm = f1['Bs67pClm']
    Bs67CSGD = f1['Bs67pCSGD']
    Bs85Clm = f1['Bs85pClm']
    Bs85CSGD = f1['Bs85pCSGD']
    f1.close()
    f2 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-rv1_week"+str(ilead+2)+".npz")
    Bs33CSGDrv1 = f2['Bs33pCSGD']
    Bs67CSGDrv1 = f2['Bs67pCSGD']
    Bs85CSGDrv1 = f2['Bs85pCSGD']
    f2.close()
    f3 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-rv2_week"+str(ilead+2)+".npz")
    Bs33CSGDrv2 = f3['Bs33pCSGD']
    Bs67CSGDrv2 = f3['Bs67pCSGD']
    Bs85CSGDrv2 = f3['Bs85pCSGD']
    f3.close()
    rpsClm = Bs33Clm + Bs67Clm + Bs85Clm       # calculate ranked probability score
    rpsCSGD = Bs33CSGD + Bs67CSGD + Bs85CSGD
    rpsCSGDrv1 = Bs33CSGDrv1 + Bs67CSGDrv1 + Bs85CSGDrv1
    rpsCSGDrv2 = Bs33CSGDrv2 + Bs67CSGDrv2 + Bs85CSGDrv2
    rpssMapCSGD[ilead,:] = 1.-np.sum(rpsCSGD,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssMapCSGDrv1[ilead,:] = 1.-np.sum(rpsCSGDrv1,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssMapCSGDrv2[ilead,:] = 1.-np.sum(rpsCSGDrv2,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssAvgCSGD[ilead] = 1.-np.sum(rpsCSGD)/np.sum(rpsClm)
    rpssAvgCSGDrv1[ilead] = 1.-np.sum(rpsCSGDrv1)/np.sum(rpsClm)
    rpssAvgCSGDrv2[ilead] = 1.-np.sum(rpsCSGDrv2)/np.sum(rpsClm)
    rpsDiffCSGDrv1 = rpsCSGD-rpsCSGDrv1
    rpsDiffCSGDrv2 = rpsCSGD-rpsCSGDrv2
    rpsDiffStdzCSGDrv1 = (rpsDiffCSGDrv1-np.mean(rpsDiffCSGDrv1,axis=(0,1))[None,None,:])/np.std(rpsDiffCSGDrv1,axis=(0,1))[None,None,:]
    rpsDiffStdzCSGDrv2 = (rpsDiffCSGDrv2-np.mean(rpsDiffCSGDrv2,axis=(0,1))[None,None,:])/np.std(rpsDiffCSGDrv2,axis=(0,1))[None,None,:]
    for lg in range(15):
        acfRv1[ilead,lg] = np.mean(rpsDiffStdzCSGDrv1[lg:,:,:]*rpsDiffStdzCSGDrv1[:(ndts-lg),:,:])         # Estimate temporal autocorrelation
        acfRv2[ilead,lg] = np.mean(rpsDiffStdzCSGDrv2[lg:,:,:]*rpsDiffStdzCSGDrv2[:(ndts-lg),:,:])
    rhoCSGDrv1 = acfRv1[ilead,1]/acfRv1[ilead,0]
    rhoCSGDrv2 = acfRv2[ilead,1]/acfRv2[ilead,0]
    print(rhoCSGDrv1,rhoCSGDrv2)
    nCSGDrv1 = round(ndts*nyrs*(1-rhoCSGDrv1)/(1+rhoCSGDrv1))
    nCSGDrv2 = round(ndts*nyrs*(1-rhoCSGDrv2)/(1+rhoCSGDrv2))
    for ixy in range(nxy):
        smplCSGDrv1 = rpsCSGD[:,:,ixy].flatten()-rpsCSGDrv1[:,:,ixy].flatten()
        smplCSGDrv2 = rpsCSGD[:,:,ixy].flatten()-rpsCSGDrv2[:,:,ixy].flatten()
        tstatCSGDrv1 = np.mean(smplCSGDrv1)/np.sqrt(np.var(smplCSGDrv1)/nCSGDrv1)        # test statistic for paired t-test
        tstatCSGDrv2 = np.mean(smplCSGDrv2)/np.sqrt(np.var(smplCSGDrv2)/nCSGDrv2)
        pvalRv1[ilead,ixy] = 1.-sp.stats.t.cdf(tstatCSGDrv1,df=nCSGDrv1-1)       # p-value for one-sided test
        pvalRv2[ilead,ixy] = 1.-sp.stats.t.cdf(tstatCSGDrv2,df=nCSGDrv2-1)
        #pval[ilead,ixy] = 2*min(1.-sp.stats.t.cdf(tstat,df=n-1),sp.stats.t.cdf(tstat,df=n-1))
    pvalRv1_srt = np.sort(pvalRv1[ilead,:])
    iCSGDrv1 = np.where(pvalRv1_srt<=0.1*np.arange(1,nxy+1)/nxy)[0]
    if len(iCSGDrv1)>0:
        alphaFDRrv1[ilead] = pvalRv1_srt[iCSGDrv1[-1]]
    pvalRv2_srt = np.sort(pvalRv2[ilead,:])
    iCSGDrv2 = np.where(pvalRv2_srt<=0.1*np.arange(1,nxy+1)/nxy)[0]
    if len(iCSGDrv2)>0:
        alphaFDRrv2[ilead] = pvalRv2_srt[iCSGDrv2[-1]]
    plt.figure(); plt.scatter(np.arange(663),0.1*np.arange(1,664)/663); plt.scatter(np.arange(663),pvalRv1_srt); plt.scatter(np.arange(663),pvalRv2_srt)




fig = plt.figure(figsize=(11.3,9.))

for ilead in range(3):
    ylim = np.array([0.26,0.052,0.026])[ilead]
    #ylim = np.amax(abs(rpssMapCSGD[ilead,:]))
    indSgnfRv1 = (pvalRv1[ilead,:]<alphaFDRrv1[ilead])
    indSgnfRv2 = (pvalRv2[ilead,:]<alphaFDRrv2[ilead])
    ax1 = fig.add_subplot(3,3,ilead+1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapCSGD[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgCSGD[ilead].round(3),fontsize=12)
    plt.title("RPSS - CSGD (week "+str(ilead+2)+")",fontsize=14)
    ax2 = fig.add_subplot(3,3,ilead+4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapCSGDrv1[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.scatter(obs_lon[indSgnfRv1],obs_lat[indSgnfRv1],c=rpssMapCSGDrv1[ilead,indSgnfRv1],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=19.75,lw=0.8,edgecolors=[.2,.2,.2])
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgCSGDrv1[ilead].round(3),fontsize=12)
    plt.title("RPSS - CSGD w stdz.",fontsize=14)
    ax3 = fig.add_subplot(3,3,ilead+7)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapCSGDrv2[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.scatter(obs_lon[indSgnfRv2],obs_lat[indSgnfRv2],c=rpssMapCSGDrv2[ilead,indSgnfRv2],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=19.75,lw=0.8,edgecolors=[.2,.2,.2])
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgCSGDrv2[ilead].round(3),fontsize=12)
    plt.title("RPSS - CSGD w MD pred.",fontsize=14)

plt.tight_layout()




###################################################################################################
#                                                                                                 #
#  Figure S2:  Maps of RPSS comparing different ANN implementations                               #
#                                                                                                 #
###################################################################################################


acfRv = np.zeros((3,15),dtype=np.float32)
pvalRv = np.zeros((3,nxy),dtype=np.float32)
alphaFDRrv = np.zeros(3,dtype=np.float32)

rpssMapANN = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)
rpssMapANNrv = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)

rpssAvgANN = ma.array(np.zeros(3,dtype=np.float32),mask=True)
rpssAvgANNrv = ma.array(np.zeros(3,dtype=np.float32),mask=True)

for ilead in range(3):
    f1 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-ann_week"+str(ilead+2)+".npz")
    Bs33Clm = f1['Bs33pClm']
    Bs33ANN = f1['Bs33pANN']
    Bs67Clm = f1['Bs67pClm']
    Bs67ANN = f1['Bs67pANN']
    Bs85Clm = f1['Bs85pClm']
    Bs85ANN = f1['Bs85pANN']
    f1.close()
    f2 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-rv3_week"+str(ilead+2)+".npz")
    Bs33ANNrv = f2['Bs33pANN']
    Bs67ANNrv = f2['Bs67pANN']
    Bs85ANNrv = f2['Bs85pANN']
    f2.close()
    rpsClm = Bs33Clm + Bs67Clm + Bs85Clm       # calculate ranked probability score
    rpsANN = Bs33ANN + Bs67ANN + Bs85ANN
    rpsANNrv = Bs33ANNrv + Bs67ANNrv + Bs85ANNrv
    rpssMapANN[ilead,:] = 1.-np.sum(rpsANN,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssMapANNrv[ilead,:] = 1.-np.sum(rpsANNrv,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    rpssAvgANN[ilead] = 1.-np.sum(rpsANN)/np.sum(rpsClm)
    rpssAvgANNrv[ilead] = 1.-np.sum(rpsANNrv)/np.sum(rpsClm)
    rpsDiffANNrv = rpsANN-rpsANNrv
    rpsDiffStdzANNrv = (rpsDiffANNrv-np.mean(rpsDiffANNrv,axis=(0,1))[None,None,:])/np.std(rpsDiffANNrv,axis=(0,1))[None,None,:]
    for lg in range(15):
        acfRv[ilead,lg] = np.mean(rpsDiffStdzANNrv[lg:,:,:]*rpsDiffStdzANNrv[:(ndts-lg),:,:])         # Estimate temporal autocorrelation
    rhoANNrv = acfRv[ilead,1]/acfRv[ilead,0]
    print(rhoANNrv)
    nANNrv = round(ndts*nyrs*(1-rhoANNrv)/(1+rhoANNrv))
    for ixy in range(nxy):
        smplANNrv = rpsANN[:,:,ixy].flatten()-rpsANNrv[:,:,ixy].flatten()
        tstatANNrv = np.mean(smplANNrv)/np.sqrt(np.var(smplANNrv)/nANNrv)        # test statistic for paired t-test
        pvalRv[ilead,ixy] = sp.stats.t.cdf(tstatANNrv,df=nANNrv-1)            # p-value for one-sided test
        #pval[ilead,ixy] = 2*min(1.-sp.stats.t.cdf(tstat,df=n-1),sp.stats.t.cdf(tstat,df=n-1))
    pvalRv_srt = np.sort(pvalRv[ilead,:])
    iANNrv = np.where(pvalRv_srt<=0.1*np.arange(1,nxy+1)/nxy)[0]
    if len(iANNrv)>0:
        alphaFDRrv[ilead] = pvalRv_srt[iANNrv[-1]]
    plt.figure(); plt.scatter(np.arange(663),0.1*np.arange(1,664)/663); plt.scatter(np.arange(663),pvalRv_srt)




fig = plt.figure(figsize=(11.8,6.))

for ilead in range(3):
    ylim = np.array([0.26,0.052,0.026])[ilead]
    #ylim = np.amax(abs(rpssMapCSGD[ilead,:]))
    indSgnfRv = (pvalRv[ilead,:]<alphaFDRrv[ilead])
    ax1 = fig.add_subplot(2,3,ilead+1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapANN[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgANN[ilead].round(3),fontsize=12)
    plt.title("RPSS - ANN (week "+str(ilead+2)+")",fontsize=14)
    ax2 = fig.add_subplot(2,3,ilead+4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapANNrv[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.scatter(obs_lon[indSgnfRv],obs_lat[indSgnfRv],c=rpssMapANNrv[ilead,indSgnfRv],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=19.75,lw=0.8,edgecolors=[.2,.2,.2])
    #plt.text(-118.5,40.4,'Avg. skill:',fontsize=12)
    #plt.text(-117.5,39.6,rpssAvgANNrv[ilead].round(3),fontsize=12)
    plt.title("RPSS - ANN w/o "+r'$p_{cl}$'+" (week "+str(ilead+2)+")",fontsize=14)

plt.tight_layout()




###################################################################################################
#                                                                                                 #
#  Figure S3:  Maps of RPSS of the alternative CNN implementation                                 #
#                                                                                                 #
###################################################################################################


acfCNN = np.zeros((3,15),dtype=np.float32)
pvalCNN = np.zeros((3,nxy),dtype=np.float32)
alphaFDR_CNN = np.zeros(3,dtype=np.float32)

rpssMapCNN = ma.array(np.zeros((3,nxy),dtype=np.float32),mask=True)
rpssAvgCNN = ma.array(np.zeros(3,dtype=np.float32),mask=True)

for ilead in range(3):
    f1 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-ann_week"+str(ilead+2)+".npz")
    Bs33Clm = f1['Bs33pClm']
    Bs33CSGD = f1['Bs33pCSGD']
    Bs67Clm = f1['Bs67pClm']
    Bs67CSGD = f1['Bs67pCSGD']
    Bs85Clm = f1['Bs85pClm']
    Bs85CSGD = f1['Bs85pCSGD']
    f1.close()
    f2 = np.load("/home/michael/Desktop/CalifAPCP/results/scores-rv5_week"+str(ilead+2)+".npz")
    Bs33CNN = f2['Bs33pCNN']
    Bs67CNN = f2['Bs67pCNN']
    Bs85CNN = f2['Bs85pCNN']
    f2.close()
    rpsClm = Bs33Clm + Bs67Clm + Bs85Clm       # calculate ranked probability score
    rpsCSGD = Bs33CSGD + Bs67CSGD + Bs85CSGD
    rpsCNN = Bs33CNN + Bs67CNN + Bs85CNN
    rpssMapCNN[ilead,:] = 1.-np.sum(rpsCNN,axis=(0,1))/np.sum(rpsClm,axis=(0,1))
    #rpssAvgCSGD[ilead] = 1.-np.sum(rpsCSGD)/np.sum(rpsClm)
    rpssAvgCNN[ilead] = 1.-np.sum(rpsCNN)/np.sum(rpsClm)
    rpsDiffCNN = rpsCSGD-rpsCNN
    rpsDiffStdzCNN = (rpsDiffCNN-np.mean(rpsDiffCNN,axis=(0,1))[None,None,:])/np.std(rpsDiffCNN,axis=(0,1))[None,None,:]
    for lg in range(15):
        acfCNN[ilead,lg] = np.mean(rpsDiffStdzCNN[lg:,:,:]*rpsDiffStdzCNN[:(ndts-lg),:,:])
    rhoCNN = acfCNN[ilead,1]/acfCNN[ilead,0]
    print(rhoCNN)
    nCNN = round(ndts*nyrs*(1-rhoCNN)/(1+rhoCNN))
    for ixy in range(nxy):
        smplCNN = rpsCSGD[:,:,ixy].flatten()-rpsCNN[:,:,ixy].flatten()
        tstatCNN = np.mean(smplCNN)/np.sqrt(np.var(smplCNN)/nCNN)
        pvalCNN[ilead,ixy] = 1.-sp.stats.t.cdf(tstatCNN,df=nCNN-1)
    pvalCNN_srt = np.sort(pvalCNN[ilead,:])
    iCNN = np.where(pvalCNN_srt<=0.1*np.arange(1,nxy+1)/nxy)[0]
    if len(iCNN)>0:
        alphaFDR_CNN[ilead] = pvalCNN_srt[iCNN[-1]]
    plt.figure(); plt.scatter(np.arange(663),0.1*np.arange(1,664)/663); plt.scatter(np.arange(663),pvalCNN_srt)



fig = plt.figure(figsize=(11.3,3.1))

for ilead in range(3):
    ylim = np.array([0.26,0.052,0.026])[ilead]
    indSgnfCNN = (pvalCNN[ilead,:]<alphaFDR_CNN[ilead])
    ax1 = fig.add_subplot(1,3,ilead+1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.scatter(obs_lon,obs_lat,c=rpssMapCNN[ilead,:],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=20,lw=0.3,edgecolors=[.2,.2,.2]); plt.colorbar()
    plt.scatter(obs_lon[indSgnfCNN],obs_lat[indSgnfCNN],c=rpssMapCNN[ilead,indSgnfCNN],marker='s',cmap=divcmp,vmin=-ylim,vmax=ylim,s=19.75,lw=0.8,edgecolors=[.2,.2,.2])
    plt.title("RPSS - CNN (week "+str(ilead+2)+")",fontsize=14)

plt.tight_layout()


