#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:21:00 2020

@author: rajat

Credit goes to David Tingley and Adrien Peyrache: 
    https://github.com/DavidTingley/RnR_methods
"""
import sys
import fastica
import numpy as np
import scipy.stats as scst
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# function to explain explained variance
def calcEV(Qpre, Qtask, Qpost):
    Rrest1 = np.corrcoef(Qpre)
    Rrun = np.corrcoef(Qtask)
    Rrest2 = np.corrcoef(Qpost)

    Rrest1 = Rrest1[np.triu_indices(Rrest1.shape[0], k=1)]
    Rrun = Rrun[np.triu_indices(Rrun.shape[0], k=1)]
    Rrest2 = Rrest2[np.triu_indices(Rrest2.shape[0], k=1)]
    
    nanidx = (np.isnan(Rrest1) | np.isnan(Rrun)) | np.isnan(Rrest2)
    Rrest1 = np.delete(Rrest1, nanidx)
    Rrun = np.delete(Rrun, nanidx)
    Rrest2 = np.delete(Rrest2, nanidx)
    
    PrePost = np.corrcoef(Rrest1, Rrest2)[0,1]
    ExpPre = np.corrcoef(Rrun, Rrest1)[0,1]
    ExpPost = np.corrcoef(Rrun, Rrest2)[0,1]
    
    EV = ( (ExpPost - ExpPre*PrePost) / (np.sqrt((1 - ExpPre**2) * (1 - PrePost**2))) )**2
    REV = ( (ExpPre - ExpPost*PrePost) / (np.sqrt((1 - ExpPost**2) * (1 - PrePost**2))) )**2
    
    return EV, REV

# function to get reactivation strength
def calcReactStrength(Qref, Qtar, method='pca'):
    # Qref = nCells x nTimes
    nCells = Qref.shape[0]
    if nCells != Qref.shape[0]:
        sys.error(2)
        
    # calculate correlation matrix
    Cref = np.corrcoef(Qref)
    
    # run pca 
    pca = PCA()
    pca = pca.fit(Cref)
    PCs = pca.components_.T # used for projection matrix
    lambdas = pca.explained_variance_ratio_*nCells
    
    # Marcenko-Pastur threshold
    lMax = (1 + np.sqrt(nCells/Qref.shape[1]))**2
    nPCs = np.sum(lambdas>lMax)
    phi = lambdas[:nPCs]/lMax
    
    if method=='ica':
        PCs, _, _ = fastica.fastica(Qref, nPCs, algorithm='parallel')
        PCs = PCs/np.tile(np.sqrt(np.nansum(PCs**2,0)),(PCs.shape[0],1)).T

    # get reactivation strength for each template across time bins
    QtarZ = scst.zscore(Qtar,1).T
    scoreTar = QtarZ.dot(PCs[:,:nPCs])
    
    print(QtarZ.shape, PCs.shape, scoreTar.shape)
    
    # this trick is used to get rid of the diagonal term in react. strength
    tmp = (QtarZ**2).dot(PCs[:,:nPCs]**2)
    R = scoreTar**2 - tmp 
    
    return R, phi