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

# function to get reactivation strength
def ReactStrength(Qref, Qtar, method='pca'):
    # Qref = nCells x nTimes
    nCells = Qref.shape[0]
    if nCells != Qref.shape[0]:
        sys.error(2)
        
    # calculate correlation matrix
    Cref = np.corrcoef(Qref)
    
    # run pca 
    pca = PCA()
    pca = pca.fit(Cref)
    PCs = pca.components_ # used for projection matrix
    lambdas = pca.explained_variance_ratio_*nCells
    
    # Marcenko-Pastur threshold
    lMax = (1 + np.sqrt(nCells/Qref.shape[1]))**2
    nPCs = np.sum(lambdas>lMax)
    phi = lambdas[:nPCs]/lMax
    
    if method=='ica':
        _, PCs, _ = fastica.fastica(Qref, nCells, algorithm='parallel')
        PCs = PCs/np.array([np.sqrt(np.sum(PCs**2,1))]*nCells)

    # get reactivation strength for each template across time bins
    QtarZ = scst.zscore(Qtar,1).T
    scoreTar = QtarZ.dot(PCs[:,:nPCs])
    
    # this trick is used to get rid of the diagonal term in react. strength
    tmp = (QtarZ**2).dot(PCs[:,:nPCs]**2)
    R = scoreTar**2 - tmp 
    
    return R, phi


Qref = np.load('Qrun.npy', allow_pickle=True)
Qtar = np.load('Qsleep.npy', allow_pickle=True)
Qref = np.delete(Qref, (80,83), axis=0)
Qtar = np.delete(Qtar, (80,83), axis=0)


R1, _ = ReactStrength(Qref, Qtar, method='pca')
R2, _ = ReactStrength(Qref, Qtar, method='ica')

plt.plot(R1[:,0])
plt.plot(R2[:,0])