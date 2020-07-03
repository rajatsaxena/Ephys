#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:20:57 2019

@author: rajat
"""
import sys
sys.setrecursionlimit(100000000)
from scipy.cluster.hierarchy import dendrogram, linkage
from EnsemblePursuit.EnsemblePursuit import EnsemblePursuit
from sklearn.decomposition import PCA
from rastermap import Rastermap
from scipy.stats import zscore
import matplotlib.pyplot as plt
import scipy.io as scsio
import fastcluster
import numpy as np

def corr_matrix(x,y):
    """
    calculate correlation matrix
    """
    x = x- np.mean(x,axis=0)
    y = y- np.mean(y,axis=0)
    x /= np.std(x,axis=0) + 1e-10
    y /= np.std(y,axis=0) + 1e-10
    c = x.T @ y
    return c

def corr_map(x, y):
    """Correlate each n with each m.
    x : Shape N X T.
    y : Shape M X T.

    Returns
    -------
      N X M array in which each element is a correlation coefficient.
    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    # return correlation matrix
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def performPCA(PV, zscale=True):
    """
    Parameters
    ----------
    PV : float32/float64
        population vector: Time x Number of cells matrix
    zscale : boolean
        perform zscore operation

    Returns
    -------
    variance: amount of variance explained by each PC
    cumulative variance: cumulative variance explained by each PC
    """
    if zscale:
        PV = zscore(PV)
    # run PCA
    pca = PCA()
    pca.fit(PV)
    # transformed PCA
    X_pca = pca.transform(PV)
    #The amount of variance that each PC explains
    variance_PV = pca.explained_variance_ratio_*PV.shape[1]
    #Cumulative Variance explains
    cdfvar_PV = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    
    # Marcenko-Pastur threshold
    nCells = PV.shape[1]
    lMax = (1 + np.sqrt(nCells/PV.shape[0]))**2
    nPCs = np.sum(variance_PV>lMax)
    phi = variance_PV[:nPCs]/lMax
    # return the data
    return variance_PV, cdfvar_PV, X_pca, phi

def getEnsemble(PV, Ncomponents=None, plot=False):
    """
    Parameters
    ----------
    PV : float32/float64
        population vector: Time x Number of cells matrix

    Returns
    -------
    U : ncells x ncomponents
        each column of U represents which neurons belong to the ensemble and 
        with what strength they contribute to the time evolution of an ensemble
        each row demarcates which ensembles a neuron belongs to and its intensity.
    V : timepoints x ncomponents
        V stores the time evolution of an ensemble. It's an average time course 
        of the neurons that belong to that ensemble.
    """
    # zscore the PV
    sp  = zscore(PV)
    if Ncomponents is None:
        Ncomponents=sp.shape[1]
    # run ensemble pursuit
    ep=EnsemblePursuit(n_components=Ncomponents,lam=0.01)
    model=ep.fit(sp)
    V=model.components_ # timepoints x ncomponents
    U=model.weights # ncells x ncomponents
    
    # plot the model weights
    if plot:
        plt.figure()
        plt.hist(np.sum(U>0, axis=1), bins=5)
        plt.xlabel('number of ensembles / neuron')
        plt.ylabel('neuron counts')
        plt.title('distribution of neurons in ensembles')
        plt.show()
    return U, V

def getRastermapEmbeddings(PV, nX=50, nbin=50):
    """
    Parameters
    ----------
    PV : float32/float64
        population vector: Time x Number of cells matrix
    Returns
    -------
    isort : manifold embedding
    Sfilt : sorted signals from all neurons.

    """
    # raster map to find underlying embeddings
    model = Rastermap(n_components=1, n_X=nX).fit(PV.T)
    # the manifold embedding is in model.embedding
    isort = np.argsort(model.embedding[:,0])
    # sort by embedding and smooth over neurons
    Sfilt = running_average(PV.T[isort, :], nbin)
    Sfilt = zscore(Sfilt, axis=1)
    return isort, Sfilt

def running_average(X, nbin = 100):
    """
    # this function performs a running average filter over the first dimension of X
    # (faster than doing gaussian filtering)
    Parameters
    """
    Y = np.cumsum(X, axis=0)
    Y = Y[nbin:, :] - Y[:-nbin, :]
    return Y

def sparsityPV(x):
    """
    x: T timepoints x N cells 
    """
    return np.nanmean(x,1)**2/ np.nanmean(x**2, 1)


# load the data - NREM and REM epochs
data = scsio.loadmat('clustering_data_urethane.mat')
pop_vector = data['pop_vector'].T
pop_vector_nrem = data['pop_vector_nrem'].T
pop_vector_rem = data['pop_vector_rem'].T
sup_cell_idx = data['sup_cell_idx'][0]-1
deep_cell_idx = data['deep_cell_idx'][0]-1
del data

# remove NREM epochs with 0 spikes
sum_nrem = np.nansum(pop_vector_nrem,1)
remove_idx = np.where(sum_nrem==0)[0]
pop_vector_nrem = np.delete(pop_vector_nrem, remove_idx, 0)

# change to superficial vs deep cortex
pop_vector_nrem_sup = pop_vector_nrem[:,sup_cell_idx]
pop_vector_nrem_deep = pop_vector_nrem[:,deep_cell_idx]
pop_vector_rem_sup = pop_vector_rem[:,sup_cell_idx]
pop_vector_rem_deep = pop_vector_rem[:,deep_cell_idx]
# overall superifical vs deep PV
pop_vector_sup = pop_vector[:,sup_cell_idx]
pop_vector_deep = pop_vector[:,deep_cell_idx]

# perform PCA to get explained variance
var_nrem_sup, cdfvar_nrem_sup, X_nrem_sup, phi_nrem_sup = performPCA(pop_vector_nrem_sup)
var_nrem_deep, cdfvar_nrem_deep, X_nrem_deep, phi_nrem_deep = performPCA(pop_vector_nrem_deep)
var_rem_sup, cdfvar_rem_sup, X_rem_sup, phi_rem_sup = performPCA(pop_vector_rem_sup)
var_rem_deep, cdfvar_rem_deep, X_rem_deep, phi_rem_deep = performPCA(pop_vector_rem_deep)
var_sup, cdfvar_sup, X_sup, phi_sup = performPCA(pop_vector_sup)
var_deep, cdfvar_deep, X_deep, phi_deep = performPCA(pop_vector_deep) 


# plotting the explained variance 
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(cdfvar_nrem_sup, label='NREM sup')
plt.plot(cdfvar_nrem_deep, label='NREM deep')
plt.plot(cdfvar_rem_sup, label='REM sup')
plt.plot(cdfvar_rem_deep, label='REM deep')
plt.xlabel('number of components', fontsize=18)
plt.ylabel('cumulative explained variance', fontsize=18)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.subplot(122)
plt.plot(cdfvar_sup, label='sup')
plt.plot(cdfvar_deep, label='deep')
plt.xlabel('number of components', fontsize=18)
plt.ylabel('cumulative explained variance', fontsize=18)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.show()


plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(var_nrem_sup, label='NREM sup')
plt.plot(var_nrem_deep, label='NREM deep')
plt.plot(var_rem_sup, label='REM sup')
plt.plot(var_rem_deep, label='REM deep')
plt.xlabel('Principal components', fontsize=18)
plt.ylabel('% of Explained variance', fontsize=18)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.subplot(122)
plt.plot(var_sup, label='sup')
plt.plot(var_deep, label='deep')
plt.xlabel('Principal components', fontsize=18)
plt.ylabel('% of Explained variance', fontsize=18)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.show()


# ensemble pursuit
# get sparse matrix factorization to get find co-activating  cells
U_nrem_sup, V_nrem_sup = getEnsemble(pop_vector_nrem_sup, Ncomponents=40)
U_nrem_deep, V_nrem_deep = getEnsemble(pop_vector_nrem_deep, Ncomponents=40)
U_rem_sup, V_rem_sup = getEnsemble(pop_vector_rem_sup, Ncomponents=40)
U_rem_deep, V_rem_deep = getEnsemble(pop_vector_rem_deep, Ncomponents=40)


plt.figure(figsize=(12,12))
plt.subplot(221)
plt.hist(np.sum(U_nrem_sup>0, axis=1), bins = np.arange(-.5,10,1))
plt.xlabel('number of ensembles / neuron')
plt.ylabel('neuron counts')
plt.title('NREM sup: dist of neurons in ensembles')
plt.subplot(222)
plt.hist(np.sum(U_nrem_deep>0, axis=1), bins = np.arange(-.5,10,1))
plt.xlabel('number of ensembles / neuron')
plt.ylabel('neuron counts')
plt.title('NREM deep: dist of neurons in ensembles')
plt.subplot(223)
plt.hist(np.sum(U_rem_sup>0, axis=1), bins = np.arange(-.5,10,1))
plt.xlabel('number of ensembles / neuron')
plt.ylabel('neuron counts')
plt.title('REM sup: dist of neurons in ensembles')
plt.subplot(224)
plt.hist(np.sum(U_rem_deep>0, axis=1), bins = np.arange(-.5,10,1))
plt.xlabel('number of ensembles / neuron')
plt.ylabel('neuron counts')
plt.title('REM deep: dist of neurons in ensembles')
plt.show()

# rastermap embeddings
# _, sig_nrem_sup = getRastermapEmbeddings(pop_vector_nrem_sup)

# plt.figure(figsize=(16,12))
# plt.imshow(sig_nrem_sup, vmin = -0.5, vmax=3, aspect='auto', cmap='gray_r')
# plt.xlabel('time points')
# plt.ylabel('sorted neurons')
# plt.show()

# hierarchical clustering
Z = fastcluster.linkage(pop_vector_nrem_sup, 'ward')
sp_PV = sparsityPV(Z)
#plt.hist(Z[:,2], bins=100, normed=True, cumulative=True, label='CDF', alpha=0.8, color='k')


# SVD methods
u, s, vh = np.linalg.svd(pop_vector_nrem_sup)