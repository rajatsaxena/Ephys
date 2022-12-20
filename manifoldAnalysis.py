# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:16:26 2022

@author: Rajat
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D


# preprocessing data
dt = 0.1 # 100ms
spiketimes = np.load('CT2-spiketimes.npy', allow_pickle=True)
awkbins = np.arange(0,9000,dt)
slpbins = np.arange(9000,54000,dt)
Qawk = []
Qslp = []
for spk in spiketimes:
    c, _ = np.histogram(spk, np.arange(0,9000,dt))
    c1, _ = np.histogram(spk, np.arange(9000,54000,dt))
    Qawk.append(c)
    Qslp.append(c1)
Qawk = np.array(Qawk, dtype='float')
Qslp = np.array(Qslp, dtype='float')
Qawk = Qawk[:,np.nansum(Qawk,0)!=0]
Qslp = Qslp[:,np.nansum(Qslp,0)!=0]
del spiketimes
Qawksum = np.nansum(Qawk,0)
Qslpsum = np.nansum(Qslp,0)

# population coupling strength
corrawk = []
corrslp = []
for i in range(Qawk.shape[0]):
    c = np.corrcoef(Qawk[i,:], Qawksum)
    c = c[0,1]
    c1 = np.corrcoef(Qslp[i,:], Qslpsum)
    c1 = c1[0,1]
    corrawk.append(c)
    corrslp.append(c1)
corrawk = np.array(corrawk)
corrslp = np.array(corrslp)
plt.hist([corrawk, corrslp])
plt.show()

#plt.subplot(121)
#corr1 = np.corrcoef(Qawk)
#np.fill_diagonal(corr1,None)
#plt.imshow(corr1, aspect='auto', vmin=-0.2, vmax=0.4)
#plt.colorbar()
#plt.subplot(122)
#corr2 = np.corrcoef(Qslp)
#np.fill_diagonal(corr2,None)
#plt.imshow(corr2, aspect='auto', vmin=-0.2, vmax=0.4)
#plt.colorbar()
#plt.show()

# apply PCA on sleep and awake dataset
pca_awk = decomposition.PCA()
X_awk = pca_awk.fit_transform(Qawk.T)
X_awk_loadings = pca_awk.components_.T # each column is a PC
pca_slp = decomposition.PCA()
X_slp = pca_slp.fit_transform(Qslp.T)
X_slp_loadings = pca_slp.components_.T

# select half random indices from sleep 
# and awake and then run PCA
awk_idx = np.arange(Qawk.shape[-1]-1)
random.shuffle(awk_idx)
awk_idx_h1 = np.sort(awk_idx[:len(awk_idx)//2])
awk_idx_h2 = np.sort(awk_idx[len(awk_idx)//2:])
slp_idx = np.arange(Qslp.shape[-1]-1)
random.shuffle(slp_idx)
slp_idx_h1 = np.sort(slp_idx[:len(slp_idx)//2])
slp_idx_h2 = np.sort(slp_idx[len(slp_idx)//2:])

# apply pca on both halves
pca_awk = decomposition.PCA(n_components=3)
pca_awk.fit(Qawk[:,awk_idx_h1].T)
pca_awk_t = pca_awk.transform(Qawk[:,awk_idx_h2].T)
pca_slp = decomposition.PCA(n_components=3)
pca_slp.fit(Qslp[:,slp_idx_h1].T)
pca_slp_t = pca_slp.transform(Qslp[:,slp_idx_h2].T)

## plot in 3D
#fig = plt.figure(figsize = (10, 7))
#ax = fig.gca(projection ="3d")
#ax.scatter3D(pca_awk_t[:,1], pca_awk_t[:,0], 
#             pca_awk_t[:,2], s=1, rasterized=True, 
#             c=np.arange(pca_awk_t.shape[0]))
#plt.show()
#
#
#fig = plt.figure(figsize = (10, 7))
#ax = fig.gca(projection ="3d")
#ax.scatter3D(pca_slp_t[:,0], pca_slp_t[:,1], 
#             pca_slp_t[:,2], s=1, rasterized=True, c=np.arange(pca_slp_t.shape[0]))
#plt.show()