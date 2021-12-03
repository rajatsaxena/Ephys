#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 01:31:39 2021

@author: rajat
"""
import os, glob
import numpy as np
import scipy.io as spio
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('tableau-colorblind10')

mpl.rcParams['axes.linewidth'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
from MulticoreTSNE import MulticoreTSNE as TSNE


# function to find internal structure in Population of each brain region
def getPVStructure(maindir,dir1,dir2,dir3,figtitle,figfname,hallnum=1):
    r1fnames = natsorted(glob.glob(os.path.join(dirname, 'analyzed', dir1, 'ClustId*'+'hall'+str(hallnum)+'_processed.mat')))
    r2fnames = natsorted(glob.glob(os.path.join(dirname, 'analyzed', dir2, 'ClustId*'+'hall'+str(hallnum)+'_processed.mat')))
    r3fnames = natsorted(glob.glob(os.path.join(dirname, 'analyzed', dir3, 'ClustId*'+'hall'+str(hallnum)+'_processed.mat')))
    
    # create ratemaps of N neurons x P position bins X T trials
    # for each brain region
    r1ratemaps = []
    r2ratemaps = []
    r3ratemaps = []
    for h1f, h2f, h3f in zip(r1fnames, r2fnames, r3fnames):
        dat = spio.loadmat(h1f)
        r1rmap = dat['smoothratemaptrial'].T
        dat = spio.loadmat(h2f)
        r2rmap = dat['smoothratemaptrial'].T
        dat = spio.loadmat(h3f)
        r3rmap = dat['smoothratemaptrial'].T
        del dat
        r1ratemaps.append(r1rmap)
        r2ratemaps.append(r2rmap)
        r3ratemaps.append(r3rmap)
    r1ratemaps = np.array(r1ratemaps)
    r2ratemaps = np.array(r2ratemaps)
    r3ratemaps = np.array(r3ratemaps)
    
    # get correlation between PV and extract upper triangle matrix
    # store it in a single matrix for each brain region
    nposbins = r2ratemaps.shape[1]
    structure_area = []
    area_labels = []
    for a, structure in zip(range(3), [r1ratemaps, r2ratemaps, r3ratemaps]):
        for nt in range(structure.shape[-1]):
            struct_trial = structure[:,:,nt] # load each trial PV
            struct_trial = np.corrcoef(struct_trial.T) # computer correlation
            struct_trial = struct_trial[np.triu_indices(nposbins,1)]
            structure_area.append(struct_trial)
            area_labels.append(a)
    structure_area = np.array(structure_area)
    area_labels = np.array(area_labels)
    
    # initialize t-sne and apply it on the structure
    tsne = TSNE(n_components=3, perplexity=20, method='exact', metric='cosine', n_jobs=20)
    dim_reduce = tsne.fit_transform(structure_area)
    
    # plot the PV activity across trials for each brain region
    colors = ['mediumorchid', 'lime', 'orange']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for area in range(3):
        ax.scatter(-dim_reduce[area_labels==area,1],dim_reduce[area_labels==area,0],
                   dim_reduce[area_labels==area,2],s=20,c=colors[area], rasterized=True)
    plt.xlabel('Component 1', fontsize=18, labelpad=2)
    plt.ylabel('Component 2', fontsize=18, labelpad=2)
    ax.set_zlabel('Component 3', fontsize=18, labelpad=2)
    plt.title(figtitle, fontsize=23)
    plt.savefig(figfname, dpi=300)
    plt.close()
    
    return [r1ratemaps, r2ratemaps, r3ratemaps]

# load hippocampus cluster data
dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Justin_Data/VR19'

# run PV internal representation for each hallway
[hcratemaps, v2ratemaps, v1ratemaps] = getPVStructure(dirname,'RatemapsHC','RatemapsAL1','RatemapsAL2','Hallway #1','./opPV/hall1PVstruct.png',hallnum=1)
[hcratemaps, v2ratemaps, v1ratemaps] = getPVStructure(dirname,'RatemapsHC','RatemapsAL1','RatemapsAL2','Hallway #2','./opPV/hall2PVstruct.png',hallnum=2)
[hcratemaps, v2ratemaps, v1ratemaps] = getPVStructure(dirname,'RatemapsHC','RatemapsAL1','RatemapsAL2','Hallway #3','./opPV/hall3PVstruct.png',hallnum=3)
[hcratemaps, v2ratemaps, v1ratemaps] = getPVStructure(dirname,'RatemapsHC','RatemapsAL1','RatemapsAL2','Hallway #4','./opPV/hall4PVstruct.png',hallnum=4)
