#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:11:26 2021

@author: rajat
"""

import glob, os
import numpy as np
import pandas as pd
import scipy.io as spio
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
plt.style.use('tableau-colorblind10')
mpl.rcParams['axes.linewidth'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

def getRatemap(dirname, hnum):
    rmaps = []
    files = natsorted(glob.glob(os.path.join(dirname,'ClustId*_hall'+str(hnum)+'_processed.mat')))
    for j, filename in enumerate(files):
        data = spio.loadmat(filename)
        rmap = data['smoothratemaptrial']
        rmaps.append(rmap)
    rmaps = np.array(rmaps)
    rmaps = np.moveaxis(rmaps, [0,1,2], [-2,-3,-1])
    return rmaps
    
pal = ['orange', 'indianred', 'c', 'royalblue']
region = 'V2'
# cluster id
dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Justin_Data/VR19'

cluidHC = np.load(os.path.join(dirname,'HCspikesorted/spikeClusterID.npy'), allow_pickle=True)
cluidV1 = np.load(os.path.join(dirname,'AL2spikesorted/spikeClusterID.npy'), allow_pickle=True)
cluidV2 = np.load(os.path.join(dirname,'AL1spikesorted/spikeClusterID.npy'), allow_pickle=True)

# load processed metrics info
dfHC = pd.read_csv(os.path.join(dirname,'HCspikesorted/cluster_info.tsv'), delimiter='\t')
dfHC = dfHC[dfHC['cluster_id'].isin(cluidHC)]
dfHC['region'] = ['HC']*len(dfHC)
dfV1 = pd.read_csv(os.path.join(dirname,'AL2spikesorted/cluster_info.tsv'), delimiter='\t')
dfV1 = dfV1[dfV1['cluster_id'].isin(cluidV1)]
dfV1['region'] = ['V1']*len(dfV1)
dfV2 = pd.read_csv(os.path.join(dirname,'AL1spikesorted/cluster_info.tsv'), delimiter='\t')
dfV2 = dfV2[dfV2['cluster_id'].isin(cluidV2)]
dfV2['region'] = ['V2']*len(dfV2)
# concatenate for entire recording
dfV2['region'] = np.where((dfV2.region=='V2') & (dfV2.depth<=350),'iHC',dfV2.region)
dfV1['region'] = np.where((dfV1.region=='V1') & (dfV1.depth<=750),'iHC',dfV1.region)


# load rate maps
if region=='iHC':
    clufname = os.path.join(dirname, 'analyzed/RatemapsAL1')
    rmaps_h1 = getRatemap(clufname, 1)
    rmaps_h2 = getRatemap(clufname, 2)
    rmaps_h3 = getRatemap(clufname, 3)
    rmaps_h4 = getRatemap(clufname, 4)
    
    # select cells that are only V2
    idx = np.where(dfV2['region']=='iHC')[0]
    rmaps_h1 = rmaps_h1[:,idx,:]
    rmaps_h2 = rmaps_h2[:,idx,:]
    rmaps_h3 = rmaps_h3[:,idx,:]
    rmaps_h4 = rmaps_h4[:,idx,:]
    
    clufname = os.path.join(dirname, 'analyzed/RatemapsAL2')
    rmaps2_h1 = getRatemap(clufname, 1)
    rmaps2_h2 = getRatemap(clufname, 2)
    rmaps2_h3 = getRatemap(clufname, 3)
    rmaps2_h4 = getRatemap(clufname, 4)
    
    # select cells that are only V2
    idx = np.where(dfV1['region']=='iHC')[0]
    rmaps2_h1 = rmaps2_h1[:,idx,:]
    rmaps2_h2 = rmaps2_h2[:,idx,:]
    rmaps2_h3 = rmaps2_h3[:,idx,:]
    rmaps2_h4 = rmaps2_h4[:,idx,:]

    rmaps_h1 = np.concatenate((rmaps_h1, rmaps2_h1),1)
    rmaps_h2 = np.concatenate((rmaps_h2, rmaps2_h2),1)
    rmaps_h3 = np.concatenate((rmaps_h3, rmaps2_h3),1)
    rmaps_h4 = np.concatenate((rmaps_h4, rmaps2_h4),1)
    
    del rmaps2_h1, rmaps2_h2, rmaps2_h3, rmaps2_h4
elif region=='V1':
    clufname = os.path.join(dirname, 'analyzed/RatemapsAL2')
    rmaps_h1 = getRatemap(clufname, 1)
    rmaps_h2 = getRatemap(clufname, 2)
    rmaps_h3 = getRatemap(clufname, 3)
    rmaps_h4 = getRatemap(clufname, 4)
    
    # select cells that are only V2
    idx = np.where(dfV1['region']=='V1')[0]
    rmaps_h1 = rmaps_h1[:,idx,:]
    rmaps_h2 = rmaps_h2[:,idx,:]
    rmaps_h3 = rmaps_h3[:,idx,:]
    rmaps_h4 = rmaps_h4[:,idx,:]
elif region=='V2':
    clufname = os.path.join(dirname, 'analyzed/RatemapsAL1')
    rmaps_h1 = getRatemap(clufname, 1)
    rmaps_h2 = getRatemap(clufname, 2)
    rmaps_h3 = getRatemap(clufname, 3)
    rmaps_h4 = getRatemap(clufname, 4)
    
    # select cells that are only V2
    idx = np.where(dfV2['region']=='V2')[0]
    rmaps_h1 = rmaps_h1[:,idx,:]
    rmaps_h2 = rmaps_h2[:,idx,:]
    rmaps_h3 = rmaps_h3[:,idx,:]
    rmaps_h4 = rmaps_h4[:,idx,:]
else:
    clufname = os.path.join(dirname, 'analyzed/RatemapsHC')
    rmaps_h1 = getRatemap(clufname, 1)
    rmaps_h2 = getRatemap(clufname, 2)
    rmaps_h3 = getRatemap(clufname, 3)
    rmaps_h4 = getRatemap(clufname, 4)

# get population vector
PV_h1 = np.nanmean(rmaps_h1,0)
PV_h2 = np.nanmean(rmaps_h2,0)
PV_h3 = np.nanmean(rmaps_h3,0)
PV_h4 = np.nanmean(rmaps_h4,0)

# get Population vector correlation across laps
corr_x1x2 = []
corr_y1y2 = []
corr_x3x4 = []
corr_z3z4 = []
corr_x1x3 = []
corr_x2x4 = []
for pv1,pv2,pv3,pv4 in zip(PV_h1, PV_h2, PV_h3, PV_h4):
    corr_y1y2.append(np.corrcoef(pv1[41:60], pv2[80:100])[0,1]) #chair in familiar context
    corr_x1x2.append(np.corrcoef(pv1[80:100], pv2[41:60])[0,1]) #star in fam context
    corr_z3z4.append(np.corrcoef(pv3[41:60], pv4[80:100])[0,1]) #drum in novel context
    corr_x3x4.append(np.corrcoef(pv3[80:100], pv4[41:60])[0,1]) #star in novel
    corr_x1x3.append(np.corrcoef(pv1[80:100], pv3[80:100])[0,1])
    corr_x2x4.append(np.corrcoef(pv1[41:60], pv4[41:60])[0,1])
corr_x1x2 = np.array(corr_x1x2)
corr_y1y2 = np.array(corr_y1y2)
corr_x3x4 = np.array(corr_x3x4)
corr_z3z4 = np.array(corr_z3z4)
corr_x1x3 = np.array(corr_x1x3)
corr_x2x4 = np.array(corr_x2x4)
