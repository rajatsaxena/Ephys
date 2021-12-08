#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:04:55 2021

@author: rajat
"""
import os
import numpy as np
import pandas as pd 
from utilsEphys import *
import scipy.stats as spst
import scipy.ndimage as scnd
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

mpl.rcParams['axes.linewidth'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

# ******************* load analysis params file ***********************************
dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Justin_Data/VR19'

# load spike data from both the probes
spktimesHC = np.load(os.path.join(dirname,'HCspikesorted/spiketimes.npy'), allow_pickle=True)
spktimesV1 = np.load(os.path.join(dirname,'AL2spikesorted/spiketimes.npy'), allow_pickle=True)
spktimesV2 = np.load(os.path.join(dirname,'AL1spikesorted/spiketimes.npy'), allow_pickle=True)
spiketimes = np.concatenate((spktimesHC, spktimesV1, spktimesV2))
del spktimesHC, spktimesV1, spktimesV2

# load processed metrics info
dfProc = pd.read_csv('dFProcessed.csv')
dfProc.drop(columns=dfProc.columns[0], inplace=True)

# load ripple data and throw away all events within 500ms
dfRipple = pd.read_csv('./opRipples/ripplesShank2.csv', index_col=0)
et = np.array(dfRipple['end_time'][1:])
st = np.array(dfRipple['start_time'][:-1])
idx = np.where((et - st)<0.5)[0]
dfRipple.drop(idx, inplace=True)
peak_time = np.array(dfRipple['peak_time'])
del dfRipple, et, st, idx

# find modulation of each cell with respect to each SWR event
# along with shuffling
pethbins=np.arange(-1,1.01,0.01)
if not os.path.exists('./opRipples/swrpeth.npy'):
    pool = mp.Pool(16)
    swrmodresults = pool.map(calcSWRmodulation, [spk for spk in spiketimes])
    pool.close()
    swrmodresults = np.array(swrmodresults)
    swrpeth_cells = swrmodresults[:,0]
    # swrpethshuf_cells = []
    swrpethpval_cells = np.array(list(swrmodresults[:,1]))
    swrmoddir_cells =  np.array(list(swrmodresults[:,2]))
    np.save('./opRipples/swrpeth.npy', swrpeth_cells)
    # np.save('./opRipples/swrshufpeth.npy', swrpethshuf_cells)
    np.save('./opRipples/swrpvals.npy', swrpethpval_cells)
    np.save('./opRipples/swrmoddir.npy', swrmoddir_cells)
else:
    swrpeth_cells = np.load('./opRipples/swrpeth.npy', allow_pickle=True)
    swrpethpval_cells = np.load('./opRipples/swrpvals.npy', allow_pickle=True)
    swrmoddir_cells = np.load('./opRipples/swrmoddir.npy', allow_pickle=True)

# find mean peth of all cells
swrPethMean = []
for peth in swrpeth_cells:
    swrPethMean.append(np.nanmean(peth,0))
swrPethMean = np.array(swrPethMean)

# find cells that are modulated significantly
modidx = np.where(swrpethpval_cells<0.01)[0]
nonmodidx = np.where(swrpethpval_cells>0.2)[0]
dfMod = dfProc.iloc[modidx]
swrPethMeanmod = swrPethMean[modidx,:]
swrModDir = swrmoddir_cells[modidx]
swrpvals = swrpethpval_cells[modidx]
dfMod['swrmoddir'] = swrModDir
dfMod['swrpval'] = swrpvals
swrPethMeanmod = np.apply_along_axis(norm1d, axis=1, arr=swrPethMeanmod)

# negative swr modulated units
negmeanswrmodHC, negstdswrmodHC = getSWRModDat(dfMod, swrPethMeanmod, region1='HC', region2='iHC')
negmeanswrmodV1, negstdswrmodV1 = getSWRModDat(dfMod, swrPethMeanmod, region1='V1')
negmeanswrmodV2, negstdswrmodV2 = getSWRModDat(dfMod, swrPethMeanmod, region1='V2')
# positive swr modualted units
posmeanswrmodHC, posstdswrmodHC = getSWRModDat(dfMod, swrPethMeanmod, region1='HC', region2='iHC', negative=False)
posmeanswrmodV1, posstdswrmodV1 = getSWRModDat(dfMod, swrPethMeanmod, region1='V1', negative=False)
posmeanswrmodV2, posstdswrmodV2 = getSWRModDat(dfMod, swrPethMeanmod, region1='V2', negative=False)
# non modulated units
swrPethMeanNonmod = swrPethMean[nonmodidx,:]
df = dfProc.iloc[nonmodidx]
index = np.where(df['region']=='HC')[0]
nonmodunits = swrPethMeanNonmod[index,:]
meanswrnomodHC = scnd.gaussian_filter1d(np.nanmean(nonmodunits,0),2)
meanswrnomodHC = meanswrnomodHC/np.nanmax(meanswrnomodHC)
stdswrnomodHC = np.nanstd(nonmodunits,0)*0.35
index = np.where(df['region']=='V1')[0]
nonmodunits = swrPethMeanNonmod[index,:]
meanswrnomodV1 = scnd.gaussian_filter1d(np.nanmean(nonmodunits,0),2)
meanswrnomodV1 = meanswrnomodV1/np.nanmax(meanswrnomodV1)
stdswrnomodV1 = np.nanstd(nonmodunits,0)*0.35
index = np.where(df['region']=='V2')[0]
nonmodunits = swrPethMeanNonmod[index,:]
meanswrnomodV2 = scnd.gaussian_filter1d(np.nanmean(nonmodunits,0),2)
meanswrnomodV2 = meanswrnomodV2/np.nanmax(meanswrnomodV2)
stdswrnomodV2 = np.nanstd(nonmodunits,0)*0.35


plt.figure(figsize=(7,4))
plt.subplot(121)
plt.fill_between(pethbins[:-1], negmeanswrmodHC-negstdswrmodHC, negmeanswrmodHC+negstdswrmodHC, color='mediumorchid', alpha=0.5)
plt.plot(pethbins[:-1], negmeanswrmodHC, color='mediumorchid', linewidth=2)
plt.fill_between(pethbins[:-1], negmeanswrmodV1-negstdswrmodV1, negmeanswrmodV1+negstdswrmodV1, color='gold', alpha=0.5)
plt.plot(pethbins[:-1], negmeanswrmodV1, color='gold', linewidth=2)
plt.fill_between(pethbins[:-1], negmeanswrmodV2-negstdswrmodV2, negmeanswrmodV2+negstdswrmodV2, color='lime', alpha=0.5)
plt.plot(pethbins[:-1], negmeanswrmodV2, color='lime', linewidth=2)
plt.plot([0,0],[0.3,1.2], color='k', linewidth=2, linestyle='--')
plt.xlim([-0.4,0.4])
plt.ylim([0.4,1.1])
plt.yticks([])
plt.subplot(122)
plt.fill_between(pethbins[:-1], posmeanswrmodHC-posstdswrmodHC, posmeanswrmodHC+posstdswrmodHC, color='mediumorchid', alpha=0.5)
plt.plot(pethbins[:-1], posmeanswrmodHC, color='mediumorchid', linewidth=2)
plt.fill_between(pethbins[:-1], posmeanswrmodV1-posstdswrmodV1, posmeanswrmodV1+posstdswrmodV1, color='gold', alpha=0.5)
plt.plot(pethbins[:-1], posmeanswrmodV1, color='gold', linewidth=2)
plt.fill_between(pethbins[:-1], posmeanswrmodV2-posstdswrmodV2, posmeanswrmodV2+posstdswrmodV2, color='lime', alpha=0.5)
plt.plot(pethbins[:-1], posmeanswrmodV2, color='lime', linewidth=2)
plt.plot([0,0],[0.3,1.2], color='k', linewidth=2, linestyle='--')
plt.xlim([-0.4,0.4])
plt.ylim([0.3,1.1])
plt.yticks([])
plt.show()