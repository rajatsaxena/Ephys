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
import matplotlib.pyplot as plt

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
del dfRipple


# find modulation of each cell with respect to each SWR event
# along with shuffling
pethbins=np.arange(-1,1.02,0.02)
swrpeth_cells = []
swrpethshuf_cells = []
swrpethpval_cells = []
swrmoddir_cells = []
for s,spktime in enumerate(spiketimes):
    swrmodop = calcSWRmodulation(spktime, peak_time)
    swrpeth_cells.append(swrmodop[0])
    swrpethshuf_cells.append(swrmodop[1])
    swrpethpval_cells.append(swrmodop[2])
    swrmoddir_cells.append(swrmodop[3])
swrpeth_cells = np.array(swrpeth_cells)
swrpethshuf_cells = np.array(swrpethshuf_cells)
swrpethpval_cells = np.array(swrpethpval_cells)
swrmoddir_cells = np.array(swrmoddir_cells)


# for i,c in enumerate(cellswrbinnedpeth[200:]):
#     plt.subplot(15,15,i+1)
#     plt.plot(bins[:-1], cellavg[i]) #, aspect='auto')
# plt.show()
    

# find % SWR in familiar, novel, and rest period
