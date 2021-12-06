#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:06:15 2019

@author: rajat
"""
import os, mea
import numpy as np
import pandas as pd
from utilsEphys import *
import multiprocessing as mp
import matplotlib.pyplot as plt


# ******************* load analysis params file ***********************************
dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Justin_Data/VR19'

# load cluster id
cluidHC = np.load(os.path.join(dirname,'HCspikesorted/spikeClusterID.npy'), allow_pickle=True)
cluidV1 = np.load(os.path.join(dirname,'AL2spikesorted/spikeClusterID.npy'), allow_pickle=True)
cluidV2 = np.load(os.path.join(dirname,'AL1spikesorted/spikeClusterID.npy'), allow_pickle=True)

# load spike data from both the probes
spktimesHC = np.load(os.path.join(dirname,'HCspikesorted/spiketimes.npy'), allow_pickle=True)
spktimesV1 = np.load(os.path.join(dirname,'AL2spikesorted/spiketimes.npy'), allow_pickle=True)
spktimesV2 = np.load(os.path.join(dirname,'AL1spikesorted/spiketimes.npy'), allow_pickle=True)

# load cell metrics and cluster information
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
df = pd.concat((dfHC, dfV1, dfV2))
df = df.drop(columns=['Amplitude', 'amp', 'ContamPct', 'KSLabel', 'fr', 'n_spikes', 'sh'])
df.reset_index(drop=True, inplace=True)
df['region'] = np.where((df.region=='AL') & (df.depth<=350),'iHC',df.region)
df['region'] = np.where((df.region=='V1') & (df.depth<=750),'iHC',df.region)
dfWF = pd.read_csv('./opWaveform/waveform_metrics.csv')
dfWF.drop(columns=dfWF.columns[[0]], axis=1, inplace=True)
dfWF.reset_index(drop=True, inplace=True)
df = pd.concat((df,dfWF),axis=1)
df.drop(columns=df.columns[[1,3,5,8,10,11,12,]], axis=1, inplace=True)
del dfHC, dfV1, dfV2, cluidHC, cluidV1, cluidV2, dfWF
print("Finished loading spike data")


# load behavior data with timestamps aligned 
fname = ['./dataBehav/hall1_df.csv', './dataBehav/hall4_df.csv', './dataBehav/hall3_df.csv', './dataBehav/hall4_df.csv']
dfBehav = loadBehavData(fname, combined=True)
speedtime = np.array(dfBehav['intantime'])
speed = np.array(dfBehav['speed'])
speed[np.isnan(speed)]=0
print("Finished loading behavior data")


# **************************** LFP analysis ************************************************
# load LFP channels by shank
Fs = 1500.0
channelsPerShank = np.array(getChannelPerShank(os.path.join(os.getcwd(), 'dataHC')))
channelsPerShank = np.reshape(channelsPerShank,-1)
lfp = np.load('VR19_HC_lfp.npy', mmap_mode='r')
lfpHC = lfp[channelsPerShank, :]
# load primary visual cortex LFP
lfp = np.load('VR19_V1_lfp.npy', mmap_mode='r')
lfpiHC1 = lfp[[32,33,96,98,100],:]
# load secondary visual cortex LFP
lfp = np.load('VR19_V2_lfp.npy', mmap_mode='r')
lfpiHC2 = lfp[[17,19,48,80,81],:]
lfp = np.concatenate((lfpHC, lfpiHC1, lfpiHC2),0)
del lfpHC, lfpiHC1, lfpiHC2
lfpts = np.linspace(0, lfp.shape[1]/Fs, lfp.shape[1])
print("Finished loading LFP")

# ************************* Binned spike sum *************************************************************
total_time = lfpts[-1]
sumbinwidth = 0.025
time_bin = np.arange(0, total_time, sumbinwidth)
spkcountsumHC = getSpikeSum(spktimesHC, time_bin)
spkcountsumV1 = getSpikeSum(spktimesV1, time_bin)
spkcountsumV2 = getSpikeSum(spktimesV2, time_bin)

print("Finished calculating binned spike count")

if os.path.exists('./opRipples/ripple_power.npy'):
    ripple_power = np.load('./opRipples/ripple_power.npy', allow_pickle=True)
else:
    pool = mp.Pool(16)
    # ************************ Ripple analysis ********************************************
    lfp_ripple = pool.map(filterdata, [row for row in lfp])
    lfp_ripple = np.array(lfp_ripple)
    
    print("Finished loading ripple filtered signal across 4 shanks")
    
    # squared ampltiude and summed across channel from each shank
    medlfp_ripple = []
    for i in range(0,lfp_ripple.shape[0],5):
        medlfp_ripple.append(np.nansum(lfp_ripple[i:i+5,:]**2,0))
    medlfp_ripple = np.array(medlfp_ripple)
    
    print("Finished loading ripple filtered signal summed")
    
    # smooth with a gaussian filter and take square root
    ripple_power = pool.map(gaussianSmoothing, [row for row in medlfp_ripple])
    ripple_power = np.array(ripple_power)
    ripple_power = ripple_power**(0.5)
    np.save('./opRipples/ripple_power.npy', ripple_power)
    pool.close()

print("Finished calculating ripple signal for each shank")

# find ripple for septal -> intermediate pole
# add speed threshold, theta power, and spike sum
rippleSeptalDf1 = findRipple(lfp[0], lfpts, Fs, ripple_power[0], time_bin, 
                             spkcountsumHC, speedtime, speed)
rippleSeptalDf2 = findRipple(lfp[5], lfpts, Fs, ripple_power[1], time_bin, 
                             spkcountsumHC, speedtime, speed)
# rippleSeptalDf3 = findRipple(lfp[10], lfpts, Fs, ripple_power[2], time_bin, 
#                              spkcountsumHC, speedtime, speed)
# rippleSeptalDf4 = findRipple(lfp[15], lfpts, Fs, ripple_power[3], time_bin, 
#                              spkcountsumHC, speedtime, speed)
rippleIntDf1 = findRipple(lfp[20], lfpts, Fs, ripple_power[4], time_bin,
                          spkcountsumHC, speedtime, speed, activityRatio=0)
rippleIntDf2 = findRipple(lfp[25], lfpts, Fs, ripple_power[5], time_bin, 
                          spkcountsumHC, speedtime, speed, activityRatio=0)


# plotting data
colors2 = {'HC':'mediumorchid', 'V2':'lime', 'V1':'gold', 'iHC':'dodgerblue'}
colors3 = ['b', 'c', 'lime', 'orange', 'm', 'r']
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
for i, spkt in enumerate(np.concatenate([spktimesHC, spktimesV1, spktimesV2])):
    ax[0].scatter(spkt, i * np.ones_like(spkt), c=colors2[df['region'][i]], marker='|', rasterized=True)
for r, rp in enumerate(ripple_power[:4]):
    ax[1].plot(lfpts, rp, c=colors3[r])
# for st,et in zip(rippleIntDf1['start_time'], rippleIntDf1['end_time']):
#     ax[1].axvspan(st, et, alpha=0.3, color='y')
for st,et in zip(rippleSeptalDf2['start_time'], rippleSeptalDf2['end_time']):
    ax[1].axvspan(st, et, alpha=0.3, color='gray')
ax[2].plot(dfBehav['intantime'], dfBehav['speed'])
plt.tight_layout()
plt.show()

rippleSeptalDf1.to_csv('./opRipples/ripplesShank1.csv')
rippleSeptalDf2.to_csv('./opRipples/ripplesShank2.csv')