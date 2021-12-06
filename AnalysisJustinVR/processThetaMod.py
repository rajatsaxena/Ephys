#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 01:16:10 2021

@author: rajat
"""
import os, mea
import numpy as np
import pandas as pd
from utilsEphys import *
import pylab as plt
import multiprocessing as mp
from detect_peaks import detect_peaks


# ******************* load analysis params file ***********************************
# load cluster id
cluidHC = np.load('./dataHC/spikeClusterID.npy', allow_pickle=True)
cluidV1 = np.load('./dataV1/spikeClusterID.npy', allow_pickle=True)
cluidV2 = np.load('./dataV2/spikeClusterID.npy', allow_pickle=True)
                   
# load spike data from both the probes
spktimesHC = np.load('./dataHC/spiketimes.npy', allow_pickle=True)
spktimesV1 = np.load('./dataV1/spiketimes.npy', allow_pickle=True)
spktimesV2 = np.load('./dataV2/spiketimes.npy', allow_pickle=True)

# load cell metrics and cluster infor
dfHC = loadCellMetrics('VR19HC.cell_metrics.cellinfo.mat', './dataHC/cluster_info.tsv', cluidHC, 'HC')
dfV1 = loadCellMetrics('VR19AL2.cell_metrics.cellinfo.mat', './dataV1/cluster_info.tsv', cluidV1, 'V1')
dfV2 = loadCellMetrics('VR19AL1.cell_metrics.cellinfo.mat', './dataV2/cluster_info.tsv', cluidV2, 'V2')
# concatenate for entire recording
df = pd.concat((dfHC, dfV1, dfV2))
df.reset_index(drop=True, inplace=True)
df = df.drop(columns=['Amplitude', 'amp', 'ContamPct', 'KSLabel', 'fr', 'n_spikes', 'sh'])
del dfHC, dfV1, dfV2, cluidHC, cluidV1, cluidV2
df['region'] = np.where((df.region=='V2') & (df.depth<=350),'iHC',df.region)
df['region'] = np.where((df.region=='V1') & (df.depth<=750),'iHC',df.region)

print("Finished loading input data")

# **************************** LFP analysis ************************************************
# load LFP channels by shank
Fs = 1500.0
# channelsPerShank = [8,9,20,21,22,41,52,53,54,55,83,84,85,86,87,105,115,116,118]
lfp = np.load('VR19_V1_lfp.npy', mmap_mode='r')

print("Finished loading LFP data")

pool = mp.Pool(16)
thetaRatio = pool.map(getBestThetaChannel, [row for row in lfp])
thetaRatio = np.array(thetaRatio)
ch = np.argmax(thetaRatio)
#dch = 51
lfptheta = lfp[ch]
dorsalHCtheta = -mea.get_bandpass_filter_signal(lfptheta, Fs, [6,10])
lfpts = np.linspace(0, lfp.shape[1]/Fs, lfp.shape[1])

print("Finished calculating best theta channel")

# ******************** spike Phase and theta modulation calculation *************************************************
ampthresh = np.nanmean(dorsalHCtheta) + 0.7*np.nanstd(dorsalHCtheta)
theta_peakindices = detect_peaks(dorsalHCtheta, mph=ampthresh, mpd=int(Fs/10.))
peakamp = dorsalHCtheta[theta_peakindices]
peaktime = lfpts[theta_peakindices]

print("Finished theta peak detection")

dat = [[spktimesHC, peaktime, dorsalHCtheta, Fs], [spktimesV1, peaktime, dorsalHCtheta, Fs], [spktimesV2, peaktime, dorsalHCtheta, Fs]]
spikePhase = pool.map(multi_run_wrapper,dat)
spikePhase =  np.concatenate(spikePhase)

print("Finished spike phase calculation")

# calculate TMI
tmiinfo = pool.map(getTMI, [unit for unit in spikePhase])
tmiinfo = np.array(tmiinfo)
df['tmi'] = tmiinfo[:,0]
df['preftmiPhase'] = tmiinfo[:,1]

print("Finished TMI calculation")

pval = pool.map(multi_run_wrapper_tmi, [[spk, tmi] for spk,tmi in zip(spikePhase,tmiinfo[:,0])])
pval = np.array(pval)
tmisig = np.copy(pval)
tmisig[pval>=0.05] = 0.0
tmisig[pval<0.05] = 1.0
df['tmipval'] = pval
df['tmisig'] = tmisig

print("Finished TMI shuffling calculation")

df.to_csv('./opTMI/VR19processedmetrics_iHC.csv')
np.save('./opTMI/VR19spikephase_iHC.npy', spikePhase)

pool.close()