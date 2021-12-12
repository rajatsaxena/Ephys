#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:19:41 2021

@author: rajat
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as spst
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.special import factorial
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

# compute firing rate
def computeFiringRates(st, tstart, tend, tau):
    nCells = len(st)
    nTimeBins = int((tend-tstart)//tau)
    win = np.linspace(tstart, tend, nTimeBins)
    firingRates = np.zeros((nCells, nTimeBins-1))
    for i in range(nCells):
        firingRates[i, :] = np.histogram(st[i], win)[0]/tau
    return firingRates

# compute spike count
def computeSpikeCounts(st, tstart, tend, tau):
    nCells = len(st)
    nTimeBins = int((tend-tstart)/tau)
    win = np.linspace(tstart, tend, nTimeBins)
    spikeCounts = np.zeros((nTimeBins-1, nCells))
    for i in range(nCells):
        spikeCounts[:, i] = np.histogram(st[i], win)[0]/tau
    return spikeCounts

# poisson pdf
def poisspdf(x, lam):
    pdf = ((lam**x)*np.exp(-lam))/factorial(x)
    return pdf

# compute likelihood
def computeLikelihood(spkC, plFields, tau):
    pFields = (plFields * tau).T
    xyBins = plFields.shape[1]
    nTimeBins = spkC.shape[0]
    likelihood = np.zeros((xyBins, nTimeBins))
    for i in range(nTimeBins):
        nSpikes = np.array([spkC[i, :]]*xyBins)
        maxL = poisspdf(nSpikes, pFields)
        maxL = np.prod(maxL, 1)  # product
        likelihood[:, i] = maxL
    return likelihood


brain_region = 'iHC'
hallways = [1,2,3,4]
dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Justin_Data/VR19'
positionBins = np.arange(0,100,0.5)
    

cluidV2 = np.load(os.path.join(dirname,'AL1spikesorted/spikeClusterID.npy'), allow_pickle=True)
dfV2 = pd.read_csv(os.path.join(dirname,'AL1spikesorted/cluster_info.tsv'), delimiter='\t')
dfV2 = dfV2[dfV2['cluster_id'].isin(cluidV2)]
dfV2['region'] = ['V2']*len(dfV2)
dfV2['region'] = np.where((dfV2.region=='V2') & (dfV2.depth<=350),'iHC',dfV2.region)
cell_idxV2 = np.where(dfV2['region']=='V2')[0]
cluidV1 = np.load(os.path.join(dirname,'AL2spikesorted/spikeClusterID.npy'), allow_pickle=True)
dfV1 = pd.read_csv(os.path.join(dirname,'AL2spikesorted/cluster_info.tsv'), delimiter='\t')
dfV1 = dfV1[dfV1['cluster_id'].isin(cluidV1)]
dfV1['region'] = ['V1']*len(dfV1)
dfV1['region'] = np.where((dfV1.region=='V1') & (dfV1.depth<=750),'iHC',dfV1.region)
cell_idxV1 = np.where(dfV1['region']=='V1')[0]

cluidV2 = np.load(os.path.join(dirname,'AL1spikesorted/spikeClusterID.npy'), allow_pickle=True)
dfV2 = pd.read_csv(os.path.join(dirname,'AL1spikesorted/cluster_info.tsv'), delimiter='\t')
dfV2 = dfV2[dfV2['cluster_id'].isin(cluidV2)]
dfV2['region'] = ['V2']*len(dfV2)
dfV2['region'] = np.where((dfV2.region=='V2') & (dfV2.depth<=350),'iHC',dfV2.region)
cell_idx1 = np.where(dfV2['region']=='iHC')[0]

cluidV1 = np.load(os.path.join(dirname,'AL2spikesorted/spikeClusterID.npy'), allow_pickle=True)
dfV1 = pd.read_csv(os.path.join(dirname,'AL2spikesorted/cluster_info.tsv'), delimiter='\t')
dfV1 = dfV1[dfV1['cluster_id'].isin(cluidV1)]
dfV1['region'] = ['V1']*len(dfV1)
dfV1['region'] = np.where((dfV1.region=='V1') & (dfV1.depth<=750),'iHC',dfV1.region)
cell_idx2 = np.where(dfV1['region']=='iHC')[0]


decodingError = []
for hallnum in hallways:
    # load rate maps data
    placeFieldsHC = np.load(os.path.join(dirname, 'analyzed', 'RatemapsHC','hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
    placeFieldsV2 = np.load(os.path.join(dirname, 'analyzed', 'RatemapsAL1','hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
    placeFieldsV2 = placeFieldsV2[cell_idxV2,:]
    placeFieldsV1 = np.load(os.path.join(dirname, 'analyzed', 'RatemapsAL2','hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
    placeFieldsV1 = placeFieldsV1[cell_idxV1,:]
    
    placeFields1 = np.load(os.path.join(dirname, 'analyzed', 'RatemapsAL1','hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
    placeFields1 = placeFields1[cell_idx1,:]
    placeFields2 = np.load(os.path.join(dirname, 'analyzed', 'RatemapsAL2','hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
    placeFields2 = placeFields2[cell_idx2,:]
    placeFieldsiHC = np.concatenate((placeFields1, placeFields2),0)
    
    print("Finished loading rate maps")
    
    # load behavior data
    hallwayTrialStartTs = []
    hallwayTrialEndTs = []
    behavdat = np.load(os.path.join(dirname, 'Behavior','hall'+str(hallnum)+'_occmap.npy'), allow_pickle=True).item()
    for b in behavdat:
        b = behavdat[b]
        hallwayTrialStartTs.append(b['intantime'][0])
        hallwayTrialEndTs.append(b['intantime'][-1])
    hallwayTrialStartTs = np.array(hallwayTrialStartTs)
    hallwayTrialEndTs = np.array(hallwayTrialEndTs)
    print("Finished loading behavior data")
    
    # load spiking data and sort it based on trial start and end
    spiketimesHC = np.load(os.path.join(dirname, 'HCspikesorted','spiketimes.npy'), allow_pickle=True)
    spiketimes1 = np.load(os.path.join(dirname, 'AL1spikesorted','spiketimes.npy'), allow_pickle=True)
    spiketimesV2 = spiketimes1[cell_idxV2]
    spiketimes2 = np.load(os.path.join(dirname, 'AL2spikesorted','spiketimes.npy'), allow_pickle=True)
    spiketimesV1 = spiketimes2[cell_idxV1]
    spiketimes1 = spiketimes1[cell_idx1]
    spiketimes2 = spiketimes2[cell_idx2]
    spiketimesiHC = np.concatenate((spiketimes1, spiketimes2))
    
    
    spiketimesHCTrial = []
    spiketimesV2Trial = []
    spiketimesV1Trial = []
    spiketimesiHCTrial = []
    for st, et in zip(hallwayTrialStartTs, hallwayTrialEndTs):
        spkt_trial_HC = []
        spkt_trial_iHC = []
        spkt_trial_V2 = []
        spkt_trial_V1 = []
        for spkt in spiketimesHC:
            spkt_trial_HC.append(spkt[np.where((spkt>=st) & (spkt<=et))[0]])
        for spkt in spiketimesV2:
            spkt_trial_V2.append(spkt[np.where((spkt>=st) & (spkt<=et))[0]])
        for spkt in spiketimesV1:
            spkt_trial_V1.append(spkt[np.where((spkt>=st) & (spkt<=et))[0]])
        for spkt in spiketimesiHC:
            spkt_trial_iHC.append(spkt[np.where((spkt>=st) & (spkt<=et))[0]])
        spiketimesHCTrial.append(np.array(spkt_trial_HC))
        spiketimesV2Trial.append(np.array(spkt_trial_V2))
        spiketimesV1Trial.append(np.array(spkt_trial_V1))
        spiketimesiHCTrial.append(np.array(spkt_trial_iHC))
    spiketimesHCTrial = np.array(spiketimesHCTrial)
    spiketimesV2Trial = np.array(spiketimesV2Trial)
    spiketimesV1Trial = np.array(spiketimesV1Trial)
    spiketimesiHCTrial = np.array(spiketimesiHCTrial)
    print("Finished loading spike times across trials")
    
    
    # decode the animal's position using the maximum likelihood estimate
    # different values of tau and three different trials
    tau = 1
    count = 1
    Ntrials = 40 #len(spiketimesHCTrial)
    plt.figure(figsize=(16, 16))
    for j in range(Ntrials):
        nTimeBins = int((hallwayTrialEndTs[j]-hallwayTrialStartTs[j])/tau)
        if nTimeBins>1:
            # compute spike count in each time bin
            spikeCountsHC = computeSpikeCounts(spiketimesHCTrial[j], hallwayTrialStartTs[j], hallwayTrialEndTs[j], tau)
            spikeCountsV2 = computeSpikeCounts(spiketimesV2Trial[j], hallwayTrialStartTs[j], hallwayTrialEndTs[j], tau)
            spikeCountsV1 = computeSpikeCounts(spiketimesV1Trial[j], hallwayTrialStartTs[j], hallwayTrialEndTs[j], tau)
            spikeCountsiHC = computeSpikeCounts(spiketimesiHCTrial[j], hallwayTrialStartTs[j], hallwayTrialEndTs[j], tau)
            # compute likelihood
            likelihoodHC = computeLikelihood(spikeCountsHC, placeFieldsHC, tau)
            likelihoodV2 = computeLikelihood(spikeCountsV2, placeFieldsV2, tau)
            likelihoodV1 = computeLikelihood(spikeCountsV1, placeFieldsV1, tau)
            likelihoodiHC = computeLikelihood(spikeCountsiHC, placeFieldsiHC, tau)
            if likelihoodHC.shape[1] and likelihoodV1.shape[1] and likelihoodV2.shape[1] and likelihoodiHC.shape[1]:
                index = np.argmax(likelihoodHC, 0)
                decodedXHC = positionBins[index]  # decoded data
                index = np.argmax(likelihoodV2, 0)
                decodedXV2 = positionBins[index]  # decoded data
                index = np.argmax(likelihoodV1, 0)
                decodedXV1 = positionBins[index]  # decoded data
                index = np.argmax(likelihoodiHC, 0)
                decodedXiHC = positionBins[index]  # decoded data
                windows = np.linspace(hallwayTrialStartTs[j], hallwayTrialEndTs[j], nTimeBins-1)
                windows2 =  np.linspace(hallwayTrialStartTs[j], hallwayTrialEndTs[j], nTimeBins)
                behav = behavdat[j]  # actual trajectory
                inds = np.digitize(behav['intantime'], windows2)
                trueX = behav['pos']
                trueXbinned, _, _ = spst.binned_statistic(behav['intantime'], values=trueX, statistic='mean', bins=windows2)
                plt.subplot(7,6,count)
                plt.plot(windows, decodedXHC, 'mediumorchid', linewidth=1.5)
                plt.plot(windows, decodedXV2, 'lime', linewidth=1.5)
                plt.plot(windows, decodedXV1, 'gold', linewidth=1.5)
                plt.plot(windows, decodedXiHC, 'fuchsia', linewidth=1.5)
                plt.plot(behav['intantime'], trueX, 'k', linewidth=0.5)
                plt.xticks([])
                plt.yticks([0,100], [0, 530])
                count += 1
    plt.suptitle('Hallway: '+str(hallnum))
    plt.savefig(os.path.join('opPV','Hall'+str(hallnum)+'.png'))
