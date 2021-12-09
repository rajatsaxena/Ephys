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

cell_idx = None
if brain_region=='AL1':
    cluidV2 = np.load(os.path.join(dirname,'AL1spikesorted/spikeClusterID.npy'), allow_pickle=True)
    dfV2 = pd.read_csv(os.path.join(dirname,'AL1spikesorted/cluster_info.tsv'), delimiter='\t')
    dfV2 = dfV2[dfV2['cluster_id'].isin(cluidV2)]
    dfV2['region'] = ['V2']*len(dfV2)
    dfV2['region'] = np.where((dfV2.region=='V2') & (dfV2.depth<=350),'iHC',dfV2.region)
    cell_idx = np.where(dfV2['region']=='V2')[0]
elif brain_region=='AL2':
    cluidV1 = np.load(os.path.join(dirname,'AL2spikesorted/spikeClusterID.npy'), allow_pickle=True)
    dfV1 = pd.read_csv(os.path.join(dirname,'AL2spikesorted/cluster_info.tsv'), delimiter='\t')
    dfV1 = dfV1[dfV1['cluster_id'].isin(cluidV1)]
    dfV1['region'] = ['V1']*len(dfV1)
    dfV1['region'] = np.where((dfV1.region=='V1') & (dfV1.depth<=750),'iHC',dfV1.region)
    cell_idx = np.where(dfV1['region']=='V1')[0]
elif brain_region=='iHC':
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
else:
    cell_idx = None


decodingError = []
for hallnum in hallways:
    # load rate maps data
    if brain_region!='iHC':
        placeFields = np.load(os.path.join(dirname, 'analyzed', 'Ratemaps'+brain_region,'hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
    else:
        placeFields1 = np.load(os.path.join(dirname, 'analyzed', 'RatemapsAL1','hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
        placeFields1 = placeFields1[cell_idx1,:]
        placeFields2 = np.load(os.path.join(dirname, 'analyzed', 'RatemapsAL2','hall'+str(hallnum)+'_ratemap.npy'), allow_pickle=True)
        placeFields2 = placeFields2[cell_idx2,:]
        placeFields = np.concatenate((placeFields1, placeFields2),0)
    if cell_idx is not None:
        placeFields = placeFields[cell_idx,:]
    cell_fr_order = np.argmax(placeFields, axis=1)
    cell_fr_order = np.argsort(cell_fr_order)
    placeFieldsSorted = placeFields[cell_fr_order]
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
    positionBins = np.arange(0,100)
    print("Finished loading behavior data")
    
    # load spiking data and sort it based on trial start and end
    if brain_region!='iHC':
        spiketimes = np.load(os.path.join(dirname, brain_region+'spikesorted','spiketimes.npy'), allow_pickle=True)
    else:
        spiketimes1 = np.load(os.path.join(dirname, 'AL1spikesorted','spiketimes.npy'), allow_pickle=True)
        spiketimes1 = spiketimes1[cell_idx1]
        spiketimes2 = np.load(os.path.join(dirname, 'AL2spikesorted','spiketimes.npy'), allow_pickle=True)
        spiketimes2 = spiketimes2[cell_idx2]
        spiketimes = np.concatenate((spiketimes1, spiketimes2))
    if cell_idx is not None:
        spiketimes = spiketimes[cell_idx]
    spiketimesTrial = []
    for st, et in zip(hallwayTrialStartTs, hallwayTrialEndTs):
        spkt_trial = []
        for spkt in spiketimes:
            spkt_trial.append(spkt[np.where((spkt>=st) & (spkt<=et))[0]])
        spkt_trial = np.array(spkt_trial)
        spiketimesTrial.append(spkt_trial)
    spiketimesTrial = np.array(spiketimesTrial)
    print("Finished loading spike times acros trials")
    
    
    # decode the animal's position using the maximum likelihood estimate
    # different values of tau and three different trials
    tau = 0.5
    decodingErrorHall = []
    count = 1
    plt.figure(figsize=(16, 16))
    for j in range(len(spiketimesTrial)):
        nTimeBins = int((hallwayTrialEndTs[j]-hallwayTrialStartTs[j])/tau)
        if nTimeBins>1:
            # compute spike count in each time bin
            spikeCounts = computeSpikeCounts(spiketimesTrial[j], hallwayTrialStartTs[j], hallwayTrialEndTs[j], tau)
            # compute likelihood
            likelihood = computeLikelihood(spikeCounts, placeFields, tau)
            if likelihood.shape[1]:
                index = np.argmax(likelihood, 0)
                # get decoded position
                decodedX = positionBins[index]  # decode data
                windows = np.linspace(hallwayTrialStartTs[j], hallwayTrialEndTs[j], nTimeBins-1)
                windows2 =  np.linspace(hallwayTrialStartTs[j], hallwayTrialEndTs[j], nTimeBins)
                behav = behavdat[j]  # actual trajectory
                inds = np.digitize(behav['intantime'], windows2)
                trueX = behav['pos']
                trueXbinned, _, _ = spst.binned_statistic(behav['intantime'], values=trueX, statistic='mean', bins=windows2)
                decodingErrorHall.append((np.nansum((trueXbinned - decodedX)**2))**(0.5))
                plt.subplot(12,12,count)
                plt.plot(windows, decodedX, 'r')
                plt.plot(behav['intantime'], trueX, 'b')
                plt.xticks([])
                count += 1
    plt.suptitle('Hallway: '+str(hallnum))
    plt.show()
    decodingError.append(np.array(decodingErrorHall))
decodingError = np.array(decodingError)