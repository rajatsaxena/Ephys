#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:21:27 2020

@author: rajat
"""

import utilsReplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matlab.engine
eng = matlab.engine.start_matlab()

# init variables
method='rankcorr'
fsl = 2000.
fs = 30000.
dt = 0.01
nBinsThresh = 4
nCellsPerEvt = 10

# load ripple data
rippleEvents = pd.read_csv('rippleAnalyzed.csv')
ripple_start_time = np.array(rippleEvents['start_time'])
ripple_end_time = np.array(rippleEvents['end_time'])
ripple_power = np.load('ripple_power.npy', allow_pickle=True)

# load spiking data
spikes = np.load('spiketimes_behav.npy', allow_pickle=True)
Qmat = np.load('Qbehav.npy', allow_pickle=True)
QmatTime = np.load('QbehavTime.npy', allow_pickle=True)
time = np.arange(0, len(ripple_power)*1./fsl, 1./fsl)    

# plot the ripple data
plot = False
if plot:
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax[0].plot(time, ripple_power)
    ax[0].set_ylim([0,100])
    for st,et in zip(ripple_start_time, ripple_end_time):
        ax[0].axvspan(st, et, alpha=0.3, color='gray')
    ax[0].set_ylabel('Ripple Power', fontsize=16)
    for i, sp in enumerate(spikes):
        ax[1].plot(sp, i * np.ones_like(sp), 'k.', markersize=.5)
    for st,et in zip(ripple_start_time, ripple_end_time):
        ax[1].axvspan(st, et, alpha=0.3, color='gray')
    ax[1].set_ylabel('Cell #', fontsize=16)
    ax[2].imshow(Qmat, extent=[0, QmatTime[-1], 0, Qmat.shape[0]], aspect='auto')
    ax[3].plot(QmatTime[:-1], np.sum(Qmat,0))
    ax[3].set_xlabel("Time [s]", fontsize=16)
    ax[3].set_ylabel('Sum Spike', fontsize=16)
    for st,et in zip(ripple_start_time, ripple_end_time):
        ax[3].axvspan(st, et, alpha=0.3, color='gray')
    ax[3].set_xlim([0, QmatTime[-1]])
    plt.show()

# load the raw template
template = np.load('hall1_ratemap.npy', allow_pickle=True)
# remove cells with 0 spikes in the template
keep = np.where(np.sum(template, 1)>0)[0]

# data to hold results
bayesRankOrder = np.full(len(rippleEvents), np.nan)
linearWeighted = np.full(len(rippleEvents), np.nan)
rankOrder = np.full(len(rippleEvents), np.nan)
pvals = np.full(len(rippleEvents), np.nan)
slope = np.full(len(rippleEvents), np.nan)
integral = np.full(len(rippleEvents), np.nan)
nCells = np.full(len(rippleEvents), np.nan)
nSpks = np.full(len(rippleEvents), np.nan)
eventDuration = np.full(len(rippleEvents), np.nan)
# data to hold id shuffling
bayesRankOrderShuf = np.full((len(rippleEvents),100), np.nan)
linearWeightedShuf = np.full((len(rippleEvents),100), np.nan)
rankOrderShuf = np.full((len(rippleEvents),100), np.nan)
pvalsShuf = np.full((len(rippleEvents),100), np.nan)
slopeShuf = np.full((len(rippleEvents),100), np.nan)
integralShuf = np.full((len(rippleEvents),100), np.nan)
# data to hold circular shuffling
bayesRankOrderCircShuf = np.full((len(rippleEvents),100), np.nan)
linearWeightedCircShuf = np.full((len(rippleEvents),100), np.nan)
rankOrderCircShuf = np.full((len(rippleEvents),100), np.nan)
pvalsCircShuf = np.full((len(rippleEvents),100), np.nan)
slopeCircShuf = np.full((len(rippleEvents),100), np.nan)
integralCircShuf = np.full((len(rippleEvents),100), np.nan)


# plt.figure(figsize=(12,14))
# iterate through the ripple events
for event, ripst, ripet in zip(np.arange(len(rippleEvents)),ripple_start_time, ripple_end_time):
    print("Event number: " + str(event))
    data, counts = utilsReplay.processReplayData(Qmat, QmatTime, [ripst, ripet], binsize=dt)    
    if data is not None and data.shape[1]>= nBinsThresh:
        # generate posterior probability matrix using template and event FRs
        Pr, prMax = utilsReplay.placeBayes(data[keep,:].T, template[keep,:], dt)
        # bad form... but some events have 0 spks in a particular time bin, 
        # doing this rather than filtering those events out
        Pr[np.isnan(Pr)] = 0 
        # bayesRankOrder
        bayesRankOrder[event], _ = utilsReplay.corrPearson(np.arange(len(prMax)), prMax)
        
        # bayes Linear Weighted
        Prmat = matlab.double(Pr.tolist())
        Pronesmat = matlab.double(np.ones((Pr.shape[0],1)).tolist())
        ret = eng.makeBayesWeightedCorr1(Prmat, Pronesmat, nargout=2)
        linearWeighted[event] = ret[0]
        
        # rank order
        idx = np.intersect1d(keep, np.where(np.sum(data, 1)>0)[0])
        _, _, ord_template = utilsReplay.sort_cells(template[idx,:])
        _, ord_firstSpk = utilsReplay.sort_rows(data[idx,:])
        rankOrder[event], pvals[event] = utilsReplay.corrPearson(ord_template,ord_firstSpk)        
        
        # radon transform
        if sum(~np.isnan(sum(Pr.T)))>5:
            PrTmat = matlab.double(Pr.T.tolist())
            ret = eng.Pr2RadonPy(PrTmat, nargout=6)
            slope[event], integral[event], x, y, xx, yy = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]
            x = x[0]
            y = y[0]
            xx = xx[0]
            yy = yy[0]
        else:
            slope[event], integral[event] = np.nan, np.nan
            
            
        # time to add some shuffling
        for i in range(100):
            ################################ cellid shuffling #####################################
            shuf = utilsReplay.shuffleCellID(template[keep,:]) 
            
            # get decoding
            Pr, prMax = utilsReplay.placeBayes(data[keep,:].T, shuf, dt); 
            Pr[np.isnan(Pr)] = 0 
            # bayesRankOrder
            bayesRankOrderShuf[event, i], _ = utilsReplay.corrPearson(np.arange(len(prMax)), prMax)
            
            # bayes Linear Weighted
            Prmat = matlab.double(Pr.tolist())
            Pronesmat = matlab.double(np.ones((Pr.shape[0],1)).tolist())
            ret = eng.makeBayesWeightedCorr1(Prmat, Pronesmat, nargout=2)
            linearWeightedShuf[event, i] = ret[0]
            
            # rank order
            idx = np.intersect1d(keep, np.where(np.sum(data, 1)>0)[0])
            _, _, ord_template = utilsReplay.sort_cells(shuf[idx,:])
            _, ord_firstSpk = utilsReplay.sort_rows(data[idx,:])
            rankOrderShuf[event, i], pvalsShuf[event, i] = utilsReplay.corrPearson(ord_template,ord_firstSpk)        
            
            # radon transform
            if sum(~np.isnan(sum(Pr.T)))>5:
                PrTmat = matlab.double(Pr.T.tolist())
                ret = eng.Pr2RadonPy(PrTmat, nargout=6)
                slopeShuf[event, i], integralShuf[event, i] = ret[0], ret[1]
            else:
                slopeShuf[event, i], integralShuf[event, i] = np.nan, np.nan
                
                
            #################### shuffle circular ###########################################################
            shuf = utilsReplay.shuffleCircular(template[keep,:])
            
            # get decoding
            Pr, prMax = utilsReplay.placeBayes(data[keep,:].T, shuf, dt); 
            Pr[np.isnan(Pr)] = 0 
            # bayesRankOrder
            bayesRankOrderCircShuf[event, i], _ = utilsReplay.corrPearson(np.arange(len(prMax)), prMax)
            
            # bayes Linear Weighted
            Prmat = matlab.double(Pr.tolist())
            Pronesmat = matlab.double(np.ones((Pr.shape[0],1)).tolist())
            ret = eng.makeBayesWeightedCorr1(Prmat, Pronesmat, nargout=2)
            linearWeightedCircShuf[event, i] = ret[0]
            
            # rank order
            idx = np.intersect1d(keep, np.where(np.sum(data, 1)>0)[0])
            _, _, ord_template = utilsReplay.sort_cells(shuf[idx,:])
            _, ord_firstSpk = utilsReplay.sort_rows(data[idx,:])
            rankOrderCircShuf[event, i], pvalsCircShuf[event, i] = utilsReplay.corrPearson(ord_template,ord_firstSpk)        
            
            # radon transform
            if sum(~np.isnan(sum(Pr.T)))>5:
                PrTmat = matlab.double(Pr.T.tolist())
                ret = eng.Pr2RadonPy(PrTmat, nargout=6)
                slopeCircShuf[event, i], integralCircShuf[event, i] = ret[0], ret[1]
            else:
                slopeCircShuf[event, i], integralCircShuf[event, i] = np.nan, np.nan
    
    nCells[event] = sum(sum(counts[keep,:])>0);
    nSpks[event] = sum(sum(counts[keep,:]))
    eventDuration[event] = ripet - ripst
    
    # plt.subplot(4,2,1)
    # plt.scatter(rankOrder,linearWeighted,c='k',marker='.')
    # plt.title('rank ord VS linear weighted')
    
    # plt.subplot(4,2,2)
    # plt.scatter(linearWeighted,bayesRankOrder,c='k',marker='.')
    # plt.title('linear weighted VS bayesian rank ord')
    
    # plt.subplot(4,2,3)
    # plt.scatter(rankOrder,integral,c='k',marker='.')
    # plt.title('rank ord VS radon in')
    
    # plt.subplot(4,2,4)
    # plt.scatter(linearWeighted,integral,c='k',marker='.')
    # plt.title('linear weighted VS radon int')
    
    # plt.subplot(4,2,5)
    # plt.scatter(rankOrder,slope,c='k',marker='.')
    # plt.title('rank ord VS radon slope')
    
    # plt.subplot(4,2,6)
    # plt.scatter(linearWeighted,slope,c='k',marker='.')
    # plt.title('linear weighted VS radon slope')
    
    # ax1 = plt.subplot(4,2,7)
    # plt.imshow(data, cmap='gray', aspect='auto')
    # plt.title(rankOrder[event])
    # plt.ylabel('position')
    # plt.xlabel(['time bins: ' + str(dt*1000) + ' ms'])
    # ax2 = ax1.twinx() 
    # plt.plot(np.mean(data,0), 'r')
    # plt.ylabel('# of spks',color='r')
    
    # ax3 = plt.subplot(4,2,8)
    # plt.imshow(Pr.T, cmap='gray', aspect='auto')
    # plt.xlabel(['time bins: ' + str(dt*1000) + ' ms'])
    # ax4 = ax3.twinx() 
    # plt.plot([x[0],x[1]], [y[0],y[1]])
    # plt.plot(xx,yy,'r')
    # plt.title(round(integral[event],3))
    # plt.ylabel('position')
    
    # plt.tight_layout()
    # plt.pause(.01)
    
    # del ax1, ax3, ax2, ax4
    # plt.clf()
    
print(np.corrcoef(rankOrder, bayesRankOrder)[0,1])
print(np.corrcoef(rankOrder, linearWeighted)[0,1])
print(np.corrcoef(rankOrder, slope)[0,1])
print(np.corrcoef(rankOrder, integral)[0,1])
print(np.corrcoef(bayesRankOrder, linearWeighted)[0,1])
print(np.corrcoef(bayesRankOrder, slope)[0,1])
print(np.corrcoef(bayesRankOrder, integral)[0,1])
print(np.corrcoef(linearWeighted, integral)[0,1])
print(np.corrcoef(linearWeighted, slope)[0,1])
print(np.corrcoef(slope, integral)[0,1])
