#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:59:10 2021

@author: rajat
"""

import numpy as np
import pandas as pd
import mea, glob, os
import scipy.io as spio
import mea, bisect, h5py
import scipy.ndimage as scnd
from natsort import natsorted
from scipy.spatial import distance
from collections import Counter
from scipy.special import factorial

#Another binary search function to find the nearest value to a list of timestamps given pivot value
#INPUT: list of timestamps, pivot timestamp to search for 
#OUTPUT: closest matching timestamps index and valux    
def binarySearch(data,val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) / 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind] 
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind, data[best_ind]

#Find rightmost value less than or equal to x
def find_le(a, x):
    i = bisect.bisect_right(a, x)
    if i:
        return i-1, a[i-1]
    else:
        return np.nan, np.nan

#Find leftmost item greater than or equal to x
def find_ge(a, x):
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i, a[i]
    else:
        return np.nan, np.nan

# get spike sum for all units
def getSpikeSum(spktimes, timebin):
    spkcount = []
    for i, spkt in enumerate(spktimes):
        c, _ = np.histogram(spkt, timebin)
        spkcount.append(c)
    spkcount = np.array(spkcount)
    return np.nansum(spkcount,0)

# channel with maxunits for a shank
def getMaxUnitChan(count, minChan, maxChan):
    maxele = 0
    maxkey = None
    for ele in count.items():
        if ele[0]>=minChan and ele[0]<=maxChan:
            if ele[1]>=maxele:
                maxkey = ele[0]
                maxele = ele[1]
    return maxkey

# find nearest channel for each max channel
def getNearestChan(chanmap, chanpositions, channum):
    dist = distance.cdist(chanpositions, chanpositions, 'euclidean')
    nearestChan = []
    for c in channum:
        idx = np.where(chanmap==c)[0]
        nearestChan.append(chanmap[np.argsort(dist[idx])[0][:5]])
    return nearestChan

# get a channel with maximum units per shank and 4 of its neighboring channels
def getChannelPerShank(dirname):
    channel_positions = np.load(dirname+'/channel_positions.npy')
    channel_map = np.ravel(np.load(dirname+'/channel_map.npy'))
    # load cluster information
    cluster_info = pd.read_csv(dirname+'/cluster_info.tsv',delimiter='\t')
    cluster_id = np.array(cluster_info['cluster_id'])
    
    # load cluster quality data and final sorted cluster amplitude and firing rate
    good_clusters_id = cluster_id[np.where(cluster_info['group']=='good')[0]]
    good_cluster_info = cluster_info.loc[cluster_info['cluster_id'].isin(good_clusters_id)]
    good_clusters_cnum = np.array(good_cluster_info['ch'])
    del cluster_info
    
    # get chan with max units per shank
    unitsPerChan = Counter(good_clusters_cnum)
    shank1chan = getMaxUnitChan(unitsPerChan, 0, 31)
    shank2chan = getMaxUnitChan(unitsPerChan, 32, 63)
    shank3chan = getMaxUnitChan(unitsPerChan, 64, 95)
    shank4chan = getMaxUnitChan(unitsPerChan, 96, 127)
    
    return getNearestChan(channel_map, channel_positions, [shank1chan, shank2chan, shank3chan, shank4chan])

# bandpass filter signals
def filterdata(data, Fs=1500., frange=[150,250]):
    # print(str(1))
    return mea.get_bandpass_filter_signal(data, Fs, frange)

# bandpass filter signals2
def filterdata2(data, Fs=1500., frange=[6,10]):
    return mea.get_bandpass_filter_signal(data, Fs, frange)


# bandpass filter signals2
def filterdata3(data, Fs=1500., frange=[3, 6]):
    return mea.get_bandpass_filter_signal(data, Fs, frange)

# 1d smoothing matlab
def gaussianSmoothing(data, Fs=1500., winlength = 0.032, std=0.004):
    n = int(winlength*Fs)
    sigma = int(std*Fs)
    r = range(-int(n/2),int(n/2)+1)
    g = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]
    return np.convolve(data, g, mode='same' )

# function to generate normalized 1d array
def norm1d(arr):
    arr = arr-np.nanmin(arr)
    return arr/np.nanmax(arr)

# get spectrogram
def getSpectrogram(data, Fs=1500, freq_band=(0,250)):
    P, F, T = mea.get_spectrogram(data, Fs, freq_band=freq_band)
    relP = mea.get_relative_power(P, F)
    return [P, F, T, relP]

# multiple arg pass
def multi_run_wrapper(args):
    return calcPhase(*args)

# multiple arg pass
def multi_run_wrapper_tmi(args):
    return calcshuffleTMI(*args)

# find phase for each spike
def phase_assignment(thetapeaktime,spiketimestamps,lfpsignal,fs):
    #variable to hold spike phase
    spikephase = [] 
    #iterate over each spike
    for i in range(0,len(spiketimestamps)):
        #assign phase=nan if it does not cross the threshold
        #the second and condition is to account for that the spike occured after the 0th theta peak timestamps or before the last peak timestamps
        if spiketimestamps[i]>=thetapeaktime[0] and spiketimestamps[i]<=thetapeaktime[-1]:
            #find before and after thetapeak timestamps nearest to the ith spike
            _, thetaPeakBefore = find_le(thetapeaktime,spiketimestamps[i])
            _, thetaPeakAfter = find_ge(thetapeaktime,spiketimestamps[i])
            #calculate the interpeak diff
            interpeak_diff = thetaPeakAfter - thetaPeakBefore
            #if theta peak after and theta peak before are same assign spike phase=0
            if thetaPeakAfter==thetaPeakBefore:
                spikephase.append(0.0)
            #this is to account for the fact that the theta peak interval is within 1/6hz to 1/12 i.e. 0.16 to 0.08
            #added a epsilon for now 0.2 to 0.05
            #everyone has to play with this as per their data
            elif 0.085 < interpeak_diff < 0.18:
                #calculate by linear interpolation
                # phase = ((spiketime - thetapeakbefore)/inter_peak_interval)*360
                phase = round((float(spiketimestamps[i]-thetaPeakBefore)/interpeak_diff)*360,3)
                spikephase.append(phase) 
            #if the inter peak difference is outside assign nan
            else:
                spikephase.append(np.nan)
        #phase=nan did not cross threshold
        else:
            spikephase.append(np.nan)
    return spikephase


# phase calculation for each spike
def calcPhase(spiketimes, pktime, theta, fs):
    phase = []
    for spk in spiketimes:
        phase.append(phase_assignment(pktime, spk, theta, fs))
    phase = np.array(phase)
    return phase

# get shuffled TMI
def calcshuffleTMI(spiketimes, tmi):
    shuffledTMI = []
    for i in range(1000):
        phase = np.random.uniform(0,360,len(spiketimes))
        tmi_cell = getTMI(phase)
        shuffledTMI.append(tmi_cell[0])
    shuffledTMI = np.array(shuffledTMI)
    p = np.nansum(tmi<=shuffledTMI)/len(shuffledTMI)
    return p

# load cell type and brain region information
def loadCellMetrics(filename, clufile, goodcluid, region):
    df = pd.read_csv(clufile,delimiter='\t')
    df = df.loc[df['cluster_id'].isin(goodcluid)]
    
    if filename is not None:
        f = h5py.File(filename, 'r')
        cell_metrics = f.get('/cell_metrics')
        celltype = cell_metrics['putativeCellType']
        thmod = cell_metrics['thetaModulationIndex']
        cellType = []
        thetaMod = []
        for c in range(len(celltype)):
            cellt = f[cell_metrics['putativeCellType'][c, 0]].value.tobytes()[::2].decode()
            cluid = cell_metrics['cluID'][c, 0]
            if cluid in goodcluid:
                thetaMod.append(thmod[c])
                if 'Pyramidal' in cellt:
                    cellType.append(0)
                elif 'Narrow' in cellt:
                    cellType.append(1)
                else:
                    cellType.append(2)
        cellType = np.array(cellType)
        brainRegion = np.array([region]*len(cellType))
        df['thetamod'] = np.array(thetaMod)
    else:
        cellType = [np.nan]*len(df)
        brainRegion = [region]*len(df)
        
    df['celltype'] = cellType
    df['region'] = brainRegion
    return df

# calculate TMI
def getTMI(phase, bins=np.arange(0,4*360+15,15)):
    phase = np.array(phase)
    phase = np.concatenate((phase,phase+360,phase+2*360,phase+3*360,phase+4*360))
    count, edges = np.histogram(phase, bins)
    count = scnd.gaussian_filter1d(count,2)
    count = np.divide(count, np.nanmax(count))
    count = count[np.where((edges>=360) & (edges<=720))[0]]
    edges = edges[np.where((edges>=360) & (edges<=720))[0]]
    edges = edges - 360
    TMI = 1 - np.nanmin(count)
    prefPhase = edges[np.argmax(count)]
    return [TMI, prefPhase]

# find best theta channel for a given recording
def getBestThetaChannel(lfp_data, fs=1500):
    P, F, _ = mea.get_spectrogram(lfp_data,fs)
    return np.nanmean(mea.get_relative_power(P, F))

# load behavior data with timestamps aligned 
def loadBehavData(fnames, combined=False):
    dfh1 = pd.read_csv(fnames[0])
    dfh2 = pd.read_csv(fnames[1])
    dfh3 = pd.read_csv(fnames[2])
    dfh4 = pd.read_csv(fnames[3])
    dfh1 = dfh1[['position', 'intantime', 'speed']]
    dfh1['hall'] = [1]*len(dfh1)
    dfh2 = dfh2[['position', 'intantime', 'speed']]
    dfh2['hall'] = [2]*len(dfh2)
    dfh3 = dfh3[['position', 'intantime', 'speed']]
    dfh3['hall'] = [3]*len(dfh3)
    dfh4 = dfh4[['position', 'intantime', 'speed']]
    dfh4['hall'] = [4]*len(dfh4)
    dfBehav = pd.concat((dfh1,dfh2,dfh3,dfh4))
    dfBehav = dfBehav.sort_values(by = 'intantime')
    speed = np.ediff1d(dfBehav['position'])/np.ediff1d(dfBehav['intantime'])
    speed = np.insert(speed,0,np.nan)
    speed[np.where(speed<0)[0]] = np.nan
    dfBehav['speed'] = speed
    if combined:
        return dfBehav
    else:
        return dfh1, dfh2, dfh3, dfh4

# function to find ripple using Deshmukh Lab's method
def findRipple(signal, times, fs, ripple_power, spksumtime, spksum, spdtime, speed, f_ripple=(150,250), duration=[0.015,0.5], 
                 lookaheadtime=0.5, peakTh=4, falloffTh=1, activityRatio=0.35, speedTh=20):
    # calculate mean and standard deviation of the ripple power
    mean_rms = np.nanmean(ripple_power)
    std_rms = np.nanstd(ripple_power)

    minThreshTime = duration[0] # minimum duration threshold in seconds
    maxThreshTime = duration[1] # maximum duration threshold in seconds
    ripplePowerThresh = mean_rms + peakTh*std_rms #peak power threshold
    falloffThresh = mean_rms + falloffTh*std_rms

    # data to hold the variables
    ripple_duration = []
    ripple_start_time = []
    ripple_start_idx = []
    ripple_end_time = []
    ripple_end_idx = []
    ripple_peak_time = []
    ripple_peak_idx = []
    ripple_peak_amp = []
    ripple_act = []
    
    # iterate to find the ripple peaks
    idx=0
    while idx < len(times):
        # exclude first and last 300ms data
        if idx/fs>=0.3 and idx/fs<=times[-1]-0.3 and ripple_power[idx] >= ripplePowerThresh:
            # nice trick: no point looking beyond +/- 300ms of the peak
            # since the ripple cannot be longer than that
            idx_max = idx + int(lookaheadtime*fs)
            idx_min = idx - int(lookaheadtime*fs)
            # find the left and right falloff points for individual ripple
            ripple_power_sub = ripple_power[idx_min:idx_max]
            idx_falloff = np.where(ripple_power_sub<=falloffThresh)[0]
            idx_falloff += idx_min
            # find the start and end index for individual ripple
            _, startidx = mea.find_le(idx_falloff, idx)
            _, endidx = mea.find_ge(idx_falloff, idx)
            if startidx is not None and endidx is not None:    
                # duration CHECK!
                dur = times[endidx]-times[startidx]
                spkstind, _ = mea.find_le(spksumtime, times[startidx]-0.15)
                spketind, _ = mea.find_ge(spksumtime, times[endidx]+0.15)
                cellactiveratio = np.nanmax(spksum[spkstind:spketind])/np.nanmax(spksum)
                spdstind, _ = mea.find_le(spdtime, times[startidx])
                spdetind, _ = mea.find_ge(spdtime, times[endidx])
                spd_ = speed[spdstind:spdetind]
                if len(spd_)>0:
                    spd_ = np.nanmin(spd_)
                else:
                    spd_ = 0
                if dur>=minThreshTime and dur<=maxThreshTime and cellactiveratio>=activityRatio and spd_<=speedTh:
                    ripple_duration.append(dur)
                    ripple_start_idx.append(startidx)
                    ripple_end_idx.append(endidx)
                    ripple_peak_idx.append(idx)
                    ripple_start_time.append(times[startidx])
                    ripple_end_time.append(times[endidx])
                    ripple_peak_time.append(times[idx])
                    ripple_peak_amp.append(ripple_power[idx])
                    ripple_act.append(cellactiveratio)
                idx = endidx+1
            else:
                idx+=1
        else:
            idx+=1
    ripple = {}
    ripple['amp'] = ripple_peak_amp
    ripple['duration'] = ripple_duration
    ripple['start_time'] = ripple_start_time
    ripple['end_time'] = ripple_end_time
    ripple['peak_time'] = ripple_peak_time
    # ripple['start_idx'] = ripple_start_idx
    # ripple['end_idx'] = ripple_end_idx
    # ripple['peak_idx'] = ripple_peak_idx
    ripple['duration'] = np.array(ripple_end_time) - np.array(ripple_start_time)
    ripple['act_ratio'] = ripple_act
    ripple = pd.DataFrame(ripple)
    return ripple

# multiple arg pass
# def multi_run_wrapper_swrmod(args):
#    return calcSWRmodulation(*args)

# function to calculate SWR modulation
def calcSWRmodulation(spktime, peak_ripple_time=None, window_time=1, 
                      pethbins=np.arange(-1,1.01,0.01), numShufIter=5000):
    if peak_ripple_time is None:
        # load ripple data and throw away all events within 500ms
        dfRipple = pd.read_csv('./opRipples/ripplesShank2.csv', index_col=0)
        et = np.array(dfRipple['end_time'][1:])
        st = np.array(dfRipple['start_time'][:-1])
        idx = np.where((et - st)<0.5)[0]
        dfRipple.drop(idx, inplace=True)
        peak_ripple_time = np.array(dfRipple['peak_time'])
        del dfRipple, et, st, idx
    realbinpeth = []
    shufbinpeth = []
    # for each swr epoch calculate peth and shuffled peth
    for pt in peak_ripple_time:
        spk = spktime[(spktime>pt-window_time) & (spktime<pt+window_time)]
        if len(spk):
            spk = spk - pt
            count, _ = np.histogram(spk, pethbins)
            realbinpeth.append(count)
            
            # run for 5000 iteration
            shufcount = []
            for itr in range(numShufIter):
                spkshuf = spk + np.random.uniform(-0.5,0.5,1)[0]
                count, _ = np.histogram(spkshuf, pethbins)
                shufcount.append(count)
            shufbinpeth.append(np.array(shufcount))
    realbinpeth = np.array(realbinpeth, dtype='int')
    shufbinpeth = np.array(shufbinpeth, dtype='int')
    
    # calculate swr modulation index
    stidx = np.where(pethbins>=0)[0][0]
    etidx = np.where(pethbins<0.2)[0][-1]
    obsmod = np.nanmean(realbinpeth[:,stidx:etidx],0)
    shufmod = np.nanmean(shufbinpeth[:,:,stidx:etidx],0)
    shufmodmean = np.nanmean(shufmod,0)
    del shufbinpeth
    
    # get significance 
    obsmodidx = np.nansum(obsmod - shufmodmean)**2
    shufmodidx = np.nansum(shufmod - shufmodmean,1)**2
    pval = 1 - sum(obsmodidx>shufmodidx)/numShufIter
    
    # get modulation direction (1-200ms) - (-500ms+(-100ms))
    bstidx = np.where(pethbins>=-1)[0][0]
    betidx = np.where(pethbins<-0.5)[0][-1]
    diffwbaseline = np.nanmean(realbinpeth[:,stidx:etidx]) - np.nanmean(realbinpeth[:,bstidx:betidx])
    moddirection = diffwbaseline/np.nanmean(realbinpeth[:,bstidx:betidx])
    
    return [realbinpeth, pval, moddirection]


# find cells that are negative and positive modulated
def getSWRModDat(df, swrpeth, region1, region2=None, negative=True):    
    if negative:
        if region2 is not None:
            index = np.where((df['swrmoddir']<0) & ((df['region']==region1) | (df['region']==region2)))[0]
        else:
            index = np.where((df['swrmoddir']<0) & (df['region']==region1))[0]
    else:
        if region2 is not None:
            index = np.where((df['swrmoddir']>=0) & ((df['region']==region1) | (df['region']==region2)))[0]
        else:
            index = np.where((df['swrmoddir']>=0) & (df['region']==region1))[0]
    modunits = swrpeth[index,:]
    df = df.iloc[index]
    meanval = scnd.gaussian_filter1d(np.nanmean(modunits,0),4.5)
    meanval = meanval/np.nanmax(meanval)
    meanstd = np.nanstd(modunits,0)*0.35
    return meanval, meanstd

# get ratemaps across hallways 
def getRatemap(dirname, hnum):
    rmaps = []
    files = natsorted(glob.glob(os.path.join(dirname,'ClustId*_hall'+str(hnum)+'_processed.mat')))
    for j, filename in enumerate(files):
        data = spio.loadmat(filename)
        rmap = data['smoothratemap1d'][0]
        rmaps.append(rmap)
    rmaps = np.array(rmaps)
    return rmaps


# get sorted ratemaps across hallway
def getNormRatemaps(dirname, region='iHC'):
    # cluster id from V1 and V2
    cluidV1 = np.load(os.path.join(dirname,'AL2spikesorted/spikeClusterID.npy'), allow_pickle=True)
    cluidV2 = np.load(os.path.join(dirname,'AL1spikesorted/spikeClusterID.npy'), allow_pickle=True)
    
    # process cluster 
    dfV1 = pd.read_csv(os.path.join(dirname,'AL2spikesorted/cluster_info.tsv'), delimiter='\t')
    dfV1 = dfV1[dfV1['cluster_id'].isin(cluidV1)]
    dfV1['region'] = ['V1']*len(dfV1)
    dfV2 = pd.read_csv(os.path.join(dirname,'AL1spikesorted/cluster_info.tsv'), delimiter='\t')
    dfV2 = dfV2[dfV2['cluster_id'].isin(cluidV2)]
    dfV2['region'] = ['V2']*len(dfV2)
    dfV1['region'] = np.where((dfV1.region=='V1') & (dfV1.depth<=750),'iHC',dfV1.region)
    dfV2['region'] = np.where((dfV2.region=='V2') & (dfV2.depth<=350),'iHC',dfV2.region)
    
    spktime1 = np.load('./dataV2/spiketimes.npy', allow_pickle=True)
    spktime2 = np.load('./dataV1/spiketimes.npy', allow_pickle=True)
    
    if region=='iHC':
        clufname = os.path.join(dirname, 'analyzed/RatemapsAL1')
        rmaps_h1 = getRatemap(clufname, 1)
        rmaps_h2 = getRatemap(clufname, 2)
        rmaps_h3 = getRatemap(clufname, 3)
        rmaps_h4 = getRatemap(clufname, 4)
        
        # select cells that are only V2
        idx = np.where(dfV2['region']=='iHC')[0]
        rmaps_h1 = rmaps_h1[idx,:]
        rmaps_h2 = rmaps_h2[idx,:]
        rmaps_h3 = rmaps_h3[idx,:]
        rmaps_h4 = rmaps_h4[idx,:]
        spktime1 = spktime1[idx]
        
        clufname = os.path.join(dirname, 'analyzed/RatemapsAL2')
        rmaps2_h1 = getRatemap(clufname, 1)
        rmaps2_h2 = getRatemap(clufname, 2)
        rmaps2_h3 = getRatemap(clufname, 3)
        rmaps2_h4 = getRatemap(clufname, 4)
        
        # select cells that are only V2
        idx = np.where(dfV1['region']=='iHC')[0]
        rmaps2_h1 = rmaps2_h1[idx,:]
        rmaps2_h2 = rmaps2_h2[idx,:]
        rmaps2_h3 = rmaps2_h3[idx,:]
        rmaps2_h4 = rmaps2_h4[idx,:]
        spktime2 = spktime2[idx]
        
        rmaps_h1 = np.concatenate((rmaps_h1, rmaps2_h1),0)
        rmaps_h2 = np.concatenate((rmaps_h2, rmaps2_h2),0)
        rmaps_h3 = np.concatenate((rmaps_h3, rmaps2_h3),0)
        rmaps_h4 = np.concatenate((rmaps_h4, rmaps2_h4),0)
        
        del rmaps2_h1, rmaps2_h2, rmaps2_h3, rmaps2_h4, dfV1, dfV2
        
        spiketimes = np.concatenate((spktime1, spktime2))
    elif region=='V2':
        clufname = os.path.join(dirname, 'analyzed/RatemapsAL1')
        rmaps_h1 = getRatemap(clufname, 1)
        rmaps_h2 = getRatemap(clufname, 2)
        rmaps_h3 = getRatemap(clufname, 3)
        rmaps_h4 = getRatemap(clufname, 4)
        
        # select cells that are only V2
        idx = np.where(dfV2['region']=='V2')[0]
        rmaps_h1 = rmaps_h1[idx,:]
        rmaps_h2 = rmaps_h2[idx,:]
        rmaps_h3 = rmaps_h3[idx,:]
        rmaps_h4 = rmaps_h4[idx,:]
        spiketimes = spktime1[idx]
    elif region=='V1':
        clufname = os.path.join(dirname, 'analyzed/RatemapsAL2')
        rmaps_h1 = getRatemap(clufname, 1)
        rmaps_h2 = getRatemap(clufname, 2)
        rmaps_h3 = getRatemap(clufname, 3)
        rmaps_h4 = getRatemap(clufname, 4)
        
        # select cells that are only V2
        idx = np.where(dfV1['region']=='V1')[0]
        rmaps_h1 = rmaps_h1[idx,:]
        rmaps_h2 = rmaps_h2[idx,:]
        rmaps_h3 = rmaps_h3[idx,:]
        rmaps_h4 = rmaps_h4[idx,:]
        spiketimes = spktime2[idx]
    
    rmapsNorm_h1 = np.apply_along_axis(norm1d, 1, rmaps_h1)
    rmapsNorm_h2 = np.apply_along_axis(norm1d, 1, rmaps_h2)
    rmapsNorm_h3 = np.apply_along_axis(norm1d, 1, rmaps_h3)
    rmapsNorm_h4 = np.apply_along_axis(norm1d, 1, rmaps_h4)
    
    # get cell order
    # cellorder_h1 = np.argsort(np.argmax(rmapsNorm_h1,1))
    # cellorder_h2 = np.argsort(np.argmax(rmapsNorm_h2,1))
    # cellorder_h3 = np.argsort(np.argmax(rmapsNorm_h3,1))
    # cellorder_h4 = np.argsort(np.argmax(rmapsNorm_h4,1))
    
    # rmapsNorm_h1 =  rmapsNorm_h1[cellorder_h1]
    # rmapsNorm_h2 =  rmapsNorm_h2[cellorder_h2]
    # rmapsNorm_h3 =  rmapsNorm_h3[cellorder_h3]
    # rmapsNorm_h4 =  rmapsNorm_h4[cellorder_h4]
    
    return rmapsNorm_h1, rmapsNorm_h2, rmapsNorm_h3, rmapsNorm_h4, spiketimes


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
def computeSpikeCounts(st, tstart, tend, tau, win=None):
    nCells = len(st)
    nTimeBins = int((tend-tstart)/tau)
    if win is None:
        win = np.linspace(tstart, tend, nTimeBins*2)
    spikeCounts = np.zeros((len(win), nCells))
    for i in range(nCells):
        spkC = []
        for j in range(len(win)):
            c, _ = np.histogram(st[i], bins=[win[j],win[j]+tau])
            if len(c):
                spkC.append(c[0])
            else:
                spkC.append(0)
        spikeCounts[:, i] = np.array(spkC)
    return spikeCounts, win

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
    # normalization constant
    likelihood = likelihood/np.nansum(likelihood,0)
    return likelihood

# calculate binned speed and position for bayesian decoding
def calcBinnedStats(window, behavtimes, speed, pos, tau):
    binspeed = []
    realPos = []
    for w in range(len(window)):
        spd = speed[(behavtimes>=window[w]) & (behavtimes<=window[w]+tau)]
        ps = pos[(behavtimes>=window[w]) & (behavtimes<=window[w]+tau)]
        if len(spd):
            binspeed.append(np.nanmean(spd))
            realPos.append(np.nanmean(ps))
        else:
            binspeed.append(np.nan)
            realPos.append(np.nan)
    binspeed = np.array(binspeed)
    realPos = np.array(realPos)
    return binspeed, realPos
