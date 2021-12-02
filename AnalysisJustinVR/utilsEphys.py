#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:59:10 2021

@author: rajat
"""

import numpy as np
import pandas as pd
import mea, bisect, h5py
import scipy.ndimage as scnd
from scipy.spatial import distance
from collections import Counter

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

#Find leftmost item greater than or equal to x
def find_ge(a, x):
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i, a[i]

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

# 1d smoothing matlab
def gaussianSmoothing(data, Fs=1500., winlength = 0.032, std=0.004):
    n = int(winlength*Fs)
    sigma = int(std*Fs)
    r = range(-int(n/2),int(n/2)+1)
    g = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]
    return np.convolve(data, g, mode='same' )

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
            elif 0.05 < interpeak_diff < 0.2:
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
    
    df = pd.read_csv(clufile,delimiter='\t')
    df = df.loc[df['cluster_id'].isin(goodcluid)]
    df['celltype'] = cellType
    df['region'] = brainRegion
    df['thetamod'] = np.array(thetaMod)
    return df

# calculate TMI
def getTMI(phase, bins=np.arange(0,4*360+15,15)):
    phase = np.array(phase)
    phase = np.concatenate((phase,phase+360,phase+2*360,phase+3*360,phase+4*360))
    count, edges = np.histogram(phase, bins)
    count = scnd.gaussian_filter1d(count,1.5)
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
def loadBehavData(fnames):
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
    return dfBehav

# function to find ripple using Deshmukh Lab's method
def findRipple(signal, times, fs, ripple_power, f_ripple=(150,250), duration=[0.015,0.5], 
                 lookaheadtime=0.5, peakTh=4, falloffTh=1):
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
                if dur>=minThreshTime and dur<=maxThreshTime:
                    ripple_duration.append(dur)
                    ripple_start_idx.append(startidx)
                    ripple_end_idx.append(endidx)
                    ripple_peak_idx.append(idx)
                    ripple_start_time.append(times[startidx])
                    ripple_end_time.append(times[endidx])
                    ripple_peak_time.append(times[idx])
                    ripple_peak_amp.append(ripple_power[idx])
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
    ripple = pd.DataFrame(ripple)
    return ripple