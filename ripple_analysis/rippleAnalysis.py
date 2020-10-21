#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:50:11 2020

@author: rajat
"""

import mea
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scnd
from detect_peaks import detect_peaks

# find best ripple channel for a given recording
def getBestRippleChannel(lfp_data):
    # vectorto hold mean and median ripple ratio
    mmRippleRatio = np.zeros(len(lfp_data))
    meanRipple = np.zeros(len(lfp_data))
    medianRipple = np.zeros(len(lfp_data))
    f_ripple = (150,250)
    # iterate through the channel number
    for cnum in range(lfp_data.shape[0]):
        eeg_sig = lfp_data[cnum,:]
        # filter in ripple range
        filt_rip_sig = mea.get_bandpass_filter_signal(eeg_sig, fs, f_ripple)
        # Root mean square (RMS) ripple power calculation
        ripple_power = mea.window_rms(filt_rip_sig, 10)
        # calculate mean and  median of the ripple power
        mean_rms = np.nanmean(ripple_power)
        median_rms = np.nanmedian(ripple_power)
        # calculate the mean and median ripple ratio
        meanRipple[cnum] = mean_rms
        medianRipple[cnum] = median_rms
        mmRippleRatio[cnum] = mean_rms/median_rms
        del eeg_sig, filt_rip_sig, ripple_power
    mmRippleRatio[meanRipple<1] = 0
    mmRippleRatio[medianRipple<1] = 0
    # find the best channel
    bestChannelNum = np.argmax(mmRippleRatio)
    return bestChannelNum, meanRipple, medianRipple, mmRippleRatio

# function to find ripple using Deshmukh Lab's method
def findRippleMK(signal, times, fs, f_ripple=(150,250), duration=[0.015,0.5], 
                 lookaheadtime=0.5, peakTh=4, falloffTh=0):
    # filter signal in ripple range
    filt_rip_sig = mea.get_bandpass_filter_signal(signal, fs, f_ripple)
    # Root mean square (RMS) ripple power calculation
    ripple_power = mea.window_rms(filt_rip_sig, 10)
    ripple_power = scnd.gaussian_filter1d(ripple_power, 0.004*fs)
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
                dur = times[endidx]-times[startidx]
                # duration CHECK!
                dur = time[endidx]-time[startidx]
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
    ripple['start_idx'] = ripple_start_idx
    ripple['end_idx'] = ripple_end_idx
    ripple['peak_idx'] = ripple_peak_idx
    ripple['duration'] = np.array(ripple_end_time) - np.array(ripple_start_time)
    ripple = pd.DataFrame(ripple)
    return ripple, filt_rip_sig, ripple_power

# function to find ripple using Deshmukh Lab's method
def findRippleMKv2(signal, times, fs, f_ripple=(150,250), duration=[0.015,1.0],  
                   lookaheadtime=0.5, peakTh=4, falloffTh=1):
    # filter signal in ripple range
    filt_rip_sig = mea.get_bandpass_filter_signal(signal, fs, f_ripple)
    # Root mean square (RMS) ripple power calculation
    ripple_power = mea.window_rms(filt_rip_sig, 10)
    # calculate mean and standard deviation of the ripple power
    mean_rms = np.nanmean(ripple_power)
    std_rms = np.nanstd(ripple_power)
    #peak power threshold
    amp_thr = mean_rms + peakTh*std_rms
    falloffThresh = mean_rms + falloffTh*std_rms

    # find the bounds of each ripple event
    swr_peakindices = detect_peaks(ripple_power, mph=amp_thr, mpd=fs*0.03)
    swr_peakamp = ripple_power[swr_peakindices]
    swr_peak_start_idx = np.array([])
    swr_peak_end_idx = np.array([])
    drop_index = []
    for r, pind, amp in zip(range(len(swr_peakindices)), swr_peakindices, swr_peakamp):
        try:
            if pind-fs>=0:
                sig_left = signal[pind-fs:pind]
                start_ind = int(pind - fs + np.max(np.where(sig_left>=falloffThresh)[0]))
            else:
                sig_left = signal[:pind]
                start_ind = int(np.max(np.where(sig_left>=falloffThresh)[0]))
            if pind+fs<=len(signal):
                sig_right = signal[pind:pind+fs]
            else:
                sig_right = signal[pind:]
            end_ind = int(pind + np.min(np.where(sig_right>=falloffThresh)[0]))
            swr_peak_start_idx = np.append(swr_peak_start_idx,int(start_ind))
            swr_peak_end_idx = np.append(swr_peak_end_idx,int(end_ind))
        except ValueError:
            drop_index.append(r)
    swr_peakindices = np.array(np.delete(swr_peakindices,drop_index), dtype=np.int64)
    swr_peakamp = np.delete(swr_peakamp,drop_index)
    swr_peak_start_idx = np.array(swr_peak_start_idx, dtype=np.int64)
    swr_peak_end_idx = np.array(swr_peak_end_idx, dtype=np.int64)
    swr_peaktime = times[swr_peakindices]
    swr_startime = times[swr_peak_start_idx]
    swr_endtime = times[swr_peak_end_idx]
    
    swr_duration = swr_endtime - swr_startime
    drop_index = []
    for i, d in enumerate(swr_duration):
        if d>=duration[0] and d<=duration[-1]:
            drop_index.append(i)
    swr_peakamp = np.delete(swr_peakamp,drop_index)
    swr_peakindices = np.delete(swr_peakindices,drop_index)
    swr_peak_start_idx = np.delete(swr_peak_start_idx,drop_index)
    swr_peak_end_idx = np.delete(swr_peak_end_idx,drop_index)
    swr_peaktime = np.delete(swr_peaktime,drop_index)
    swr_startime = np.delete(swr_startime,drop_index)
    swr_endtime = np.delete(swr_endtime,drop_index)
    
    ripple = {}
    ripple['amp'] = swr_peakamp
    ripple['start_time'] = swr_startime
    ripple['end_time'] = swr_endtime
    ripple['peak_time'] = swr_peaktime
    ripple['start_idx'] = swr_peak_start_idx
    ripple['end_idx'] = swr_peak_end_idx
    ripple['peak_idx'] = swr_peakindices
    ripple['duration'] = swr_endtime - swr_startime
    ripple = pd.DataFrame(ripple)
    # return the data
    return ripple, filt_rip_sig, ripple_power

# function to remove low spiking events
def getSpikeSumThreshold(spktimefname, times, fs, ripDf, dt=0.01):
    # load the raw data
    spiketimes = np.load(spktimefname, allow_pickle=True)
    # bin spike count in 10ms to get the Q matrix
    total_time = times[-1]
    spike_time_bins = np.arange(np.min(times)/fs,total_time, dt)
    spike_counts = np.zeros((len(spiketimes), len(spike_time_bins)-1))
    for ind, st in enumerate(spiketimes):
        spike_counts[ind,:], _ = np.histogram(st, bins=spike_time_bins)
    sum_spike_counts = np.sum(spike_counts, axis=0)
    threshold_spiking = np.nanmedian(sum_spike_counts) + np.nanstd(sum_spike_counts)
    
    isHSE = []
    countHSE  = []
    for r in range(len(ripDf)):
        rip = ripDf.iloc[r]
        stind, _ = mea.find_le(spike_time_bins, rip['start_time'])
        etind, _ = mea.find_ge(spike_time_bins, rip['end_time'])
        spkcount = np.nanmax(sum_spike_counts[stind:etind])
        countHSE.append(spkcount)
        isHSE.append(spkcount>=threshold_spiking)
    isHSE = np.array(isHSE)
    countHSE = np.array(countHSE)
    drop_index = np.where(isHSE==False)[0]
    return drop_index, countHSE, spike_counts, sum_spike_counts, spike_time_bins    
        
# function to get speed threshold
def getSpeedThreshold(halldata, ripDf, speedTh=3):
    # load speed and position data
    posSpeed = np.array([])
    posTime = np.array([])
    for hall in halldata:
        hallmap = np.load(hall, allow_pickle=True).item()
        for k in hallmap.keys():
            s = hallmap[k]['speed']
            t = hallmap[k]['intantime']
            if k==1 and len(posSpeed)<1 and len(posTime)<1:
                posSpeed = s
                posTime = t
            else:
                posSpeed = np.concatenate((posSpeed, s))
                posTime = np.concatenate((posTime, t))
    sortidx = np.argsort(posTime)
    posTime = posTime[sortidx]
    posSpeed = posSpeed[sortidx]
    
    isOffline = []
    speed = []
    for r in range(len(ripDf)):
        rip = ripDf.iloc[r]
        stind, _ = mea.find_le(posTime, rip['start_time'])
        etind, _ = mea.find_ge(posTime, rip['end_time'])
        speedrip = np.nanmin(posSpeed[stind:etind])
        speed.append(speedrip)
        isOffline.append(speedrip<=speedTh)
    isOffline = np.array(isOffline)
    speed = np.array(speed)
    drop_index = np.where(isOffline==False)[0]
    return drop_index, speed


# load the raw data
spiketimes = np.load('spiketimes_behav.npy', allow_pickle=True)
# sampling rate
fs = 2000.
# dt
dt = 1./fs
# load the filename
filename = 'SWIL3_lfp_data.npy'
lfp_sig = np.load(filename, mmap_mode='r')
time = np.arange(0, lfp_sig.shape[1]/fs, 1./fs)
# select behavior index
epoch_startidx = 0
epoch_endidx = np.where(time <= 5400)[0][-1]
lfp_sig = lfp_sig[:, epoch_startidx:epoch_endidx]                            
time = time[epoch_startidx:epoch_endidx]
# get the best channel using the mean to median ratio 
#bestChannel, mRipple, medRipple, mmRippleRatio = getBestRippleChannel(lfp_sig)  
bestChannel = 92 
# select the best channel lfp
lfp_sig = lfp_sig[bestChannel,:]

# find ripple data
rippleDf, ripple_filt_sig, ripple_power = findRippleMK(lfp_sig,time,int(fs))
# remove low spiking events
drop_index, countSpiking, ensembleSpikes, ensembleSpikesum, timeSpiking = getSpikeSumThreshold('spiketimes_behav.npy', time, fs, rippleDf)
rippleDf['spkSum'] = countSpiking
rippleDf = rippleDf.drop(drop_index)
rippleDf = rippleDf.reset_index(drop=True)

# save ripple data and other relevant data
rippleDf.to_csv('rippleAnalyzed.csv')
np.save('ripple_power.npy', ripple_power)
np.save('Qbehav.npy', ensembleSpikes)
np.save('QbehavTime.npy', timeSpiking)

# remove running epochs
hallwaydata = ['hall1_occmap.npy', 'hall2_occmap.npy', 'hall28_occmap.npy']
drop_index, runningSpeed = getSpeedThreshold(hallwaydata, rippleDf, speedTh=3)
rippleDf['speed'] = runningSpeed
rippleDf = rippleDf.drop(drop_index)
rippleDf = rippleDf.reset_index(drop=True)

# plot the processed data
fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
ax[0].plot(time, lfp_sig, linewidth=1)
ax[0].set_ylim([-750,750])
for st,et in zip(rippleDf['start_time'], rippleDf['end_time']):
    ax[0].axvspan(st, et, alpha=0.3, color='gray')
ax[0].set_ylabel('Raw signal', fontsize=16)
ax[1].plot(time, ripple_filt_sig)
ax[1].set_ylim([-100,100])
for st,et in zip(rippleDf['start_time'], rippleDf['end_time']):
    ax[1].axvspan(st, et, alpha=0.3, color='gray')
ax[1].set_ylabel('Filtered signal', fontsize=16)
ax[2].plot(time, ripple_power)
for st,et in zip(rippleDf['start_time'], rippleDf['end_time']):
    ax[2].axvspan(st, et, alpha=0.3, color='gray')
ax[2].set_ylabel('Ripple Power', fontsize=16)
ax[2].set_ylim([0,100])
for i, spiketrain in enumerate(spiketimes):
    ax[3].plot(spiketrain, i * np.ones_like(spiketrain), 'k.', markersize=.5)
for st,et in zip(rippleDf['start_time'], rippleDf['end_time']):
    ax[3].axvspan(st, et, alpha=0.3, color='gray')
ax[3].set_ylabel('Cell #', fontsize=16)
ax[4].plot(timeSpiking[:-1]-timeSpiking[0], ensembleSpikesum)
ax[4].set_xlabel("Time [s]", fontsize=16)
ax[4].set_xlim(time[0], time[-1])
ax[4].set_ylabel('Sum Spike', fontsize=16)
for st,et in zip(rippleDf['start_time'], rippleDf['end_time']):
    ax[4].axvspan(st, et, alpha=0.3, color='gray')
plt.show()
