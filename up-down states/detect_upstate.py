#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:06:15 2019

@author: rajat
"""
import mea
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detect_peaks import detect_peaks

# load the raw lfp data
eeg_sig = np.load('lfp_sleep.npy',mmap_mode='r')
eeg_sig = eeg_sig[0,:]
fsl = 1500.
dt = 1./fsl
eeg_time = np.arange(0, len(eeg_sig)*dt, dt)

# calculate spectrogram to find delta/theta ratio
F,T,P = mea.get_spectrogram(eeg_sig, fsl, freq_band=(0,50), norm=False)
P_rel = np.nansum(P[:9],0)/np.nansum(P[14:25],0)
# find non-rem and rem events
is_nrem = np.zeros(len(T))
# find low delta events or down-states
nrem_th = np.nanmean(P_rel) + 0.25*np.nanstd(P_rel)
nrem_idx = np.where(P_rel>=nrem_th)[0]
is_nrem[nrem_idx] = 50
# smooth and reupdate nrem epochs
# HACK JOB! needs to be improved
is_nrem = nd.gaussian_filter1d(is_nrem,4)
is_nrem = nd.gaussian_filter1d(is_nrem,5)
nrem_idx = np.where(is_nrem>=10)[0]
is_nrem[nrem_idx] = 50
rem_idx = np.where(is_nrem<10)[0]
is_nrem[rem_idx] = 0
# find start and end time for nrem
is_nremdiff = np.ediff1d(is_nrem)
nrem_idx = np.where((is_nremdiff==-50) | (is_nremdiff==50))[0] + 1
nrem_start_time = [0]
nrem_end_time = []
rem_start_time = []
rem_end_time = []
for i in range(0,len(nrem_idx)-1,2):
    if T[nrem_idx[i+1]]<3000: # endtime for the session
        nrem_end_time.append(T[nrem_idx[i]])
        rem_start_time.append(T[nrem_idx[i]])
        nrem_start_time.append(T[nrem_idx[i+1]])
        rem_end_time.append(T[nrem_idx[i+1]])
nrem_end_time.append(3000)

# load the raw spiking data
spike_clusters = np.ravel(np.load('spike_clusters.npy',mmap_mode='r'))
spike_times = np.ravel(np.load('spike_times.npy',mmap_mode='r'))

# convert the channel position to microns
channel_positions = np.load('channel_positions.npy')
channel_positionsX = channel_positions[:,0]*0.001
channel_positionsY = channel_positions[:,1]
# load the channel map
channel_map = np.ravel(np.load('channel_map.npy'))

# load cluster information
cluster_info = pd.read_csv('cluster_info.tsv',delimiter='\t')
cluster_id = np.array(cluster_info['id'])
cluster_cnum = np.array(cluster_info['ch'])
cluster_depth = np.array(cluster_info['depth'])*0.001
cluster_num_spike = np.array(cluster_info['n_spikes'])

# load cluster quality data and final sorted cluster amplitude and firing rate
good_clusters_id = cluster_id[np.where(cluster_info['group']=='good')[0]]
good_cluster_info = cluster_info.loc[cluster_info['id'].isin(good_clusters_id)]
good_clusters_depth = list(good_cluster_info['depth']*0.001)
good_clusters_cnum = list(good_cluster_info['ch'])
good_clusters_channel_index = []
for cnum in good_clusters_cnum:
    good_clusters_channel_index.append((np.where(cnum==channel_map)[0][0]))
good_clusters_xpos = list(channel_positionsX[good_clusters_channel_index])
del cluster_info

# load the spiking data and plotting it according to the depth
spiketimes = []
fs=30000.
total_time = np.max(spike_times)/fs
for cluid in good_clusters_id:
    spike_t = spike_times[np.where(spike_clusters==cluid)[0]]/fs
    idx = np.where(spike_t>=6000)[0]
    spike_t = spike_t[idx]
    spiketimes.append(spike_t)
spiketimes = np.array(spiketimes)
del spike_times

# calculate binned sum spikes
dt_s = 0.025
fs_s = 1/dt_s
spike_time_bins = np.arange(6000,total_time, dt_s)
spike_counts = np.zeros((len(spiketimes), len(spike_time_bins)-1))
for ind, st in enumerate(spiketimes):
    spike_counts[ind,:], _ = np.histogram(st, bins=spike_time_bins)
sum_spike_counts = np.sum(spike_counts, axis=0)
spike_time_bins = spike_time_bins - 6000

# find up-down states
sum_spike_counts_sm = nd.gaussian_filter1d(sum_spike_counts,2)
up_state_thresh = 2.5 # 5spikes/50ms
falloff_thresh = 1 # falloff = 1spike/bin
minThreshTime = 0.1
maxThreshTime = 6
# variables to hold the upstate data
upstate_start_idx =  np.array([],dtype='int')
upstate_end_idx =  np.array([],dtype='int')
upstate_start_time =  np.array([],dtype='int')
upstate_end_time =  np.array([],dtype='int')
upstate_duration =  np.array([],dtype='int')
# iterate over each NREM epochs
for st,et in zip(nrem_start_time,nrem_end_time):
    idx = np.where((spike_time_bins>=st) & (spike_time_bins<=et))[0]
    sum_spike_counts_nrem = sum_spike_counts_sm[idx]
    spike_time_bins_nrem = spike_time_bins[idx]
    idx_upstate_peak = detect_peaks(sum_spike_counts_nrem, mph=up_state_thresh, mpd=0.5/dt_s)
    # iterate over each peak
    for idx in idx_upstate_peak:
        # nice trick: no point looking beyond +/- 5s of the peak
        # since the upstate probably is not longer than that
        idx_max = idx + int(5*fs_s)
        idx_min = idx - int(5*fs_s)
        # boundary check
        if idx_min<0:
            idx_min = 0
        # find the left and right falloff points for individual ripple
        sum_spike_nrem_sub = sum_spike_counts_nrem[idx_min:idx_max]
        idx_falloff = np.where(sum_spike_nrem_sub<=falloff_thresh)[0]
        idx_falloff += idx_min
        # find the start and end index for individual ripple
        _, startidx = mea.find_le(idx_falloff, idx)
        _, endidx = mea.find_ge(idx_falloff, idx)
        # accounting for some boundary conditions
        # HACK JOB! Needs to be updated
        if startidx is not None and endidx is not None:
            # duration CHECK!
            dur = spike_time_bins_nrem[endidx]-spike_time_bins_nrem[startidx]
            # print(time[idx], time[endidx], time[startidx], dur, dur>=minThreshTime and dur<=maxThreshTime)
            # add the ripple to saved data if it passes duration threshold
            if dur>=minThreshTime and dur<=maxThreshTime:
                upstate_duration = np.append(upstate_duration, dur)
                upstate_start_idx = np.append(upstate_start_idx, startidx)
                upstate_end_idx = np.append(upstate_end_idx, endidx)
                upstate_start_time = np.append(upstate_start_time, spike_time_bins_nrem[startidx])
                upstate_end_time = np.append(upstate_end_time, spike_time_bins_nrem[endidx])
                
# plot the analyzed data
_extent = [np.min(eeg_time-eeg_time[0]), np.max(eeg_time), np.min(F), np.max(F)]
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12,18))
ax1.plot(eeg_time, eeg_sig, label='Raw')
ax1.set_ylabel("Amplitude", fontsize=16)
ax1.set_xlim([np.min(eeg_time-eeg_time[0]), np.max(eeg_time)])
for ind_ in range(len(nrem_start_time)):
    rect = patches.Rectangle((nrem_start_time[ind_],np.nanmin(eeg_sig)),
                             nrem_end_time[ind_] - nrem_start_time[ind_],
                             np.nanmax(eeg_sig) - np.nanmin(eeg_sig),
                             linewidth=1,edgecolor='none',alpha=0.5,facecolor='g')
    ax1.add_patch(rect)
for ind_ in range(len(rem_start_time)):    
    rect = patches.Rectangle((rem_start_time[ind_],np.nanmin(eeg_sig)),
                             rem_end_time[ind_] - rem_start_time[ind_],
                             np.nanmax(eeg_sig) - np.nanmin(eeg_sig),
                             linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
    ax1.add_patch(rect)
ax2.imshow(P,aspect='auto',extent=_extent,origin='lower', cmap='Spectral_r')
ax2.plot(T, is_nrem/5, 'w')
ax2.set_ylabel("Frequency", fontsize=16)
ax2.set_xlim([np.min(T), 2700])
ax2.set_ylim([0,20])
for i, spiketrain in enumerate(spiketimes):
    t = np.array(spiketrain)-6000
    ax3.plot(t, i * np.ones_like(t), 'k.', markersize=.5)
ax3.set_ylabel('Cell ID', fontsize=16)
ax4.plot(spike_time_bins[:-1], sum_spike_counts_sm, c='k')
ax4.set_xlabel('Time [s]', fontsize=16)
ax4.set_ylabel('Spike Sum', fontsize=16)
for ind_ in range(len(upstate_start_time)):
    rect = patches.Rectangle((upstate_start_time[ind_],0),
                             upstate_end_time[ind_] - upstate_start_time[ind_],
                             len(good_clusters_cnum),
                             linewidth=1,edgecolor='none',alpha=0.25,facecolor='m')
    ax3.add_patch(rect)
    rect1 = patches.Rectangle((upstate_start_time[ind_],0),
                             upstate_end_time[ind_] - upstate_start_time[ind_],
                             np.nanmax(sum_spike_counts_sm),
                             linewidth=1,edgecolor='none',alpha=0.25,facecolor='m')
    ax4.add_patch(rect1)
plt.show()
