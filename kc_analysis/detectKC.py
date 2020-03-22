#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 09:47:05 2020

@author: rajat
"""

import mea
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detect_peaks import detect_peaks

# sampling rate
fs=1500.
# dt
dt = 1./fs
# load the raw eeg data
eeg_sig = np.load('kc.npy')
minThreshTime = .02 # minimum KC duration threshold in seconds
maxThreshTime = .2 # maximum KC duration threshold in seconds
        
# get the timestamps
times = np.arange(0, len(eeg_sig)*dt, dt)
# apply notch filter to remove 60hz
eeg_sig = mea.notch(eeg_sig, 60., 2, fs)
# band pass in low freq band
filt_sig_delta = mea.butter_bandpass_filter(eeg_sig,0.1,7,fs,order=2)  
# amplitude threshold
amp_thr = np.mean(filt_sig_delta) + 2.25*np.std(filt_sig_delta)

# function to detect peaks in the delta signal range 
kc_peakindices_delta = detect_peaks(-filt_sig_delta, mph=amp_thr, mpd=int(fs/4.))
kc_peakamp_delta = filt_sig_delta[kc_peakindices_delta]
kc_peakamp_sig = eeg_sig[kc_peakindices_delta]
# set falloff threshold to detect KC event boundaries
falloffThresh = np.nanmean(kc_peakamp_sig) + 4*np.std(kc_peakamp_sig)

#find_start_end(peakindices_delta)
kc_peak_idx =  np.array([],dtype='int')
kc_start_idx =  np.array([],dtype='int')
kc_end_idx =  np.array([],dtype='int')
kc_peak_time =  np.array([],dtype='int')
kc_start_time =  np.array([],dtype='int')
kc_end_time =  np.array([],dtype='int')
kc_duration =  np.array([],dtype='int')
kc_peak_amp =  np.array([],dtype='int')
dur = []
# iterate over all the KC peak
for r, pind, amp in zip(range(len(kc_peakindices_delta)), kc_peakindices_delta, kc_peakamp_delta):
    thresholded_sig = min(abs(0.05*amp), falloffThresh)
    # find the left and right falloff points for individual KC
    idx_max = int(pind+fs)
    idx_min = int(pind-fs)
    # check on boundary condition
    if idx_min<0:
        idx_min = 0
    if idx_max>len(filt_sig_delta):
        idx_max=len(filt_sig_delta)-1
    # select the subset of KC event to find boundaries
    filt_sig_delta_sub = filt_sig_delta[idx_min:idx_max]
    idx_falloff = np.where(filt_sig_delta_sub>=falloffThresh)[0]
    idx_falloff += idx_min
    # find the start and end index for individual ripple
    _, startidx = mea.find_le(idx_falloff, pind)
    _, endidx = mea.find_ge(idx_falloff, pind)
    # accounting for some edge conditions by throwing them out
    if startidx is not None and endidx is not None:
        # duration CHECK!
        duration = np.round(times[endidx]-times[startidx],3)
        dur.append(duration)
        if duration<=1.25 and duration>=0.01:
            kc_start_idx = np.append(kc_start_idx,startidx)
            kc_end_idx = np.append(kc_end_idx,endidx)
            kc_peak_idx = np.append(kc_peak_idx,pind)
            kc_start_time = np.append(kc_start_time,times[startidx])
            kc_end_time = np.append(kc_end_time,times[endidx])
            kc_peak_time = np.append(kc_peak_time,times[pind])
            kc_duration = np.append(kc_duration,duration)
            kc_peak_amp = np.append(kc_peak_amp,filt_sig_delta[pind])
    # else:
    #     plt.figure()
    #     plt.plot(filt_sig_delta_sub)
    #     plt.show()
kc_peak_idx = np.array(kc_peak_idx)
kc_start_idx = np.array(kc_start_idx)
kc_end_idx = np.array(kc_end_idx)
kc_peak_time = np.array(kc_peak_time)
kc_start_time = np.array(kc_start_time)
kc_end_time = np.array(kc_end_time)
kc_duration = np.array(kc_duration)
kc_peak_amp = np.array(kc_peak_amp)

# calculate the spectrogram for the entire signal 
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16,8))
# try to maximize on the time resolution (frequnecy resolution can be compromised for this)
Pxx, freqs, T, im = ax3.specgram(eeg_sig, NFFT=512, Fs=fs, noverlap=448)
# calculate the sum of frequency from >40hz 
Pxx_sum = np.sum(Pxx[15:],axis=0)
ax3.set_title('Spectrogram')
# set a high gamma powert threshold
hg_thr = np.mean(Pxx_sum) - 0.35*np.std(Pxx_sum)
# remove the KC peaks that do not correspond to a dip in power 
drop_index= []
for pi in range(len(kc_peak_idx)):
    start_t, end_t = times[int(kc_start_idx[pi])], times[int(kc_end_idx[pi])]
    start_i, start_T = mea.find_ge(T, start_t-0.01)
    end_i, end_T = mea.find_ge(T, end_t+0.01)
    # check for the dip in power corresponding to KC peak
    is_low_hgpower = np.where(Pxx_sum[start_i:end_i]<hg_thr)[0]
    if len(is_low_hgpower)<1:
        drop_index.append(pi)
drop_index = np.unique(drop_index)
# drop the KC peak not passing the HG power threshold
kc_peak_idx_upd = np.delete(kc_peak_idx, drop_index)
kc_start_idx_upd = np.delete(kc_start_idx, drop_index)
kc_end_idx_upd = np.delete(kc_end_idx, drop_index)
kc_peak_time_upd = np.delete(kc_peak_time, drop_index)
kc_start_time_upd = np.delete(kc_start_time, drop_index)
kc_end_time_upd = np.delete(kc_end_time, drop_index)
kc_duration_upd = np.delete(kc_duration, drop_index)
kc_peak_amp_upd = np.delete(kc_peak_amp, drop_index)
# plot the raw signal 
ax1.plot(times,eeg_sig)
ax1.set_title('Raw Signal')
# plot the delta filtered signal 
ax2.plot(times,filt_sig_delta)
ax2.axhline(y=-amp_thr,xmin=times[0],xmax=times[-1],c='k')
ax2.set_title('Filtered Delta signal')
# plot the delta filtered signal times and peak amplitudes 
ax2.scatter(times[kc_peak_idx_upd],kc_peak_amp_upd,c='r',s=10)
# plot the KC peak events 
for ind_ in range(len(kc_peak_idx_upd)):
    rect = patches.Rectangle((kc_start_time_upd[ind_],np.nanmin(filt_sig_delta)),
                                      kc_end_time[ind_] - kc_start_time[ind_],
                                      np.nanmax(filt_sig_delta) - np.nanmin(filt_sig_delta),
                                      linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
    ax2.add_patch(rect)
# plot the power threshold 
ax4.plot(T, Pxx_sum)
ax4.axhline(y=hg_thr,xmin=0,xmax=times[-1],c='k')
ax4.set_title('Filtered Power sum HG')
ax4.set_ylim([0, 1000])
ax4.set_xlim([times[0], times[-1]])
plt.show()
