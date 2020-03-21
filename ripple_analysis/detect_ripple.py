# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:02:39 2018

@author: Rajat Saxena
"""

import os, mea
import warnings
import numpy as np
import scipy.signal as spsig
import neuralynxio as nlxio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detect_peaks import detect_peaks
warnings.filterwarnings('ignore')

for filename in os.listdir(os.getcwd()):
    if 'CSC36' in filename and '.ncs' in filename:
        # load the data
        lfpdata = nlxio.load_ncs(filename)
        eeg_sig = lfpdata['data']
        time = lfpdata['time']/1e6 # convert to seconds
        time= time - time[0]
        fs = lfpdata['sampling_rate']
        del lfpdata
        
        # load the filtered ripple data
        f_ripple = (140,250)
        filt_rip_sig = mea.get_bandpass_filter_signal(eeg_sig, fs, f_ripple)
#        mea.plot_lfp(eeg_sig[:fs*1000], filt_rip_sig[:fs*1000], time[:fs*1000])
        # calculate the envelope of filtered data
        filt_rip_env = abs(spsig.hilbert(filt_rip_sig))
        filt_rip_env_zscore = mea.zscore(filt_rip_env)
         
        # Root mean square (RMS) ripple power calculation
        ripple_power = mea.window_rms(filt_rip_sig, 10)
        # calculate mean and standard deviation of the ripple power
        mean_rms = np.nanmean(ripple_power)
        std_rms = np.nanstd(ripple_power)
        
        minThreshTime = .02 # minimum duration threshold in seconds
        maxThreshTime = .2 # maximum duration threshold in seconds
        ripplePowerThresh = mean_rms + 5*std_rms #peak power threshold
        falloffThresh = mean_rms # falloffthresh = mean + .5*std_rms
        
        # data to hold the variables
        ripple_duration = []
        ripple_start_time = []
        ripple_end_time = []
        ripple_start_idx = []
        ripple_end_idx = []
        ripple_peak_time = []
        ripple_peak_idx = []
        ripple_peak_amp = []
        
        # data to hold the variables
        ripple_durationv2 = []
        ripple_start_timev2 = []
        ripple_end_timev2 = []
        ripple_start_idxv2 = []
        ripple_end_idxv2 = []
        ripple_peak_timev2 = []
        ripple_peak_idxv2 = []
        ripple_peak_ampv2 = []
        
        # ripple peak detection using ripple power threshold
        # find peaks using height as power thershold, min ripple distance = 200ms
        idx_peak = detect_peaks(ripple_power, mph=ripplePowerThresh, mpd=.1*fs)
        # iterate over each peak
        for idx in idx_peak:
            # nice trick: no point looking beyond +/- 300ms of the peak
            # since the ripple cannot be longer than that
            idx_max = idx + int(maxThreshTime*fs)
            idx_min = idx - int(maxThreshTime*fs)
            # boundary check
            if idx_min<0:
                idx_min = 0
            # find the left and right falloff points for individual ripple
            ripple_power_sub = ripple_power[idx_min:idx_max]
            idx_falloff = np.where(ripple_power_sub<=falloffThresh)[0]
            idx_falloff += idx_min
            # find the start and end index for individual ripple
            _, startidx = mea.find_le(idx_falloff, idx)
            _, endidx = mea.find_ge(idx_falloff, idx)
            # accounting for some boundary conditions
            if startidx is None:
                th = (maxThreshTime + minThreshTime)//2
                startidx = idx - int(th*fs)
            if endidx is None:
                endidx = 2*idx - startidx
            # duration CHECK!
            dur = time[endidx]-time[startidx]
#            print(time[idx], time[endidx], time[startidx], dur, dur>=minThreshTime and dur<=maxThreshTime)
            # add the ripple to saved data if it passes duration threshold
            if dur>=minThreshTime and dur<=maxThreshTime:
                ripple_duration.append(dur)
                ripple_start_idx.append(startidx)
                ripple_end_idx.append(endidx)
                ripple_peak_idx.append(idx)
                ripple_start_time.append(time[startidx])
                ripple_end_time.append(time[endidx])
                ripple_peak_time.append(time[idx])
                ripple_peak_amp.append(ripple_power[idx])
                
        # ripple peak detection using z-score threshold
        # find peaks using height as power thershold, min ripple distance = 200ms
        idx_peak2 = detect_peaks(filt_rip_env_zscore, mph=3.25, mpd=.1*fs)
        # iterate over each peak
        for idx in idx_peak2:
            # nice trick: no point looking beyond +/- 300ms of the peak
            # since the ripple cannot be longer than that
            idx_max = idx + int(maxThreshTime*fs)
            idx_min = idx - int(maxThreshTime*fs)
            # boundary check
            if idx_min<0:
                idx_min = 0
            # find the left and right falloff points for individual ripple
            filt_rip_env_zscore_sub = filt_rip_env_zscore[idx_min:idx_max]
            idx_falloff = np.where(filt_rip_env_zscore_sub<=0.15)[0]
            idx_falloff += idx_min
            # find the start and end index for individual ripple
            _, startidx = mea.find_le(idx_falloff, idx)
            _, endidx = mea.find_ge(idx_falloff, idx)
            # accounting for some boundary conditions
            if startidx is None:
                th = (maxThreshTime + minThreshTime)//2
                startidx = idx - int(th*fs)
            if endidx is None:
                endidx = 2*idx - startidx
            # duration CHECK!
            dur = time[endidx]-time[startidx]
#            print(time[idx], time[endidx], time[startidx], dur, dur>=minThreshTime and dur<=maxThreshTime)
            # add the ripple to saved data if it passes duration threshold
            if dur>=minThreshTime and dur<=maxThreshTime:
                ripple_durationv2.append(dur)
                ripple_start_idxv2.append(startidx)
                ripple_end_idxv2.append(endidx)
                ripple_peak_idxv2.append(idx)
                ripple_start_timev2.append(time[startidx])
                ripple_end_timev2.append(time[endidx])
                ripple_peak_timev2.append(time[idx])
                ripple_peak_ampv2.append(filt_rip_env_zscore[idx])
        
        # plot the processed data
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
        ax1.plot(time, eeg_sig, c='k')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],np.nanmin(eeg_sig)),
                                     ripple_end_time[ind_] - ripple_start_time[ind_],
                                     np.nanmax(eeg_sig) - np.nanmin(eeg_sig),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax1.add_patch(rect)
        ax1.set_ylabel('Raw signal', fontsize=16)
        ax2.plot(time, filt_rip_sig, c='r')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],np.nanmin(filt_rip_sig)),
                                     ripple_end_time[ind_] - ripple_start_time[ind_],
                                     np.nanmax(filt_rip_sig) - np.nanmin(filt_rip_sig),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax2.add_patch(rect)
        ax2.set_ylabel('Filtered signal', fontsize=16)
        ax3.plot(time, filt_rip_env_zscore, c='m')
        for ind_ in range(len(ripple_peak_timev2)):
            rect = patches.Rectangle((ripple_start_timev2[ind_],np.nanmin(filt_rip_env_zscore)),
                                     ripple_end_timev2[ind_] - ripple_start_timev2[ind_],
                                     np.nanmax(filt_rip_env_zscore) - np.nanmin(filt_rip_env_zscore),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='gray')
            ax3.add_patch(rect)
        ax3.set_ylabel('zscored filtered envelope', fontsize=16)
        ax4.plot(time, ripple_power, c='g')
        ax4.plot(time[ripple_peak_idx], ripple_power[ripple_peak_idx], 'b.')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],0),ripple_end_time[ind_] - ripple_start_time[ind_],np.nanmax(ripple_power),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax4.add_patch(rect)
        ax4.set_xlabel("Time [s]", fontsize=16)
        ax4.set_ylabel("Ripple power", fontsize=16)
        plt.show()
