# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 02:05:49 2021

@author: jshobe
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import resample

def calculate_waveform_metrics(waveforms, 
                               cluster_id, 
                               peak_channel, 
                               channel_map, 
                               sample_rate, 
                               upsampling_factor, 
                               spread_threshold,
                               site_spacing):
    snr = calculate_snr(waveforms[:, peak_channel, :])
    
    mean_2D_waveform = np.squeeze(np.nanmean(waveforms[:, channel_map, :], 0))
    local_peak = np.argmin(np.abs(channel_map - peak_channel))

    num_samples = waveforms.shape[2]
    new_sample_count = int(num_samples * upsampling_factor)

    mean_1D_waveform = resample(mean_2D_waveform[local_peak, :], new_sample_count)

    timestamps = np.linspace(0, num_samples / sample_rate, new_sample_count)

    duration = calculate_waveform_duration(mean_1D_waveform, timestamps)
    halfwidth = calculate_waveform_halfwidth(mean_1D_waveform, timestamps)
    PT_ratio = calculate_waveform_PT_ratio(mean_1D_waveform)
    repolarization_slope = calculate_waveform_repolarization_slope( mean_1D_waveform, timestamps)
    recovery_slope = calculate_waveform_recovery_slope(mean_1D_waveform, timestamps)

    data = [[cluster_id, peak_channel, snr, duration, halfwidth, PT_ratio, repolarization_slope, recovery_slope]]

    metrics = pd.DataFrame(data,
                           columns=['cluster_id', 'peak_channel', 'snr', 'duration', 'halfwidth',
                                     'PT_ratio', 'repolarization_slope', 'recovery_slope'])

    return metrics


# ==========================================================

# EXTRACTING 1D FEATURES

# ==========================================================


def calculate_snr(W):
    
    """
    Calculate SNR of spike waveforms.

    Converted from Matlab by Xiaoxuan Jia

    ref: (Nordhausen et al., 1996; Suner et al., 2005)

    Input:
    -------
    W : array of N waveforms (N x samples)

    Output:
    snr : signal-to-noise ratio for unit (scalar)

    """

    W_bar = np.nanmean(W, axis=0)
    A = np.max(W_bar) - np.min(W_bar)
    e = W - np.tile(W_bar, (np.shape(W)[0], 1))
    snr = A/(2*np.nanstd(e.flatten()))

    return snr


def calculate_waveform_duration(waveform, timestamps):
    
    """ 
    Duration (in seconds) between peak and trough

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)

    Outputs:
    --------
    duration : waveform duration in milliseconds

    """

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform)

    # to avoid detecting peak before trough
    if waveform[peak_idx] > np.abs(waveform[trough_idx]):
        duration =  timestamps[peak_idx:][np.where(waveform[peak_idx:]==np.min(waveform[peak_idx:]))[0][0]] - timestamps[peak_idx] 
    else:
        duration =  timestamps[trough_idx:][np.where(waveform[trough_idx:]==np.max(waveform[trough_idx:]))[0][0]] - timestamps[trough_idx] 

    return duration * 1e3


def calculate_waveform_halfwidth(waveform, timestamps):
    
    """ 
    Spike width (in seconds) at half max amplitude

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)

    Outputs:
    --------
    halfwidth : waveform halfwidth in milliseconds

    """

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform)

    try:
        if waveform[peak_idx] > np.abs(waveform[trough_idx]):
            threshold = waveform[peak_idx] * 0.5
            thresh_crossing_1 = np.min(
                np.where(waveform[:peak_idx] > threshold)[0])
            thresh_crossing_2 = np.min(
                np.where(waveform[peak_idx:] < threshold)[0]) + peak_idx
        else:
            threshold = waveform[trough_idx] * 0.5
            thresh_crossing_1 = np.min(
                np.where(waveform[:trough_idx] < threshold)[0])
            thresh_crossing_2 = np.min(
                np.where(waveform[trough_idx:] > threshold)[0]) + trough_idx

        halfwidth = (timestamps[thresh_crossing_2] - timestamps[thresh_crossing_1]) 

    except ValueError:

        halfwidth = np.nan

    return halfwidth * 1e3


def calculate_waveform_PT_ratio(waveform):

    """ 
    Peak-to-trough ratio of 1D waveform

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)

    Outputs:
    --------
    PT_ratio : waveform peak-to-trough ratio

    """

    trough_idx = np.argmin(waveform)

    peak_idx = np.argmax(waveform)

    PT_ratio = np.abs(waveform[peak_idx] / waveform[trough_idx])

    return PT_ratio


def calculate_waveform_repolarization_slope(waveform, timestamps, window=20):
    
    """ 
    Spike repolarization slope (after maximum deflection point)

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)
    window : int
        Window (in samples) for linear regression

    Outputs:
    --------
    repolarization_slope : slope of return to baseline (V / s)

    """

    max_point = np.argmax(np.abs(waveform))

    waveform = - waveform * (np.sign(waveform[max_point])) # invert if we're using the peak

    repolarization_slope = linregress(timestamps[max_point:max_point+window], waveform[max_point:max_point+window])[0]

    return repolarization_slope * 1e-6



def calculate_waveform_recovery_slope(waveform, timestamps, window=20):

    """ 
    Spike recovery slope (after repolarization)

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)
    window : int
        Window (in samples) for linear regression

    Outputs:
    --------
    recovery_slope : slope of recovery period (V / s)

    """

    max_point = np.argmax(np.abs(waveform))

    waveform = - waveform * (np.sign(waveform[max_point])) # invert if we're using the peak

    peak_idx = np.argmax(waveform[max_point:]) + max_point

    recovery_slope = linregress(timestamps[peak_idx:peak_idx+window], waveform[peak_idx:peak_idx+window])[0]

    return recovery_slope * 1e-6


# ==========================================================
