#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:58:37 2019

@author: rajat
"""

# load the data
import numpy as np
import scipy.signal as scsig
import matplotlib.pyplot as plt
#import seaborn as sns

def generate_correlation_map(x, y):
    """Correlate each n with each m.
    x : Shape N X T.
    y : Shape M X T.

    Returns
    -------
      N X M array in which each element is a correlation coefficient.
    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

# filter the data bw 300 to 3khz
def butter_bandpass(lowcut,highcut,fs,order=8):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = scsig.butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=8):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)
    return np.around(np.array(scsig.filtfilt(b,a,data), dtype=np.float32), decimals=3)

# load the data
filename = 'Mouse_216_RUN2_190517_162838.npy'
amp_data = np.load(filename, mmap_mode='r')
fs = 20000.

# frequency range to filter in hz
f_low = 300
f_high = 6000  
filtered_amp_data = butter_bandpass_filter(amp_data,f_low, f_high, fs, order=4)
del amp_data

# remove common noise
common_signal = np.median(filtered_amp_data, axis=0)
amp_data_noise_rem = filtered_amp_data - common_signal
del filtered_amp_data, common_signal

# get the correlation matrix
correlation_matrix = generate_correlation_map(amp_data_noise_rem, amp_data_noise_rem)

# plotting the data
plt.imshow(correlation_matrix, cmap='viridis')
plt.colorbar()

g = sns.clustermap(correlation_matrix, cmap='viridis', xticklabels=True)
g.fig.set_figheight(15)
g.fig.set_figwidth(15)
plt.show()