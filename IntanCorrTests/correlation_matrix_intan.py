#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:58:37 2019

@author: rajat
"""

# load the data
#import h5py
import numpy as np
import scipy.signal as scsig
import matplotlib.pyplot as plt

# load the data
filename = 'Mouse_216_RUN2_190517_160339.npy'
amp_data = np.load('Mouse_216_RUN2_190517_160339.npy', mmap_mode='r')
fs = 20000.
#hf = h5py.File(filename,'r', driver='core')
#amp_data = hf['amp_data'][:]
#amp_data = np.transpose(amp_data)
#fs = hf['fs'][:]
#hf.close()

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

f_low = 300
f_high = 6000  
filtered_amp_data = butter_bandpass_filter(amp_data,f_low, f_high, fs, order=4)

del amp_data

# remove common noise
common_signal = np.median(filtered_amp_data, axis=0)
amp_data_noise_rem = filtered_amp_data - common_signal

del filtered_amp_data, common_signal
    
# calculate correlation coefficient 
correlation_matrix = np.zeros((256,256))
for i in range(256):
    for j in range(256):
        corr = np.corrcoef(amp_data_noise_rem[i,:], amp_data_noise_rem[j,:])
        correlation_matrix[i,j] = corr[0][1]

# plotting the data
plt.imshow(correlation_matrix, cmap='jet')
plt.colorbar()