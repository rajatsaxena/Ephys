#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:51:56 2019

@author: rajat
"""

from utilsACG import temporalCrossCorrelogram
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def func(x, a, b, c, d, e, f, g, h):
    return max(c*(np.exp(-(x-f)/a)-d*np.exp(-(x-f)/b))+h*np.exp(-(x-f)/g)+e,0)

a0 = np.array([20, 1, 30, 2, 0.5, 5, 1.5, 2])
lb = np.array([1, 0.1, 0, 0, -30, 0,0.1, 0])
ub = np.array([500, 50, 500, 15, 50, 20,5,100])

spike_times = np.load('./dataHC/spiketimes.npy', allow_pickle=True)
bins = np.linspace(-1,1,2001)
# convert to ms
for i in range(len(spike_times)):
    st1 = np.array(spike_times[i]*1000, dtype=np.float)
    acg = np.asarray(temporalCrossCorrelogram.getBinnedISI(st1, len(st1), st1, len(st1)))
    acg[1001] = 0
    acg = acg[1002:1101]
    bins = bins[1002:1101]
    acg = np.divide(acg, np.nanmax(acg))
    popt, pcov = curve_fit(func, bins, acg, p0=a0, bounds=(lb, ub))
    acgfit = func(bins, *popt)
    
    plt.figure()
    plt.plot(bins, acg)
    plt.plot(bins, acgfit)
    plt.show()