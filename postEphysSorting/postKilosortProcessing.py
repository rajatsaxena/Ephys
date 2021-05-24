#!/usr/bin/env python
# coding: utf-8

# ## import functions

# In[1]:


import os
import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter1d


# ## Util functions to calculate ISI violations, firing rate, presence ration, ampltide cutoff

# In[2]:


def calculate_isi_violations(spike_times, spike_clusters, total_units, isi_threshold, min_isi):
    cluster_ids = np.unique(spike_clusters)
    viol_rates = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        viol_rates[idx], num_violations = isi_violations(spike_times[for_this_cluster],
                                                       min_time = np.min(spike_times[for_this_cluster]),
                                                       max_time = np.max(spike_times[for_this_cluster]),
                                                       isi_threshold=isi_threshold,
                                                       min_isi = min_isi)
    return viol_rates


def calculate_firing_rate(spike_times, spike_clusters, total_units):
    cluster_ids = np.unique(spike_clusters)
    firing_rates = np.zeros((total_units,))
    min_time = np.min(spike_times)
    max_time = np.max(spike_times)
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        firing_rates[idx] = firing_rate(spike_times[for_this_cluster],
                                        min_time = np.min(spike_times),
                                        max_time = np.max(spike_times))
    return firing_rates


def calculate_presence_ratio(spike_times, spike_clusters, total_units):
    cluster_ids = np.unique(spike_clusters)
    ratios = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        ratios[idx] = presence_ratio(spike_times[for_this_cluster],
                                                       min_time = np.min(spike_times),
                                                       max_time = np.max(spike_times))
    return ratios


def calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units):
    cluster_ids = np.unique(spike_clusters)
    amplitude_cutoffs = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        amplitude_cutoffs[idx] = amplitude_cutoff(amplitudes[for_this_cluster])
    return amplitude_cutoffs


def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
    """Calculate ISI violations for a spike train.
    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz
    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    isi_threshold : threshold for isi violation
    min_isi : threshold for duplicate spikes
    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a fpRate = 0
        A unit with some contamination has a fpRate < 0.5
        A unit with lots of contamination has a fpRate > 1.0
    num_violations : total number of violations
    """

    duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]

    spike_train = np.delete(spike_train, duplicate_spikes + 1)
    isis = np.diff(spike_train)

    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2*num_spikes*(isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, min_time, max_time)
    violation_rate = num_violations/violation_time
    fpRate = violation_rate/total_rate
    return fpRate, num_violations


def firing_rate(spike_train, min_time = None, max_time = None):
    """Calculate firing rate for a spike train.
    If no temporal bounds are specified, the first and last spike time are used.
    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    min_time : float
        Time of first possible spike (optional)
    max_time : float
        Time of last possible spike (optional)
    Outputs:
    --------
    fr : float
        Firing rate in Hz
    """
    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)
    fr = spike_train.size / duration
    return fr


def presence_ratio(spike_train, min_time, max_time, num_bins=100):
    """Calculate fraction of time the unit is present within an epoch.
    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking
    """
    h, b = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))
    return np.sum(h > 0) / num_bins


def amplitude_cutoff(amplitudes, num_histogram_bins = 500, histogram_smoothing_value = 3):
    """ Calculate approximate fraction of spikes missing from a distribution of amplitudes
    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)
    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible
    """
    h,b = np.histogram(amplitudes, num_histogram_bins, density=True)
    pdf = gaussian_filter1d(h,histogram_smoothing_value)
    support = b[:-1]
    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index
    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:])*bin_size
    fraction_missing = np.min([fraction_missing, 0.5])
    return fraction_missing


# ## cutoff threshold 

# In[3]:


params = {}
params['isi_threshold']=0.0015
params['min_isi']=0.00
params['isi_viol_th']=0.2 #20% violations
params['presence_ratio']=0.4
params['firing_rate_th']=0.5 #0.5Hz
params['amp_cutoff_th']=0.01
params['amp_th']=25 #25uV


# ## load the data 

# In[4]:


fs = 30000.0
spike_times = np.ravel(np.load('spike_times.npy', allow_pickle=True))/fs
spike_clusters = np.ravel(np.load('spike_clusters.npy', allow_pickle=True))
spike_templates = np.ravel(np.load('spike_templates.npy', allow_pickle=True))
amplitudes = np.ravel(np.load('amplitudes.npy', allow_pickle=True))
templates = np.load('templates.npy')
channel_map = np.load('channel_map.npy')[0]
cluster_info = pd.read_csv('cluster_info.tsv', sep='\t')
total_units = len(np.unique(spike_clusters))
epoch = [1000, np.inf]
if epoch[0]==np.inf:
    in_epoch = (spike_times <= epoch[-1])
elif epoch[-1]==np.inf:
    in_epoch = (spike_times >= epoch[0])
else:
    in_epoch = (spike_times > epoch[0]) * (spike_times < epoch[-1])
metrics = pd.DataFrame()


# # Calculate unit quality metrics

# In[ ]:


print("Calculating isi violations")
isi_viol = calculate_isi_violations(spike_times[in_epoch], spike_clusters[in_epoch], total_units, params['isi_threshold'], params['min_isi'])


# In[ ]:


print("Calculating presence ratio")
presence_ratio = calculate_presence_ratio(spike_times[in_epoch], spike_clusters[in_epoch], total_units)


# In[ ]:


print("Calculating firing rate")
firing_rate = calculate_firing_rate(spike_times[in_epoch], spike_clusters[in_epoch], total_units)


# In[ ]:


print("Calculating amplitude cutoff")
amplitude_cutoff = calculate_amplitude_cutoff(spike_clusters[in_epoch], amplitudes[in_epoch], total_units)


# In[ ]:


cluster_ids = np.unique(spike_clusters)
epoch_name = ['Experiment'] * len(cluster_ids)


# ## finalize the metrics

# In[ ]:


metrics = pd.concat((metrics, pd.DataFrame(data= OrderedDict((('cluster_id', cluster_ids),
                               ('firing_rate' , firing_rate),
                               ('presence_ratio' , presence_ratio),
                               ('isi_viol' , isi_viol),
                               ('amp_cutoff' , amplitude_cutoff),
                               ('epoch_name' , epoch_name),
                               )))))


# In[ ]:


metrics['group'] = cluster_info['group']
metrics['depth'] = cluster_info['depth']
metrics['ch'] = cluster_info['ch']
metrics['num_spikes'] = cluster_info['n_spikes']
metrics['amp'] = cluster_info['amp']


# ## find good cell based on cutoff

# In[ ]:


isGoodCell = (metrics['isi_viol']<params['isi_viol_th']) & (metrics['amp_cutoff']<params['amp_cutoff_th']) & (metrics['amp']>params['amp_th']) & (metrics['presence_ratio']>params['presence_ratio']) & (metrics['firing_rate']>params['firing_rate_th']) & (metrics['group']=='good')
metrics['isGood'] = isGoodCell


# ## save the metrics

# In[ ]:


fname = os.path.join(os.getcwd(),'VR9UnitMetrics.csv')
metrics.to_csv(fname, index= True)

