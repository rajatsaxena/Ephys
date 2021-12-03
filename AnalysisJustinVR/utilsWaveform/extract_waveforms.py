# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 01:56:07 2021

@author: jshobe
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
from waveform_metrics import calculate_waveform_metrics


def extract_waveforms(raw_data, 
                      spike_times, 
                      spike_clusters, 
                      templates, 
                      cluster_ids,
                      peak_channels,
                      channel_map, 
                      bit_volts, 
                      sample_rate, 
                      site_spacing, 
                      params):
    
    # #############################################

    samples_per_spike = params['samples_per_spike']
    pre_samples = params['pre_samples']
    spikes_per_epoch = params['spikes_per_epoch']
    upsampling_factor = params['upsampling_factor']
    spread_threshold = params['spread_threshold']

    # #############################################

    metrics = pd.DataFrame()

    total_units = len(cluster_ids)
    
    mean_waveforms = np.zeros((total_units, 2, raw_data.shape[1], samples_per_spike))
    firing_rate = np.zeros((total_units, 1))

    for cluster_idx, cluster_id in enumerate(cluster_ids):
        print(cluster_idx)
        in_cluster = (spike_clusters == cluster_id)

        if np.sum(in_cluster) > 0:
            times_for_cluster = spike_times[in_cluster]
            
            firing_rate[cluster_idx] = len(times_for_cluster)/(np.nanmax(times_for_cluster) - np.nanmin(times_for_cluster))

            waveforms = np.empty((spikes_per_epoch, raw_data.shape[1], samples_per_spike))
            waveforms[:] = np.nan
            np.random.shuffle(times_for_cluster)

            total_waveforms = np.min([times_for_cluster.size, spikes_per_epoch])

            for wv_idx, peak_time in enumerate(times_for_cluster[:total_waveforms]):
                start = int(peak_time-pre_samples)
                end = start + samples_per_spike
                rawWaveform = raw_data[start:end, :].T

                # in case spike was at start or end of dataset
                if rawWaveform.shape[1] == samples_per_spike:
                    waveforms[wv_idx, :, :] = rawWaveform * bit_volts

            # concatenate to existing dataframe
            metrics = pd.concat([metrics, calculate_waveform_metrics(waveforms[:total_waveforms, :, :],
                                                                     cluster_id, 
                                                                     peak_channels[cluster_idx], 
                                                                     channel_map,
                                                                     sample_rate, 
                                                                     upsampling_factor,
                                                                     spread_threshold,
                                                                     site_spacing,
                                                                     )])
    
            
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_waveforms[cluster_idx, 0, :, :] = np.nanmean(waveforms, 0)
                mean_waveforms[cluster_idx, 1, :, :] = np.nanstd(waveforms, 0)

                # remove offset
                for channel in range(0, mean_waveforms.shape[3]):
                    mean_waveforms[cluster_idx, 0, channel, :] = mean_waveforms[cluster_idx, 0, channel, :] - mean_waveforms[cluster_idx, 0, channel, 0]

    return mean_waveforms, metrics


def load(folder, filename):
    return np.load(os.path.join(folder, filename))

def read_cluster_group_tsv(filename):
    info = pd.read_csv(filename, sep='\t')
    cluster_ids = info['cluster_id'].values.astype('int')
    cluster_quality = info['group'].values
    return cluster_ids, cluster_quality

def load_kilosort_data(folder,  sample_rate = None, convert_to_seconds = True, template_zero_padding= 21):
    spike_times = load(folder,'spike_times.npy')
    spike_clusters = load(folder,'spike_clusters.npy')
    spike_templates = load(folder, 'spike_templates.npy')
    templates = load(folder,'templates.npy')
    unwhitening_mat = load(folder,'whitening_mat_inv.npy')
    channel_map = np.squeeze(load(folder, 'channel_map.npy'))
    cluster_ids, cluster_quality = read_cluster_group_tsv(os.path.join(folder, 'cluster_group.tsv'))
    cluster_info = pd.read_csv(os.path.join(folder, 'cluster_info.tsv'), delimiter='\t')
    
    templates = templates[:,template_zero_padding:,:] # remove zeros
    spike_clusters = np.squeeze(spike_clusters) # fix dimensions
    spike_templates = np.squeeze(spike_templates) # fix dimensions
    spike_times = np.squeeze(spike_times)# fix dimensions

    if convert_to_seconds and sample_rate is not None:
       spike_times = spike_times / sample_rate 
       
    unwhitened_temps = np.zeros((templates.shape))
    for temp_idx in range(templates.shape[0]):
        unwhitened_temps[temp_idx,:,:] = np.dot(np.ascontiguousarray(templates[temp_idx,:,:]),np.ascontiguousarray(unwhitening_mat))
                    
    return spike_times, spike_clusters, unwhitened_temps, channel_map, cluster_ids, cluster_info


