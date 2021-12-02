#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:06:05 2019

@author: rajat
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings
import scipy.io as spio
from scipy import stats
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Another binary search function to find the nearest value to a list of timestamps given pivot value
#INPUT: list of timestamps, pivot timestamp to search for 
#OUTPUT: closest matching timestamps index and value
def binarySearch(data,val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind] 
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind, data[best_ind]

# function to sort list1 given list2
def sort_list_another(list1, list2, list3):
    # find the indices order for sorting list2
    idx   = np.argsort(list2)
    # use the indices from above (list 2) to sort list1
    sorted_list1 = list(np.array(list1)[idx])
    sorted_list2 = list(np.array(list2)[idx])
    sorted_list3 = list(np.array(list3)[idx])
    # return sorted list1 and list2
    return sorted_list1, sorted_list2, sorted_list3

# find time in single trials
def time_in_single_trial(value):
    if len(value)>1:
        return value.iloc[-1] - value.iloc[0]
    else:
        return 0.001

# get binned speed trial data
def get_binned_speed_trial(trial_movement):
    # arrange bins from with 0.5 difference from -100 to 100
    bins = np.arange(0,100,0.5)
    # matrix to hold the binned speed trials
    binned_speed_trial = np.zeros((len(trial_movement), len(bins)))
    # matrix to hold the binned time trials count
    binned_time_count_trial = np.zeros((len(trial_movement), len(bins))) 
    # matrix to hold the binned time trials sum
    binned_time_sum_trial = np.zeros((len(trial_movement), len(bins)))
    # iterate over each trial data
    for trial_num in trial_movement:
        # load individual trial data
        movement_data = trial_movement[trial_num]
        # load the speed, position corr. to individual trial
        speed = movement_data['speed']
        pos = movement_data['pos']
        time = movement_data['time']
        # sort speed data according to position (may not be necessary)
        speed_sorted_pos, pos_sorted, time_sorted = sort_list_another(speed, pos, time)
        # find the indices to which element in speed and pos belong in the bin array
        # hack to remove nan from pos_sorted
        speed_sorted_pos_ = []
        pos_sorted_ = []
        time_sorted_ = []
        for x,v,t in zip(pos_sorted, speed_sorted_pos, time_sorted):
            if not np.isnan(x) and x<=100:
                pos_sorted_.append(x)
                speed_sorted_pos_.append(v)
                time_sorted_.append(t)
        inds = np.digitize(pos_sorted_, bins)
        # find unique bin array
        inds_unique = np.unique(inds)-1
        # create dataframe with indices, sorted speed, sorted pos
        df_ = {'inds':inds,'speed':speed_sorted_pos_, 'pos':pos_sorted_, 'time':time_sorted_}
        df_ = pd.DataFrame(df_)
        
        # group the dataframe according to inds
        grouped_df_ = df_.groupby('inds')
        # calculate the nanmean of speed column of the grouped data
        mean_group_speed = grouped_df_['speed'].agg(np.nanmean)
        # calculate the length of xposition of the grouped data
        len_group_time = grouped_df_['pos'].agg('count')
        # calculate the nansum of time of the grouped data
        time_spent = grouped_df_['time']
        
        # create empty binned speed array for each trial
        binned_speed_ = np.zeros(len(bins))
        # assign mean grouped speed to their respective indices
        binned_speed_[inds_unique] = mean_group_speed
        # add this binned for each trial to the matrix for speed across trials
        binned_speed_trial[trial_num,:] = binned_speed_
        
        # create empty binned time count array for each trial
        binned_time_count_ = np.zeros(len(bins))
        # assign time count to their respective indices
        binned_time_count_[inds_unique] = len_group_time
        # add this binned for each trial to the matrix for time across trials
        binned_time_count_trial[trial_num,:] = binned_time_count_
        
        # create empty binned time sum array for each trial
        binned_time_sum_ = np.zeros(len(bins))
        time_spent = time_spent.apply(time_in_single_trial)
        # assign count of time to their respective indices
        binned_time_sum_[inds_unique] = time_spent
        # add this binned for each trial to the matrix for speed across trials
        binned_time_sum_trial[trial_num,:] = binned_time_sum_
        
    return binned_speed_trial, binned_time_count_trial, binned_time_sum_trial, bins

# function to plots total number of licks in each trial
def plot_num_licks_trial(trial_lick_, hname, fname):
    figtitle = 'Hall' + str(hname) + '_' + fname
    # number of licks in each trial
    licks_num_trial = []
    # iterate over each trial to get number of licks in each trial
    for trial_num in range(len(trial_lick_)):
        num_licks = len(trial_lick_[trial_num]['lick_ts'])
        licks_num_trial.append(num_licks)
    # plot the number of licks/ trial
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    fig10 = plt.figure(figsize=(12,6),dpi=100)
    ax2=plt.subplot(211)
    ax2.plot(np.arange(len(trial_lick_)), licks_num_trial)
    ax2.set_ylabel('# licks/ trial', fontsize=18)
    ax2.set_xlabel('Trial#', fontsize=18)
    ax2.set_xlim([0, len(trial_lick_)])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig10.tight_layout()
    ax3=plt.subplot(212)
    lick_position = []
    for trial_number in range(len(trial_lick_)):
        lick_position.append(trial_lick_[trial_number]['lick_pos'])
    lick_position = np.array(lick_position)
    plt.eventplot(lick_position)
    plt.axvline(x=70,ymin=0,ymax=len(trial_lick_),c='r',alpha=0.75,linestyle='--')
    ax3.set_xlim([0,100])
    ax3.set_xlabel('Position', fontsize=18)
    ax3.set_ylabel('Trial#', fontsize=18)
    ax3.set_title('Position of licks', fontsize=18)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    fig10.tight_layout()
    plt.suptitle(figtitle, fontsize=18)
    # return the number of licks per trial and the figure
    return licks_num_trial, lick_position, fig10

# environments lookup tables
env_lookup_table = {'1': 'Snowy_drum', '2':'Snowy_star','3':'Grass_drum', '4':'Grass_star', '5':'Desert_drum', '6':'Desert_star', 
                    '9':'Blank', '80':'dark_star_NR','100':'dark_star_R', '120':'ring_drum_R','140':'ring_drum_NR','160':'dark_drum_R','180':'dark_drum_NR'}
dirname = 'VR19Behavior'

# load the intan aligned timestamp
intanTransitionTime = np.load('intan_aligned_timestamps.npy', mmap_mode='r')

# iterate over all the files
for filename in os.listdir(os.getcwd()):
    if filename.endswith('.csv') and 'VR' in filename and 'proc' not in filename:
        print('Processing : ' + filename)
        # read the pandas files
        maindf = pd.read_csv(filename, skiprows=5, names=['time','loc','state','x-value','position','location'])
        # setup the columns
        maindf.columns = ['time','loc','state','x-value','position','location']
        # find the indices where the strings hallway pushe and start pushed appear
        hallway_prob_idx = np.where(maindf['loc'].str.contains('hallway_prob'))[0] 
        start_pushed_idx = np.where(maindf['loc'].str.contains('start pushed'))[0]
        start_pushed_idx = start_pushed_idx[1:]
        
        # get dataframes for different contexts being used and store it in a list
        df_ctxt = []
        for i in range(len(hallway_prob_idx)):
            if i==len(hallway_prob_idx)-1:
                df_ = maindf.iloc[hallway_prob_idx[i]+1:]
            else:
                df_ = maindf.iloc[hallway_prob_idx[i]+1:start_pushed_idx[i]-1]
            df_ctxt.append(df_)
        del df_
        
        df_ctxt = df_ctxt[0]
        df_ctxt = df_ctxt.reset_index(drop=True)
        stop_idx = np.where(df_ctxt['state'].str.contains('stop'))[0]
        df_ctxt = df_ctxt.iloc[:stop_idx[-1]]
        # find the location where the reward is delivered and replaced it with hall number
        reward_idx = np.where(df_ctxt['loc'].str.contains('reward'))[0]
        reward_time = np.array(df_ctxt['time'][reward_idx])
        df_ctxt = df_ctxt.drop(reward_idx)
        # reset index 
        df_ctxt = df_ctxt.reset_index(drop=True)
        # RESET TIME TO 0
        df_ctxt['time'] = df_ctxt['time'] - df_ctxt['time'][0]
        df_ctxt['intantime'] = intanTransitionTime
        df_ctxt = [df_ctxt]
        # iterate over data from each context
        for df in df_ctxt:
            # find speed for each position datapoint
            speed = np.abs(np.ediff1d(df.loc[:,'position']))/np.ediff1d(df.loc[:,'time'])
            # add the speed columns
            speed[np.where(speed>100)]=None
            speed = np.insert(speed, 0, None)
            df.loc[:,'speed'] =  speed
            
            # remove blank periods
            state = df['x-value']
            stateIdx = np.where(state!=9)[0]
            df = df.loc[stateIdx,:]
            df.reset_index(drop=True)
    
            # get unique hallway number
            hallway_num = df['x-value']
            hallway_num_unique = np.unique(hallway_num)
            hallway_num_unique = hallway_num_unique[~np.isnan(hallway_num_unique)]
            
            # reset index 
            df = df.reset_index(drop=True)
            # iterate over all the hallway in each contexts
            fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize=(10,7))
            hallway_trials = {}
            for hallnum in hallway_num_unique:
                print('Analyzing hallway: ' + str(hallnum))
                # sanity check to skip certain hallways 
                if len(np.where(df['x-value']==hallnum)[0])>5:
                    # hallway number
                    hallway_num_new = df['x-value']                    
                    # find the location where hallway number is hallnum
                    idx = np.where(hallway_num_new==hallnum)[0]
                    df_ = df.loc[idx,:]
                    # find the start and end trial movement index
                    trial_change_endidx = np.where(np.ediff1d(df_['location'])==-1)[0]+1
                    trial_change_startidx = np.where(np.ediff1d(df_['location'])==-1)[0]+1
                    trial_change_startidx = np.insert(trial_change_startidx,0,0)
                    trial_change_endidx = np.append(trial_change_endidx, idx[-1])
                    
                    # variables to hold the trial movement and trial lick dict
                    trial_movement = {}
                    trial_lick = {}
                    # start trial number is 0
                    trial_num = 0
                    # iterate over each trial
                    for st,et in zip(trial_change_startidx,trial_change_endidx):
                        # get speed, position, time 
                        pos = np.array(df_[st:et]['position'])
                        time = np.array(df_[st:et]['time'])
                        intantime = np.array(df_[st:et]['intantime'])
                        sp = np.ediff1d(pos)/np.ediff1d(time)
                        sp = np.insert(sp,0,np.nan)
                              
                        # add all the data to trial movement 
                        trial_movement[trial_num] = {'speed':sp, 'pos':pos, 'time':time, 'intantime':intantime}
                        
                        # lick data analysis
                        # find indices where lick occured and the data lies between start ts and end ts
                        lick_idx = list(np.where((df.loc[:,'loc'] == 'lick') & (df.loc[:,'time'] >= time[0]) & (df.loc[:,'time'] <= time[-1]))[0])
                        # find the timestamps and position when lick happened and return it
                        lick_ts = list(df.iloc[lick_idx]['time'])
                        dfnotna = df.fillna(method='ffill')
                        lick_pos = list(dfnotna.iloc[lick_idx]['position'])
                        # store the lick data for each trial as well as position and speed
                        trial_lick[trial_num] = {'lick_index':lick_idx, 'lick_ts':lick_ts,'lick_pos':lick_pos}
                        trial_num = trial_num + 1
                    print('Ran ' + str(trial_num) + ' trials in Hallway: ' + str(hallnum))
                    # get binned speed, binned time count, binned time sum plot
                    binned_speed_trial, binned_time_count_trial, binned_time_sum_trial, bins = get_binned_speed_trial(trial_movement)
                    
                    # get the number of licks per trial and plot it
                    # num_licks_trial, lick_position_trial, fig2 = plot_num_licks_trial(trial_lick, hallnum, filename)
                    
                    # mean and std speed across trials at different spatial position
                    binned_speed_trial[binned_speed_trial==0]=np.nan
                    mean_speed_trial = np.nanmedian(binned_speed_trial,axis=0)
                    std_speed_trial = np.nanstd(binned_speed_trial,axis=0)
                    iqr_speed_trial = stats.iqr(binned_speed_trial, axis=0, rng=(35,65), nan_policy='omit')
                    
                    # create the dictionary to late store the data
                    hallway_trials[hallnum] = {'trial_movement':trial_movement, 'binned_speed':binned_speed_trial, 'binned_time':binned_time_count_trial,
                                  'binned_time_sum':binned_time_sum_trial,'mean_speed':mean_speed_trial, 'trial_change_startidx':trial_change_startidx,
                                  'trial_change_endidx':trial_change_endidx, 'std_speed':std_speed_trial, 'iqr_speed':iqr_speed_trial}
                    
                    # plot the data to look at mean speed
                    y1, y2 = hallway_trials[hallnum]['mean_speed'][2:]+0.5*hallway_trials[hallnum]['std_speed'][2:], hallway_trials[hallnum]['mean_speed'][2:]-0.5*hallway_trials[hallnum]['std_speed'][2:]
                    xposdata = np.linspace(0,100,len(hallway_trials[hallnum]['mean_speed'])-2)
                    yspeeddata = hallway_trials[hallnum]['mean_speed'][2:]
                    yerr = hallway_trials[hallnum]['iqr_speed'][2:]
                    ax1.plot(xposdata, yspeeddata, label=env_lookup_table[str(int(hallnum))])
                    ax1.fill_between(xposdata, yspeeddata - yerr, yspeeddata + yerr, alpha=0.4)
                    
                    df = df.reset_index(drop=True)
                    # save the individual trial data and dataframes
                    np.save(os.path.join('hall'+str(int(hallnum))+'_occmap.npy'), trial_movement)
                    df_.to_csv(os.path.join('hall'+str(int(hallnum))+'_df.csv'))
                    
                    #save mat file
                    mfname = os.path.join(filename[:-4]+'_hall'+str(int(hallnum))+'_trialdata.mat')
                    spio.savemat(mfname, mdict={'binned_speed':binned_speed_trial, 'binned_time':binned_time_count_trial,
                                  'binned_time_sum':binned_time_sum_trial,'mean_speed':mean_speed_trial,'std_speed':std_speed_trial})
            # ax1.set_ylim([0,25])
            ax1.plot([50]*30,np.linspace(0,30,30),'k--')  #number in brackets = reward vertical line 
            ax1.plot([100]*30,np.linspace(0,30,30),'k--') 
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.set_xlabel('Linear Position', fontsize=18)
            ax1.set_ylabel('Speed (cm/s)', fontsize=18)
            plt.rc('xtick',labelsize=16)
            plt.rc('ytick',labelsize=16)
            fig.legend(frameon=False, fontsize=16)
            ax1.set_title(filename, fontsize=18)
            fig.tight_layout()
            plt.show()    
