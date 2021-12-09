#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:36:11 2021

@author: rajat
"""
import glob, os
import numpy as np
import pandas as pd
import scipy.io as spio
import seaborn as sns
import scipy.stats as spst
from natsort import natsorted
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import gaussian_filter1d
plt.style.use('tableau-colorblind10')
mpl.rcParams['axes.linewidth'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

def getRatemap(dirname, hnum):
    rmaps = []
    files = natsorted(glob.glob(os.path.join(dirname,'ClustId*_hall'+str(hnum)+'_processed.mat')))
    for j, filename in enumerate(files):
        data = spio.loadmat(filename)
        rmap = data['smoothratemaptrial']
        rmaps.append(rmap)
    rmaps = np.array(rmaps)
    rmaps = np.moveaxis(rmaps, [0,1,2], [-2,-3,-1])
    return rmaps

def add_orientation_legend(ax, trial_types, pal = ['orange', 'indianred', 'c', 'royalblue']):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for k in range(len(trial_types))]
    labels = ['Hall {}'.format(t) for t in trial_types]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])
    
pal = ['orange', 'indianred', 'c', 'royalblue']

# cluster id
dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Justin_Data/VR19'
figfname = './opPV/iHCPCA.mp4'

cluidHC = np.load(os.path.join(dirname,'HCspikesorted/spikeClusterID.npy'), allow_pickle=True)
cluidV1 = np.load(os.path.join(dirname,'AL2spikesorted/spikeClusterID.npy'), allow_pickle=True)
cluidV2 = np.load(os.path.join(dirname,'AL1spikesorted/spikeClusterID.npy'), allow_pickle=True)

# load processed metrics info
dfHC = pd.read_csv(os.path.join(dirname,'HCspikesorted/cluster_info.tsv'), delimiter='\t')
dfHC = dfHC[dfHC['cluster_id'].isin(cluidHC)]
dfHC['region'] = ['HC']*len(dfHC)
dfV1 = pd.read_csv(os.path.join(dirname,'AL2spikesorted/cluster_info.tsv'), delimiter='\t')
dfV1 = dfV1[dfV1['cluster_id'].isin(cluidV1)]
dfV1['region'] = ['V1']*len(dfV1)
dfV2 = pd.read_csv(os.path.join(dirname,'AL1spikesorted/cluster_info.tsv'), delimiter='\t')
dfV2 = dfV2[dfV2['cluster_id'].isin(cluidV2)]
dfV2['region'] = ['V2']*len(dfV2)
# concatenate for entire recording
dfV2['region'] = np.where((dfV2.region=='V2') & (dfV2.depth<=350),'iHC',dfV2.region)
dfV1['region'] = np.where((dfV1.region=='V1') & (dfV1.depth<=750),'iHC',dfV1.region)


# load rate maps
clufname = os.path.join(dirname, 'analyzed/RatemapsAL1')
rmaps_h1 = getRatemap(clufname, 1)
rmaps_h2 = getRatemap(clufname, 2)
rmaps_h3 = getRatemap(clufname, 3)
rmaps_h4 = getRatemap(clufname, 4)

# select cells that are only V2
idx = np.where(dfV2['region']=='iHC')[0]
rmaps_h1 = rmaps_h1[:,idx,:]
rmaps_h2 = rmaps_h2[:,idx,:]
rmaps_h3 = rmaps_h3[:,idx,:]
rmaps_h4 = rmaps_h4[:,idx,:]

clufname = os.path.join(dirname, 'analyzed/RatemapsAL2')
rmaps2_h1 = getRatemap(clufname, 1)
rmaps2_h2 = getRatemap(clufname, 2)
rmaps2_h3 = getRatemap(clufname, 3)
rmaps2_h4 = getRatemap(clufname, 4)

# select cells that are only V2
idx = np.where(dfV1['region']=='iHC')[0]
rmaps2_h1 = rmaps2_h1[:,idx,:]
rmaps2_h2 = rmaps2_h2[:,idx,:]
rmaps2_h3 = rmaps2_h3[:,idx,:]
rmaps2_h4 = rmaps2_h4[:,idx,:]

rmaps_h1 = np.concatenate((rmaps_h1, rmaps2_h1),1)
rmaps_h2 = np.concatenate((rmaps_h2, rmaps2_h2),1)
rmaps_h3 = np.concatenate((rmaps_h3, rmaps2_h3),1)
rmaps_h4 = np.concatenate((rmaps_h4, rmaps2_h4),1)

trialtype_h1 = np.array([1]*rmaps_h1.shape[0])
trialtype_h2 = np.array([2]*rmaps_h2.shape[0])
trialtype_h3 = np.array([3]*rmaps_h3.shape[0])
trialtype_h4 = np.array([4]*rmaps_h4.shape[0])

# trials is a list of K Numpy arrays of shape N×T (number of neurons by number of time points).
trials = np.concatenate((rmaps_h1, rmaps_h2, rmaps_h3, rmaps_h4))
# trial_type is a list of length K containing the type (i.e. the orientation) of every trial.
trial_type = np.concatenate((trialtype_h1, trialtype_h2, trialtype_h3, trialtype_h4))
# trial_types is a list containing the unique trial types (i.e. orientations) in ascending order.
trial_types = [1, 2, 3, 4]

del rmaps_h1, rmaps_h2, rmaps_h3, rmaps_h4
del trialtype_h1, trialtype_h2, trialtype_h3, trialtype_h4

t_type_ind = [np.argwhere(np.array(trial_type) == t_type)[:, 0] for t_type in trial_types]
trial_size   = trials[0].shape[1]
Nneurons     = trials[0].shape[0]



############################# Trial-averaged PCA #################################################
trial_averages = []
for ind in t_type_ind:
    trial_averages.append(np.array(trials)[ind].mean(axis=0))
Xa = np.hstack(trial_averages)
Xa = spst.zscore(Xa, nan_policy='omit') 
Xa[np.isnan(Xa)] = 0

pca = PCA(n_components=20)
Xa_p = pca.fit_transform(Xa.T).T

fig, axes = plt.subplots(1, 6, figsize=(15,4), sharey='row')
for comp in range(6):
    ax = axes[comp]
    for kk, type in enumerate(trial_types):
        x = Xa_p[comp, kk * trial_size :(kk+1) * trial_size]
        x = gaussian_filter1d(x, sigma=1)
        ax.plot(x, c=pal[kk])
    ax.set_ylabel('PC {}'.format(comp+1), fontsize=22)
# add_orientation_legend(axes[0], trial_types)
axes[2].set_xlabel('Position ', fontsize=22)
plt.tight_layout()
plt.show()



############################################################################
# Find the indices of the three largest elements of the first threee eigenvector
for e in range(3):
    units = np.abs(pca.components_[e, :].argsort())[::-1][0:3]
    f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=False,sharex=True)
    for ax, unit in zip(axes, units):
        ax.set_title('Neuron {}'.format(cluidHC[unit]))
        for t, ind in enumerate(t_type_ind):
            x = np.nanmean(np.array(trials)[ind][:, unit, :],0)
            ax.plot(np.arange(x.shape[0]), x, color=pal[t])
    axes[1].set_xlabel('Position')
    sns.despine(fig=f, right=True, top=True)
    add_orientation_legend(axes[2], trial_types)
plt.show()



################### average-concatenated PCAs#################################
ss = StandardScaler(with_mean=True, with_std=True)
Xav_sc = ss.fit_transform(Xa.T).T
Xav_sc[np.isnan(Xav_sc)] = 0.
pca = PCA(n_components=20) 
pca.fit(Xav_sc.T) # only call the fit method

projected_trials = []
for trial in trials:
    # scale every trial using the same scaling applied to the averages 
    trial = ss.transform(trial.T).T
    trial[np.isnan(trial)]=0
    # project every trial using the pca fit on averages
    proj_trial = pca.transform(trial.T).T
    projected_trials.append(proj_trial)

n_components=20
gt = {comp: {t_type: [] for t_type in trial_types} for comp in range(n_components)}
for comp in range(n_components):
    for i, t_type in enumerate(trial_type):
        t = projected_trials[i][comp, :]
        gt[comp][t_type].append(t)

f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=True, sharex=True)
for comp in range(3):
    ax = axes[comp]
    for t, t_type in enumerate(trial_types):
        ax.plot(np.nanmean(gt[comp][t_type],0), color=pal[t])
    #ax.axvline(x=37, alpha=0.8, color='gray', ls='--')
    ax.set_ylabel('PC {}'.format(comp+1))
axes[1].set_xlabel('Position')
sns.despine(right=True, top=True)
add_orientation_legend(axes[2], trial_types)
plt.tight_layout()
plt.show()


################## 3D PCA trajectories ###########################################
# utility function to clean up and label the axes
def style_3d_ax(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xlabel('PC 1', fontsize=22)
    ax.set_ylabel('PC 2', fontsize=22)
    ax.set_zlabel('PC 3', fontsize=22)

# create the figure
fig = plt.figure(figsize=[15, 15])
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

# red dot to indicate when stimulus is being presented
stimdot1, = ax1.plot([np.nanmax(Xa_p[0])-0.5], [np.nanmax(Xa_p[2])-1.5], 'o', c='g', markersize=20, alpha=0.5)
# annotate with stimulus and time information
text1     = ax1.text(np.nanmax(Xa_p[0])-0.5, 0, np.nanmax(Xa_p[2])-0.5, 'Reward OFF \npos = {:}'.format(0), fontdict={'fontsize':25})
time = np.arange(trials.shape[2])

def animate(i):
    ax.clear() # clear up trajectories from previous iteration
    style_3d_ax(ax1)
    # ax1.view_init(elev=22, azim=30)
    for t, t_type in enumerate(trial_types):    
        x = Xa_p[0, t * trial_size :(t+1) * trial_size]
        y = Xa_p[1, t * trial_size :(t+1) * trial_size]
        z = Xa_p[2, t * trial_size :(t+1) * trial_size]
        ax1.plot(x[0:i], y[0:i], z[0:i], color=pal[t])
    ax1.set_xlim([np.nanmin(Xa_p[0])-0.5, np.nanmax(Xa_p[0])+0.5])
    ax1.set_ylim([np.nanmin(Xa_p[1])-0.5, np.nanmax(Xa_p[1])+0.5])
    ax1.set_zlim([np.nanmin(Xa_p[2])-0.5, np.nanmax(Xa_p[2])+0.5])

    if time[i]>48 and time[i]<52:
        stimdot1.set_color('r')
        text1.set_text('Reward 1 ON \npos = {:}'.format(int(time[i]*5.3)))
    elif time[i]>96 and time[i]<100:
        stimdot1.set_color('r')
        text1.set_text('Reward 2 ON \npos = {:}'.format(int(time[i]*5.3)))
    else:
        stimdot1.set_color('g')
        text1.set_text('Reward OFF \npos = {:}'.format(int(time[i]*5.3)))          
    return []

anim = animation.FuncAnimation(fig, animate,frames=trial_size, interval=2, blit=True)
anim.save(figfname, fps=5, extra_args=['-vcodec', 'libx264'])
plt.show()