# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:44:06 2020

@author: Rajat
"""
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.special import factorial
 
# compute firing rate
def computeFiringRates(st, tstart, tend, tau):
    nCells = len(st)
    nTimeBins = int((tend-tstart)//tau)
    win = np.linspace(tstart, tend, nTimeBins)
    firingRates = np.zeros((nCells, nTimeBins-1))
    for i in range(nCells):
        firingRates[i,:] = np.histogram(st[i], win)[0]/tau
    return firingRates

# compute spike count
def computeSpikeCounts(st, tstart, tend, tau):
    nCells = len(st)
    nTimeBins = int(int(tend-tstart)/tau)
    win = np.linspace(tstart, tend, nTimeBins)
    spikeCounts = np.zeros((nTimeBins-1, nCells))
    for i in range(nCells):
        spikeCounts[:,i] = np.histogram(st[i], win)[0]/tau
    return spikeCounts

# poisson pdf
def poisspdf(x, lam):
    pdf = ((lam**x)*np.exp(-lam))/factorial(x)
    return pdf

# compute likelihood
def computeLikelihood(spkC, plFields, tau):
    pFields = (plFields * tau).T
    xyBins = plFields.shape[1]
    nTimeBins = spkC.shape[0]
    likelihood = np.zeros((xyBins, nTimeBins))
    for i in range(nTimeBins):
        nSpikes = np.array([spkC[i,:]]*xyBins)
        maxL = poisspdf(nSpikes,pFields)
        maxL = np.prod(maxL,1)
        likelihood[:,i] = maxL
    return likelihood

#load the data into the workspace
data = spio.loadmat('DataWilson.mat')
TimestampPosition = data['TimestampPosition'][0]
AnimalPosition = data['AnimalPosition'][0]
PlaceFields = data['PlaceFields']
SpikeTimesTrial1 = data['SpikeTimesTrial1'][0]
Trial1Start, Trial1End = data['Trial1Start'][0][0], data['Trial1End'][0][0]
SpikeTimesTrial2 = data['SpikeTimesTrial2'][0]
Trial2Start, Trial2End = data['Trial2Start'][0][0], data['Trial2End'][0][0]
SpikeTimesTrial3 = data['SpikeTimesTrial3'][0]
Trial3Start, Trial3End = data['Trial3Start'][0][0], data['Trial3End'][0][0]

# view the animal's trajectory
plt.figure()
plt.plot(TimestampPosition, AnimalPosition)
plt.xlabel('Time (sec)', fontsize=16)
plt.ylabel('Position (m)', fontsize=16)
plt.ylim([0,3.6])
plt.title('Position Trajectory', fontsize=16)
plt.show()

# View the place fields for all the cells
cell_fr_order = np.argmax(PlaceFields, axis=1)
cell_fr_order = np.argsort(cell_fr_order)
PlaceFields_sorted = PlaceFields[cell_fr_order]
PositionBins = np.round(data['PositionBins'][0],2)
plt.figure()
plt.imshow(PlaceFields_sorted, cmap='jet')
xtl = np.arange(0,PlaceFields_sorted.shape[1],20)
plt.xticks(xtl, PositionBins[xtl])
plt.xlabel('Position (m)', fontsize=16)
plt.ylabel('Cell Number', fontsize=16)
plt.title('1d rate maps', fontsize=16)
plt.show()

# display the raw neural spike trains recorded from four place cells
# during one traversal of the animal along the linear track
plt.figure()
for i in range(len(SpikeTimesTrial1)):
    t = np.ravel(SpikeTimesTrial1[i])
    plt.plot(t, i * np.ones_like(t), 'k.', markersize=5)
plt.xlabel('Time (sec)', fontsize=16)
plt.ylabel('Cell Number', fontsize=16)
plt.title('Trial#1 spikes', fontsize=16)
plt.show()

"""
 Decoding the animal's trajectory using a simple winner-take-all strategy
 At each moment, determine which place cell has the highest firing rate,
 use the preferred location for this place cell as the instantaneous 
 location of the animal.
"""

# determine the preferred location for each place cell from its place field:
maxIndex = np.argmax(PlaceFields, axis=1)
maxFiringRate = np.max(PlaceFields, axis=1)
# convert the indices stored in maxIndex to position on the track
maxPlace = PositionBins[maxIndex]
# compute firing rates for all place cells during Trial 1
tau = 0.25
firingRatesTrial1 = computeFiringRates(SpikeTimesTrial1, Trial1Start, Trial1End, tau)
cell_binfr_order = np.argmax(firingRatesTrial1, axis=1)
cell_binfr_order = np.argsort(cell_binfr_order)
firingRatesTrial1_sorted = firingRatesTrial1[cell_binfr_order]
plt.figure()
plt.imshow(firingRatesTrial1_sorted, cmap='jet')
xt = np.arange(0,firingRatesTrial1_sorted.shape[1],20)
xtl =  np.round(np.linspace(Trial1Start,Trial1End,len(xtl)),2)
plt.xticks(xt, xtl)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Cell Number', fontsize=16)
plt.title('Trial1 sorted binned spiking', fontsize=16)
plt.show()

# determine which place cell has the maximum firing rate at each moment, 
# and then determine the preferred location for this place cell
maxIndex = np.argmax(firingRatesTrial1, axis=0)
maxFiringRate = np.max(firingRatesTrial1, axis=0)
# convert the indices stored in maxIndex to position on the track
posTrial1 = maxPlace[maxIndex]
# display the computed trajectory of the animal for Trial 1
plt.figure()
timesTrial1 = np.linspace(Trial1Start, Trial1End, 75)
plt.scatter(timesTrial1, posTrial1, c='r', s=5)
plt.plot(TimestampPosition, AnimalPosition, c='b', linewidth=2)
plt.xlim([Trial1Start, Trial1End])
plt.xlabel('Time (sec)', fontsize=16)
plt.ylabel('Position (m)', fontsize=16)
plt.show()


"""
decoding the animal's trajectory using a maximum likelihood approach
which computes a location at each moment that maximizes the likelihood of 
occurrence, given the spiking activity of the place cells. Steps:

1. discretize the total time of interest into (non-overlapping) bins of duration ,
2. count the number of spikes generated by each place cell in each time bin 
 these are the values  that will be used for each time bin,
3. compute the likelihood across all position bins using the spike count of 
 all the place cells for each time bin,
4. for each time bin, find the maximum-likelihood estimate of the animal's 
 position - this is the decoded position.
"""

# Step1,2,3
tau = 0.25
# compute spike count in each time bin
spikeCounts = computeSpikeCounts(SpikeTimesTrial1, Trial1Start, Trial1End, tau)
# compute likelihood
likelihood = computeLikelihood(spikeCounts, PlaceFields, tau)
# display the likelihoods for all position bins
timeBins = np.arange(45,50)
plt.figure()
for i in range(5):
    plt.subplot(5,1,i+1)
    plt.plot(PositionBins, likelihood[:, timeBins[i]])
    plt.xlabel('Position (m)')
    plt.xlim([0,3.6])
plt.title('Likelihood', fontsize=16)
plt.show()

# decode the animal's position using the maximum likelihood estimate
# different values of tau and three different trials
tau = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
trialSpikeTimes = [SpikeTimesTrial1, SpikeTimesTrial2, SpikeTimesTrial3]
trialTimestamps = [[Trial1Start, Trial1End], [Trial2Start, Trial2End], [Trial3Start, Trial3End]]
nrows=len(trialSpikeTimes)
ncols=len(tau)
count = 1
plt.figure()
for j in range(len(trialSpikeTimes)):
    for i in range(len(tau)):
        # compute spike count in each time bin
        spikeCounts = computeSpikeCounts(trialSpikeTimes[j], trialTimestamps[j][0], trialTimestamps[j][1], tau[i])
        # compute likelihood
        likelihood = computeLikelihood(spikeCounts, PlaceFields, tau[i])
        index = np.argmax(likelihood, 0)
        # get decoded position
        maxL = PositionBins[index]
        nTimeBins = int((trialTimestamps[j][1]-trialTimestamps[j][0])/tau[i])
        windows = np.linspace(trialTimestamps[j][0], trialTimestamps[j][1], nTimeBins-1)
        plt.subplot(nrows, ncols, count)
        # decode data
        plt.plot(windows, maxL, 'r.', markersize=5)
        # actual trajectory
        plt.plot(TimestampPosition, AnimalPosition, 'b', linewidth=2)
        plt.xlim([trialTimestamps[j][0], trialTimestamps[j][1]])
        # some plotting hack
        if count>=2*ncols+1:
            plt.xlabel('Position (m)', fontsize=15)
        else:
            plt.xticks([])
        if count%ncols==1:
            plt.ylabel('Time (sec)', fontsize=15)
        else:
            plt.yticks([])
        plt.title('Trial' + str(j) + ' tau='+ str(tau[i]), fontsize=12)
        count+=1
plt.show()