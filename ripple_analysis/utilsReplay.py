# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:33:20 2020

@author: Rajat
"""
import sys
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage.transform import radon

"""
matlab style gaussian function
1/1+exp^(-a*(x-c))
"""
def sigmoid(x,c,a):
    return 1. / (1 + np.exp(-a*(x-c)))

"""
Inputs:
Cr = [nTemporalBin X nCell] matrix of binned firing rates
rateMap = [nSpatialBin X nCell] firing rate 'template'
binLength = scalar of the duration of the bins in 'Cr'

Outputs:
Pr = [nTemporalBin X nSpatialBins] matrix of posterior probabilities
prMax = the spatial bin with higher spatial probabilities for each
  temporalBin in Cr
"""
def placeBayes(Cr, rateMap, binLength):
    Cr = Cr*binLength
    rateMap = rateMap.T
    term2 = np.exp((-1)*binLength*np.sum(rateMap.T,0))
    mp = 1./rateMap.shape[1]
    Pr_ = []
    
    c = np.repeat(Cr[:, :, np.newaxis], rateMap.shape[0], axis=2)
    b = np.repeat(rateMap.T[:, :, np.newaxis], c.shape[0], axis=2)
    b = np.moveaxis(b, -1, 0)
    
    u = mp*np.prod(b**c, 1)
    Pr_ = u*np.repeat(term2[:,np.newaxis], u.shape[0], axis=1).T
    Pr_ = Pr_/np.repeat(np.sum(Pr_,1)[:,np.newaxis], Pr_.shape[1], axis=1)
    
    m = np.argmax(Pr_,1)
    prMax_ = m.T
    
    if np.sum(np.isinf(Pr_))>0:
        sys.error('Do Not Approach the Infinite')
    return Pr_, prMax_

def fPolyFit(x,y,n):
    V = np.ones((len(x),n+1))
    for j in np.arange(1,0,-1):
        V[:,j-1] = V[:,j]*x
    # Solve least squares problem
    Q, R = sp.linalg.qr(V, mode='economic')
    y = np.reshape(y, (len(y),1))
    p =  np.transpose(np.linalg.inv(R).dot(Q.T.dot(y)))[0]
    return p

"""
computes max projection line, using radon transform, on Pr matrix
"""
def Pr2Radon(Pr, plotting=0):
    try:
        Pr[np.isnan(Pr)] = 0
        theta = np.arange(0,180+0.5,0.5)
        R = radon(Pr,theta,circle=False)
        bw_ = R.shape[0]//2
        xp = np.linspace(-bw_,bw_,R.shape[0])
        
        y=[None, None]
        x=[None, None]
        y[0] = np.ceil((Pr.shape[0])//2.)+1
        x[0] = np.ceil((Pr.shape[1])//2.)+1
        
        I = np.nanargmax(R,0)
        Y = R[I]
        locs = np.arange(len(Y))
        slope = np.zeros(len(locs))
        integral = np.zeros(len(locs))
        curve = np.zeros(Pr.shape[1])
        for pk in range(len(locs)):
            angle = theta[locs[pk]]
            offset = xp[I[locs[pk]]]
            if offset==0:
                offset=0.01
    
            y[1] = y[0] + offset*np.sin(np.deg2rad(-angle))
            x[1] = x[0] + offset*np.cos(np.deg2rad(-angle))
            coeffs = fPolyFit(x, y, 1)
            xx = np.arange(1,Pr.shape[1]+1)
            yy = (-1/coeffs[0])*(xx - x[0]) + y[0] - offset
            coeffs = fPolyFit(xx, yy, 1)
            slope[pk] = coeffs[0]
            # rise/run limit to calc integral (must be in the frame)
            if abs(slope[pk]) < 2*Pr.shape[0]/Pr.shape[1] and abs(slope[pk]) > 1.5:
                for i in range(len(xx)):
                    if yy[i] > .5 and yy[i] < Pr.shape[0] - .5:                    
                        curve[i] = Pr[int(yy[i]),int(xx[i])]
                    else:
                        curve[i] = np.nan
                integral[pk] = np.nanmean(curve)
            else:
                integral[pk] = np.nan
                slope[pk] = np.nan
        
        # weird typecasting fix
        integral = np.array(integral, dtype='float64')
        idx = np.nanargmax(integral)
        integral = integral[idx]
        slope = slope[idx]
    except:
        return np.nan, np.nan
    
    return slope, integral


"""
Sorts two matrices by ordering the maximums of the first and using that
order to rearrange both
#1 = max, #2 = min
"""
def sort_cells(item,item2=None,num=1):
    if item2 is None:
        item2 = np.copy(item)  
    new_item = np.zeros((item.shape[0], item.shape[1]))
    new_item2 = np.zeros((item2.shape[0], item2.shape[1]))    
    if num==1:
        d = np.argmax(item, axis=1)
        ddd = np.argsort(d)
        new_item = item[ddd,:]
        new_item2 = item2[ddd,:] 
        order = ddd        
    if num==2:
        d = np.argmin(item, axis=1)
        ddd = np.argsort(d)
        new_item = item[ddd,:]
        new_item2 = item2[ddd,:] 
        order = ddd        
    return new_item, new_item2, order

"""
sorts the rows of a matrix in ascending/descending order based on the elements 
in the first column. When the first column contains repeated elements, 
sortrows sorts according to the values in the next column and repeats this 
behavior for succeeding equal values.
"""
def sort_rows(mat, order='descending'):
    df = pd.DataFrame(mat)
    if order=='descending':
        df.sort_values(list(np.arange(len(df.columns))), ascending=False,inplace=True)
    else:
        df.sort_values(list(np.arange(len(df.columns))), ascending=True,inplace=True)
    return np.array(df), np.array(df.index)

"""
circularly shift each row
"""
def shuffleCircular(data):
    data1 = np.copy(data)
    for i in range(data1.shape[0]):
        shift_ = np.random.randint(0, data1.shape[1], 1)
        shift_ = shift_[0]
        data1[i,:] = np.roll(data1[i,:], shift_)
    return data1

"""
shuffle individual cell
"""
def shuffleCellID(data):
    data1 = np.copy(data)
    data1 = data1[np.random.permutation(data1.shape[0]),:]
    return data1

"""
function that 'pulls out' a start and stop time around each
ripple/population event.
"""
def processReplayData(Q, Qt, ripts, binsize=0.01, thresh=2):
    ripst = ripts[0]
    ripet = ripts[-1]
    start = int(np.where(Qt<ripst)[0][-1] - int(0.05//binsize)) 
    stop = int(np.where(Qt>ripet)[0][0] + int(0.05//binsize))
    # data and counts    
    data = Q[:,start:stop]
    counts = Q[:,start:stop]
    # cut 0 and 1 spk count bins from the start/end
    while counts.size!=0 and counts.shape[0]>1 and np.sum(counts[:,0])<thresh:
        data = data[:,1:]
        counts = counts[:,1:]
        start = start + 1 
    while counts.size!=0 and counts.shape[0]>1 and np.sum(counts[:,-1])<thresh:
        data = data[:,:-1]
        counts = counts[:,:-1]
        stop = stop-1
    if data.size==0 or counts.size==0:
        return None, None
    # convert the Q to firing rate
    data = data/binsize
    return data, counts

"""
mean2d
"""
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

"""
matlab 2d correlation
"""
def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r

"""
spearman correlation
"""
def corr(a,b):
    r, p = sp.stats.spearmanr(a,b)
    return round(r,3), round(p,3)

"""
pearson correlation
"""
def corrPearson(a,b):
    r, p = sp.stats.pearsonr(a,b)
    return round(r,3), round(p,3)

"""Weighted Mean"""
def m(x, w):
    if x.shape[0]==w.shape[0]:
        w = w.T
    return np.sum(x * w) / np.sum(w)

"""Weighted Covariance"""
def cov(x, y, w):
    xd = (x - m(x, w))
    yd = (y - m(y, w))
    yd = np.resize(yd,xd.shape)
    if w.shape[0]==yd.shape[0]:
        w=w.T
    return np.sum(w * xd * yd) / np.sum(w)

"""Weighted Correlation"""
def corrWeighted(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

"""
get bayes weighted correlation (Grossmark and Buzsaki 2016)
"""
def makeBayesWeightedCorr1(Pr, bID):
    bID = np.ones(Pr.shape[0])
    outID = np.unique(bID)
    
    h = np.array([Pr.shape[0]]*Pr.shape[0])
    Q = makeQForWeightedCorr(h, Pr.shape[1])
    
    outR = np.zeros(len(outID))
    for i in range(len(outR)):
        w = np.reshape(Pr[bID==outID[i],:],-1)
        xy = Q[h[i]-1]+1
        outR[i] = makeWeightedCorr1(xy,w)
    return outR[-1], outID[-1]
    

def makeQForWeightedCorr(UniqueNumBins, numPlaces):
    a1 = np.tile(np.array([np.arange(np.max(UniqueNumBins))]).T, numPlaces)
    b1 = np.tile(np.array([np.arange(numPlaces)]).T, np.max(UniqueNumBins))
    Q = np.zeros((len(UniqueNumBins), a1.shape[0]*a1.shape[1], 2))
    for i in range(len(UniqueNumBins)): 
        x = np.array([np.reshape(a1.T,-1)])
        y = np.array([np.reshape(b1,-1)])
        Q[UniqueNumBins[i]-1,:,:] = np.concatenate((x,y)).T
    Q=np.array(Q)
    return Q

def makeWeightedCorr1(xy, w):
    mxy = np.sum(xy*(np.tile(np.array([w]).T,2)/np.sum(w)),0)
    covxy = np.sum(w*(xy[:,0] - mxy[0])*(xy[:,1] - mxy[1]))/np.sum(w)
    covxx = np.sum(w*(xy[:,0] - mxy[0])*(xy[:,0] - mxy[0]))/np.sum(w)
    covyy = np.sum(w*(xy[:,1] - mxy[1])*(xy[:,1] - mxy[1]))/np.sum(w)
    out = covxy/np.sqrt(covyy*covxx)    
    return out
