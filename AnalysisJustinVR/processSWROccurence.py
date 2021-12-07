#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:03:29 2021

@author: rajat
"""
import os
import numpy as np
import pandas as pd 
from utilsEphys import *
import matplotlib.pyplot as plt

# ******************* load analysis params file ***********************************
dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Justin_Data/VR19'

# load processed metrics info
dfBehav = pd.read_csv(os.path.join(dirname, 'Behavior', 'dfCombined.csv'))
dfBehav.drop(columns=dfBehav.columns[0], inplace=True)


# find blank period start and end time
blankIdx = np.where(dfBehav['x-value']==9)[0]
blankTime = np.array(dfBehav[dfBehav['x-value']==9]['intantime'])
changeidx = np.ediff1d(blankTime)
blankEndTime = blankTime[np.where(changeidx>=1)[0]]
blankEndTime = np.append(blankEndTime,blankTime[-1])
blankStartTime = blankTime[np.where(changeidx>=1)[0]+1]
blankStartTime = np.insert(blankStartTime,0,blankTime[0])
blankHallway = []


# load ripple data and throw away all events within 500ms
dfRipple = pd.read_csv('./opRipples/ripplesShank2.csv', index_col=0)
et = np.array(dfRipple['end_time'][1:])
st = np.array(dfRipple['start_time'][:-1])
idx = np.where((et - st)<0.5)[0]
dfRipple.drop(idx, inplace=True)
peak_time = np.array(dfRipple['peak_time'])
# del dfRipple


ripple_state = []
ripple_pos = []
for pt in peak_time:
    if pt<dfBehav.iloc[-1]['intantime']:
        idx, _ = find_le(dfBehav['intantime'], pt)
        if np.isnan(idx):
            ripple_state.append(0)
            ripple_pos.append(-30)
        else:
            df = dfBehav.iloc[idx]
            if df['x-value']!=9:
                ripple_state.append(int(df['x-value']))
                ripple_pos.append(df['position'])
            else:
                df = dfBehav.iloc[idx-1]
                intantime = df['intantime'] - 0.75
                idx = dfBehav['intantime'].sub(intantime).abs().idxmin()
                xval = dfBehav.iloc[idx]['x-value']
                pos = dfBehav.iloc[idx]['position']
                ripple_state.append(int(xval))
                ripple_pos.append(pos)
    else:
        ripple_state.append(0)
        ripple_pos.append(-30)
ripple_state = np.array(ripple_state)
ripple_pos = np.array(ripple_pos)


state_pie, edge1 = np.histogram(ripple_state)
pos_pie, edge2 = np.histogram(ripple_pos, np.arange(-35,105,5))