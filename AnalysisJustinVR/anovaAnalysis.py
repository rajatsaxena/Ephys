# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:03:53 2023

@author: Rajat
"""
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np

def ols2(X, y):
  # Compute theta_hat using OLS
  theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
  return theta_hat

# load hallway data for all cells across trials
dath1 = np.load('VR19-hall1.npy', allow_pickle=True)
dath2 = np.load('VR19-hall2.npy', allow_pickle=True)
dath3 = np.load('VR19-hall3.npy', allow_pickle=True)
dath4 = np.load('VR19-hall4.npy', allow_pickle=True)

# select a subset of trials and concatenate across trials
dath1 = dath1[:,:60,:]
dath2 = dath2[:,:60,:]
dath3 = dath3[:,:60,:]
dath4 = dath4[:,:60,:]
dat = np.concatenate((dath1, dath2, dath3, dath4),1)
del dath1, dath2, dath3, dath4

# create analysis variables: position, hallnum, object
position = np.ravel([np.arange(100)]*dat.shape[1])
hallnum = np.ravel(np.repeat([1,2,3,4], dat.shape[-1]*dat.shape[1]//4))
objectpos = np.ravel(np.zeros([100,1]))
objecth1 = np.copy(objectpos)
objecth1[40:50] = 2
objecth1[90:] = 1
objecth1 = np.tile(objecth1,dat.shape[1]//4)
objecth2 = np.copy(objectpos)
objecth2[40:50] = 1
objecth2[90:] = 3
objecth2 = np.tile(objecth2,dat.shape[1]//4)
objecth3 = np.copy(objectpos)
objecth3[40:50] = 2
objecth3[90:] = 1
objecth3 = np.tile(objecth3,dat.shape[1]//4)
objecth4 = np.copy(objectpos)
objecth4[40:50] = 1
objecth4[90:] = 3
objecth4 = np.tile(objecth4,dat.shape[1]//4)
objpos = np.concatenate([objecth1, objecth2, objecth3, objecth4])
del objecth1, objecth2, objecth3, objecth4, objectpos

# perform two-way ANOVA with interaction
formula = 'y ~ C(pos) + C(hall) + C(obpos) + C(pos):C(hall) + C(pos):C(obpos) + C(hall):C(obpos)'

# create pandas data analysis
sig_neuron = []
coef_neuron = []
for n in range(dat.shape[0]):
    y = np.ravel(dat[n])
    df = pd.DataFrame({'y':y, 'pos':position, 'hall':hallnum, 'obpos':objpos})
    
    # run anova for each neuron
    model = ols(formula, df).fit()
    aov_table = anova_lm(model, typ=3)
    p_val = np.array(aov_table['PR(>F)'])[:-1]
    sig_neuron.append(p_val)
    
    # run multiple linear regression
    X = df.iloc[:,1:]
    poly = PolynomialFeatures(3, interaction_only=True)
    X = poly.fit_transform(X)
    coef_neuron.append(ols2(X, df['y'])[1:])

    del df, model, X
    print(n)
    print('***********')
sig_neuron = pd.DataFrame(sig_neuron, columns=['pos','hall','obpos','p*h','p*o','h*o','p*h*o'])
coef_neuron = pd.DataFrame(coef_neuron, columns=['pos','hall','obpos','p*h','p*o','h*o', 'p*h*o'])