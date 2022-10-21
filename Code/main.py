# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:43:38 2022

@author: lucaf
"""

import dataset_analysis as analys

#load the training set and slit Data (DTR) and labels (LTR)
DTR,LTR = analys.load('../Data/Train.txt')

#compute meand and std
mu=analys.compute_mean(DTR)
sigma=analys.compute_variance(DTR)

#compute Z-normalization on training data
scaled_DTR=analys.scale_ZNormalization(DTR, mu, sigma)
#TBR: to test the Z-normalization has been perfomed correctly compute again mean and variance of scaled_DTR anc check mean= [0,0...0]; sigma 0 [1,1.....1]

gaussianized_DTR = analys.gaussianize(DTR)

#save histograms of the distribution of all the features in '../Images' folder. E
analys.plot_histograms(scaled_DTR, LTR)

#analys.plot_histograms(gaussianized_DTR, LTR)



