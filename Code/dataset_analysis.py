# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:15:35 2022

@author: lucaf
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt



def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    
    #setup visualization font
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:11]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def compute_mean (D):
    mu = D.mean(1) #D is a matrix where each column is a vector, i need the mean for each feature
    return mu.reshape(mu.shape[0],1)

def compute_variance (D):
    sigma = D.std(1) #D is a matrix where each column is a vector, i need the variance for each feature
    return sigma.reshape(sigma.shape[0],1)

def scale_ZNormalization(D, mu, sigma):
    scaled_DTR = (D-mu) #TBR: CORRECT OR NOT???????? 
    scaled_DTR = scaled_DTR / sigma
    return scaled_DTR
    
def plot_histograms(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
        }

    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 80, density = True, alpha = 0.8, label = '0 - Low quality' , color= 'red')
        plt.hist(D1[dIdx, :], bins = 80, density = True, alpha = 0.8, label = '1 - Good quality', color= 'green')
        #TBR: bins represents the 'number of towers' showed in the histogram
        #TBR: density=true: draw and return a probability density: each bin will display the bin's raw count divided by the total number of counts and the bin width (density = counts / (sum(counts) * np.diff(bins))), so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1).
        #TBR: The alpha blending value, between 0 (transparent) and 1 (opaque).

    
        plt.legend()
        plt.tight_layout() # TBR: Use with non-default font size to keep axis label inside the figure
        plt.savefig('../Images/DatasetAnalysis/histogram_%d.pdf' % dIdx)
    plt.show()
    
def plot_scatters(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
        }

    for dIdx1 in range(11):
        for dIdx2 in range(11):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = '0 - Low quality' , color= 'red', alpha=0.2)
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = '1 - Good quality', color= 'green', alpha=0.2)
            #TBR: The alpha blending value, between 0 (transparent) and 1 (opaque).
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('../Images/DatasetAnalysis/scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()
        
#---------functions for computing Gaussiziation-------------------------
#for each sample of the feature I have to call this function
def compute_rank(x_one_value, x_all_samples):
    rank=0
    for xi in x_all_samples:
        if xi<x_one_value:
            rank+=1
    return (rank +1)/ (x_all_samples.shape[1] +2)

