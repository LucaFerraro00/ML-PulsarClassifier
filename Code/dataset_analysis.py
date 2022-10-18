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
        
        
#TBR: leave this following code here or move in to a new 'main.py' file ?
DTR,LTR = load('../Data/Train.txt')
plt.rc('font', size=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plot_histograms(DTR, LTR)
plot_scatters(DTR, LTR)
