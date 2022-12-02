# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:15:35 2022

@author: lucaf
"""

import numpy
import pylab
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as statist



def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))


def loda_training_set(fname):
    #setup visualization font
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    DTR = numpy.hstack(DList)
    LTR =numpy.array(labelsList, dtype=numpy.int32)
    
    mean=compute_mean(DTR)
    std=compute_std(DTR)
    print ('The mean of the features for the entire training set is:')
    print(mean.ravel())
    print ('The standard deviation of the features for the entire training set is:')
    print(std.ravel())
    
    return DTR, LTR

def loda_evaluation_set(fname):

    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    DEV = numpy.hstack(DList)
    LEV =numpy.array(labelsList, dtype=numpy.int32)

    return DEV, LEV



def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1]) #idx contains N numbers from 0 to N (where is equals to number of training samples) in a random  order
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain] 
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return DTR, LTR, DTE, LTE

def compute_mean (D):
    mu = D.mean(1) #D is a matrix where each column is a vector, i need the mean for each feature
    return mu.reshape(mu.shape[0],1)

def compute_std (D):
    sigma = D.std(1) #D is a matrix where each column is a vector, i need the variance for each feature
    return sigma.reshape(sigma.shape[0],1)

def scale_ZNormalization(DTR, DEV = None, normalize_ev=False): 
    mu=compute_mean(DTR)
    sigma=compute_std(DTR)
    scaled_DTR = (DTR-mu) 
    scaled_DTR = scaled_DTR / sigma
    print('Z-Normalization done!')
    if normalize_ev:
        DEV=(DEV-mu)/sigma #normalize evaluation_set with mean and std of training set
    return scaled_DTR, DEV
    
def plot_histograms(D, L, gaussianize):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
         0: 'Mean of the integrated profile',
         1: 'Standard deviation of the integrated profile',
         2: 'Excess kurtosis of the integrated profile',
         3: 'skewness of the integrated profile',
         4: 'Mean of the DM-SNR curve',
         5: 'Standard deviation of the DM-SNR curve',
         6: 'Excess kurtosis of the DM-SNR curve',
         7: 'skewness of the DM-SNR curve'
        }

    for dIdx in range(8):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 45, density = True, alpha = 0.8, label = '0 - Not pulsar' , color= 'red')
        plt.hist(D1[dIdx, :], bins = 45, density = True, alpha = 0.8, label = '1 - Pulsar', color= 'green')
        #TBR: bins represents the 'number of towers' showed in the histogram
        #TBR: density=true: draw and return a probability density: each bin will display the bin's raw count divided by the total number of counts and the bin width (density = counts / (sum(counts) * np.diff(bins))), so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1).
        #TBR: The alpha blending value, between 0 (transparent) and 1 (opaque).

        plt.legend()
        plt.tight_layout() # TBR: Use with non-default font size to keep axis label inside the figure
        if gaussianize:
            plt.savefig('../Images/DatasetAnalysis/histogram_afterGaussianization_%d.jpg' % dIdx)
        else:
            plt.savefig('../Images/DatasetAnalysis/histogram_beforeGaussianization_%d.jpg' % dIdx)

    plt.show()
    
def plot_scatters(D, L, gaussianize):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
         0: 'Mean of the integrated profile',
         1: 'Standard deviation of the integrated profile',
         2: 'Excess kurtosis of the integrated profile',
         3: 'skewness of the integrated profile',
         4: 'Mean of the DM-SNR curve',
         5: 'Standard deviation of the DM-SNR curve',
         6: 'Excess kurtosis of the DM-SNR curve',
         7: 'skewness of the DM-SNR curve'
        }

    for dIdx1 in range(8):
        for dIdx2 in range(8):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = '0 - Not pulsar' , color= 'red', alpha=0.2)
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = '1 - Pulsar', color= 'green', alpha=0.2)
            #TBR: The alpha blending value, between 0 (transparent) and 1 (opaque).
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('../Images/DatasetAnalysis/scatter_%d_%d.jpg' % (dIdx1, dIdx2))
        plt.show()
        
        
#################################################
#---------GAUSSIANIZATION-----------------------#
#################################################

#for each sample of the feature I have to call this function
def compute_rank(x_one_value, x_all_samples):
    rank=0
    for xi in x_all_samples:
        if xi<x_one_value:
            rank+=1
    return (rank +1)/ (x_all_samples.shape[0] +2)

"""
def gaussianize (DTR):
    rank_DTR = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rank_DTR[i][j] = (DTR[i] < DTR[i][j]).sum()
    rank_DTR = (rank_DTR+1) / (DTR.shape[1]+2)
    return statist.norm.ppf(rank_DTR) 
"""

def gaussianize_training (DTR):
    rank_DTR = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rank_DTR[i][j] = compute_rank(DTR[i][j], DTR[i])
    return statist.norm.ppf(rank_DTR)

def gaussianize_evaluation (DTE, DTR):
    rank_DTE = numpy.zeros(DTE.shape)
    for i in range(DTE.shape[0]):
        for j in range(DTE.shape[1]):
            rank_DTE[i][j] = compute_rank(DTE[i][j], DTR[i])
    return statist.norm.ppf(rank_DTE)


####################################
#-----------HEATMAP----------------#
####################################

def pearce_correlation_map (D, L, gaussianize):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    plt.figure()
    plt.imshow(numpy.absolute(numpy.corrcoef(D0)), cmap='Oranges')
    if gaussianize:
        plt.savefig('../Images/DatasetAnalysis/correlation_class_zero_afterGauss.jpg')
    else:
        plt.savefig('../Images/DatasetAnalysis/correlation_class_zero_beforeGauss.jpg')
    plt.figure()
    plt.imshow(numpy.absolute(numpy.corrcoef(D1)), cmap='Greens')
    if gaussianize:
        plt.savefig('../Images/DatasetAnalysis/correlation_class_one_afterGauss.jpg')
    else:
        plt.savefig('../Images/DatasetAnalysis/correlation_class_one_beforeGauss.jpg')
    plt.figure()
    plt.imshow(numpy.absolute(numpy.corrcoef(D)), cmap='Greys')
    if gaussianize:
        plt.savefig('../Images/DatasetAnalysis/correlation_all_training_set_afterGauss.jpg')
    else:
        plt.savefig('../Images/DatasetAnalysis/correlation_all_training_set_beforeGauss.jpg')
    plt.show()

def plot_heatmap(correlations):  
    plt.figure()
    plt.imshow(correlations)
    plt.show()
    
    
####################################
#-------------PCA------------------#
####################################

def pca(m, D):
    mu = compute_mean(D)
    DCentered = D - mu #center the data
    C=numpy.dot(DCentered,DCentered.transpose())/float(D.shape[1]) #compute emprical covariance matrix C
    _, U = numpy.linalg.eigh(C) #U contains the eigenvectors corresponding to eigenvalues of C in ascending order
    #I need to take the first m eigenvectors corresponding to the m largest eigenvalues
    P = U[:, ::-1][:, 0:m] #I invert the columns of U then I take the firsts m
    DProjected = numpy.dot(P.T, D)
    return DProjected, P


    
    
    
    
    
    
    