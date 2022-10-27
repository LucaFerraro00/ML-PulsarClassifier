# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:42:22 2022

@author: lucaf
"""

import numpy
import dataset_analysis as analys


def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

#to be changed
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)


def full_compute_mean_covariance(D):
    mu=mcol(D.mean(1))
    C=numpy.dot(D-mu, (D-mu).T)/float(D.shape[1])
    return mu,C

def naive_byas_compute_mu_covariance(D):
    mu = D.mean(1)
    mu=mu.reshape(mu.size,1)
    DC = D - mu
    C = numpy.dot(DC,DC.T)
    C=C/float(D.shape[1])
    C_diagonal = C * numpy.identity(C.shape[0])
    return mu,C_diagonal

def tied_compute_mu_covariance(D,L):
    C_tied=0
    for i in range(0,3):
        Dclass=D[:,L==i]
        mu = Dclass.mean(1)
        mu=mu.reshape(mu.size,1)
        DC = Dclass - mu
        C = numpy.dot(DC,DC.T)
        C=C/float(Dclass.shape[1])
        C_tied+=C * Dclass.shape[1]
    C_tied = C_tied/D.shape[1]
    return mu,C_tied

def logpdf_onesample(x, mu, C):
    P=numpy.linalg.inv(C)
    res= -0.5 * x.shape[0] * numpy.log(2*numpy.pi)
    res += -0.5*numpy.linalg.slogdet(C)[1] #slogdet return an array, the second element of the aray is the (natural) logarithm of the determinant of an array
    res += -0.5* numpy.dot( (x-mu).T,numpy.dot(P,(x-mu)))
    #res is a 1x1 matrix. I transform res in a 1-dimensional vector that contains only one value with the function ravel()
    return res.ravel()

def logpdf_GAU_ND(X,mu,C): #this function take as input a matrix of samples
    y=[logpdf_onesample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    #maybe alternatively mcol(X[: , i])
    return numpy.array(y).ravel()
    #if i don't use ravel() i obtain a row-vector, while I want a 1-Dimensional vector

def gaussian_classifier(D,L,classifier_type):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    mu0, C0 = full_compute_mean_covariance(D0)
    mu1, C1 = full_compute_mean_covariance(D1)
    
    """
    match classifier_type:
        case 'full':
             full_compute_mu_covariance(D, L)
        ...
    """
    
    (DTrain, LTrain), (DTest, LTest) = split_db_2to1(D, L)
    
    #why accuracy doesn't change if I gaussianize or not the features
    DTrain = analys.gaussianize_training(DTrain)
    DTest = analys.gaussianize_evaluation(DTest, DTrain)
    
    
    ScoreJoint=numpy.zeros((2, DTest.shape[1]))
    ScoreJoint[0, :] = numpy.exp(logpdf_GAU_ND(DTest, mu0, C0).ravel())* 0.5 #class posterior probability is set to 0.5 for the moment
    ScoreJoint[1, :] = numpy.exp(logpdf_GAU_ND(DTest, mu1, C1).ravel())* 0.5 #class posterior probability is set to 0.5 for the moment

    SMarginal = ScoreJoint.sum(0)
    Posterior = ScoreJoint/mrow(SMarginal)
    LPred=Posterior.argmax(0)
    
    Compare=LTest==LPred #Compare is an array that contain true when the compared elements are equal, False otherwise
    TotCorrect=Compare.sum()
    Accuracy=TotCorrect/LTest.size
    return LPred, Accuracy

    

    
    
    
