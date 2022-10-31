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

def train_gaussian_classifier(DTR,LTR):
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    #Insert kind of switch case which compure full, naive or tied
    mu0, C0 = full_compute_mean_covariance(D0)
    mu1, C1 = full_compute_mean_covariance(D1)
    return mu0,C0,mu1,C1

def compute_score(DTE,DTR,LTR):
    mu0,C0,mu1,C1= train_gaussian_classifier(DTR, LTR)
    log_density_c0 = logpdf_GAU_ND(DTE, mu0, C0)
    log_density_c1 = logpdf_GAU_ND(DTE, mu0, C1)
    score = log_density_c1 - log_density_c0
    return score


    

    
    
    
