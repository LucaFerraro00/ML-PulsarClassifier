# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:43:37 2022

@author: lucaf
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:08:28 2022

@author: lucaf
"""

import numpy
import scipy.special
import json


def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))

#copy and paste from lab 4
def logpdf_GAU_ND_opt1 (X,mu,C):
#una prima idea per rendere piu efficiente questa funzione è dividerla in due parti:
#una parte della formula può essere calcolata solo una volta, solo l'addendo che considera xi deve essere
#calcolata iterativamente
     P=numpy.linalg.inv(C)
     const= -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
     const += -0.5*numpy.linalg.slogdet(C)[1]
     
     Y=[]
     for i in range (X.shape[1]):
         x=X[: , i:i+1]
         res=const -0.5 * numpy.dot( (x-mu).T, numpy.dot(P,(x-mu)))
         Y.append(res)
     return numpy.array(Y).ravel()

def GMM_ll_perSample(X, gmm):
    #return an array where each element is the log-likelihood of each element of the X matrix
    G= len(gmm)
    N=X.shape[1]
    S=numpy.zeros((G,N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_opt1(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0]) #the second addend represents the prior for each gaussian
    return scipy.special.logsumexp(S,axis=0)

def GMM_EM(X,gmm):
    llNew=None
    llOld=None
    G=len(gmm)
    N=X.shape[1]
    while llOld is None or llNew - llOld > 1e-6 :
        llOld=llNew
        SJ=numpy.zeros((G,N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_opt1(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0]) #the second addend represents the prior for each gaussian
        SM= scipy.special.logsumexp(SJ,axis=0) #M stays for Marginal
        llNew=SM.sum()/N #This is the average (because divided by N) log-likelihood for all the dataset
        P =numpy.exp(SJ-SM) #P is the posterior. P is computed the same as MVG
        #this following is the new part for GMM different from MVG:
        gmmNew=[] #we need to compute the updated parameter for our gmm. We'll use zero, first and second order statistics
        for g in range(G):
            gamma=P[g,:]
            Z=gamma.sum() #Z=zero order statistic
            F=(mrow(gamma)*X).sum(1)
            S=numpy.dot(X,(mrow(gamma)*X).T)
            w=Z/N
            mu = mcol(F/Z)
            Sigma=S/Z - numpy.dot(mu, mu.T)
            gmmNew.append((w,mu,Sigma))
        gmm=gmmNew
        print(llNew)
    print (llNew-llOld) #Check that new lilelihood is always greather than old likelihood
    return gmm

