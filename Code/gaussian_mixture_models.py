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


def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))


def train_gmm(DTR,LTR, iterations_LBG, Type):
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    
    gmm0 = LBG(D0, iterations_LBG, Type)
    gmm1 = LBG(D1, iterations_LBG, Type)
    return gmm0, gmm1
    
def compute_score(DTE,DTR,LTR, Options):
    if Options['iterations'] == None:
        Options['iterations'] = 8
    if Options['Type'] == None:
        Options['Type'] = 'full'
        
    gmm0, gmm1 = train_gmm(DTR, LTR, Options['iterations'], Options['Type'])
    
    ll0 = GMM_ll_perSample(DTE, gmm0)
    ll1 = GMM_ll_perSample(DTE, gmm1)
    return ll1-ll0
    
def LBG(D, iterations, Type,alfa=0.1):
    start_mu = mcol(D.mean(1))
    start_sigma = numpy.cov(D)
        
    gmm = [(1.0, start_mu, start_sigma)]
    for i in range(iterations):
            gmm_double = []
            for g in gmm:
                
                #Split the components of each g-components of gmm adding and subtracting epsilon to the mean g[1]
                #A good value for epsilon can be a displacement along the principal eigenvector of the covariance matrix 
                U, s, Vh = numpy.linalg.svd(g[2])
                d = U[:, 0:1] * s[0]**0.5 * alfa
                #g[0]=sigma, g[1]=mu, g[2]=w
                component1 = (g[0]/2, g[1]+d, g[2])
                component2 = (g[0]/2, g[1]-d, g[2])
                gmm_double.append(component1)
                gmm_double.append(component2)
            if Type == "full":
                gmm = EM_full(D, gmm_double)
            if Type == "diag":
                gmm = EM_diag(D, gmm_double)
            if Type == "full-tied":
                gmm = EM_full_tied(D, gmm_double)
            if Type == "diag-tied":
                gmm = EM_diag_tied(D, gmm_double)

    return gmm   
    
def logpdf_GAU_ND_opt1 (X,mu,C):
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

def EM_full(X, gmm):
        ll_new = None
        ll_old = None
        G = len(gmm)
        N = X.shape[1]
        
        psi = 0.01
        
        while ll_old is None or ll_new-ll_old>1e-6:
            ll_old = ll_new
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = logpdf_GAU_ND_opt1(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            ll_new = SM.sum() / N
            P = numpy.exp(SJ - SM)
            
            gmm_new = []
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (mrow(gamma)*X).T)
                w = Z/N
                mu = mcol(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                #constraint
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, mcol(s)*U.T)
                
                gmm_new.append((w, mu, sigma))
            gmm = gmm_new
        return gmm
    
def EM_diag(X, gmm):
    ll_new = None
    ll_old = None
    G = len(gmm)
    N = X.shape[1]
    
    psi = 0.01
    
    while ll_old is None or ll_new-ll_old>1e-6:
        ll_old = ll_new
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_opt1(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        ll_new = SM.sum() / N
        P = numpy.exp(SJ - SM)
        
        gmm_new = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
          
            sigma = sigma * numpy.eye(sigma.shape[0])
            
            #constraint
            U, s, _ = numpy.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = numpy.dot(U, mcol(s)*U.T)
            
            gmm_new.append((w, mu, sigma))
        gmm = gmm_new
    return gmm

def EM_full_tied(X, gmm):
    ll_new = None
    ll_old = None
    G = len(gmm)
    N = X.shape[1]
    
    psi = 0.01
    
    while ll_old is None or ll_new-ll_old>1e-6:
        ll_old = ll_new
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_opt1(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        ll_new = SM.sum() / N
        P = numpy.exp(SJ - SM)
        
        gmm_new = []
        summatory = numpy.zeros((X.shape[0], X.shape[0]))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            summatory += Z*sigma
            gmm_new.append((w, mu, sigma))
       
        sigma = summatory / G
        #constraint
        U, s, _ = numpy.linalg.svd(sigma)
        s[s<psi] = psi
        sigma = numpy.dot(U, mcol(s)*U.T)
        gmm = gmm_new
    return gmm
    
def EM_diag_tied(X, gmm):
    ll_new = None
    ll_old = None
    G = len(gmm)
    N = X.shape[1]
    
    psi = 0.01
    
    while ll_old is None or ll_new-ll_old>1e-6:
        ll_old = ll_new
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_opt1(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        ll_new = SM.sum() / N
        P = numpy.exp(SJ - SM)
        
        gmm_new = []
        summatory = numpy.zeros((X.shape[0], X.shape[0]))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            #diagonalization
            sigma = sigma * numpy.eye(sigma.shape[0])
            summatory += Z*sigma
            gmm_new.append((w, mu, sigma))
      
        sigma = summatory / G
        #constraint
        U, s, _ = numpy.linalg.svd(sigma)
        s[s<psi] = psi
        sigma = numpy.dot(U, mcol(s)*U.T)
        gmm = gmm_new
    return gmm


