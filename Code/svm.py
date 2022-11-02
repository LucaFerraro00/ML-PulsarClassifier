# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:57:08 2022

@author: lucaf
"""
import numpy
import scipy


def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))
"""
----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------LINEAR SVM-------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""
#we are considering a binary problem, we're supposing that LTR is labels 0 or 1
#C and K are hyperparameters
def train_SVM_linear(DTR, LTR, C, K=1):
    
    #first we simulate the bias by extending our data with ones, this allow us to simulate the effect of a bias
    DTREXT=numpy.vstack([DTR, numpy.ones((1,DTR.shape[1]))*K])
    #DTREXt is the original data matrix  wich contains the original training data x1...xn. Each x has an added feature equals to K=1 appendend
    
    #compute Z to be used to do compute: zi(wTx+b)
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    #compute matrix H which is inside the dual formula
    #Hij=zizjxiTxj (xi and xj are vectors)
    G=numpy.dot(DTREXT.T,DTREXT)
    H=mcol(Z) * mrow(Z) * G #mcol(Z) * mrow(Z) result in a matrix of the same dimension of G?
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad
    
    def JPrimal(w):
        S = numpy.dot(mrow(w), DTREXT)
        loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
        return 0.5 * numpy.linalg.norm(w)**2+C*loss
    
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, 
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = [(0,C)]*DTR.shape[1], #create a list of DTR.shape[1] elements. Each element is a tuple (0,C)
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation (see lab_8_lecture) 
                                                      )
    
    wStar = numpy.dot(DTREXT, mcol(alphaStar)*mcol(Z))

    return wStar, alphaStar





"""
----------------------------------------------------------------------------------------------------------------------
----------------------------------KERNEL SVM (example with gaussian radial basis)-------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""

#it changes only the way of computing C with respect to linear SVM
def train_SVM_kernel(DTR, LTR, C, gamma, K=1):
    
    #first we simulate the bias by extending our data with ones, this allow us to simulate the effect of a bias
    DTREXT=numpy.vstack([DTR, numpy.ones((1,DTR.shape[1]))*K])
    #DTREXt is the original data matrix  wich contains the original training data x1...xn. Each x has an added feature equals to K=1 appendend
    
    #compute Z to be used to do compute: zi(wTx+b)
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    Dist= mcol((DTR**2).sum(0)) +mrow((DTR**2).sum(0)) - 2 * numpy.dot(DTR.T,DTR)
    """
    Alternatively Dist can be computed with for loops
    Dist = numpy.zeros(DTR.shape[0],DTR.shape[0])
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            xi= DTR[:, i]
            xj= DTR[:, j]
            Dist[i,j]=numpy.linalg.norm(xi-xj)**2
    """
    H= numpy.exp(-gamma* Dist) + K #K account for byas term
    H=mcol(Z) *mrow(Z) * H
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad
    
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, 
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = [(0,C)]*DTR.shape[1], #create a list of DTR.shape[1] elements. Each element is a tuple (0,C)
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation (see lab_8_lecture) 
                                                      )
    
    wStar = numpy.dot(DTREXT, mcol(alphaStar)*mcol(Z))
    
    return wStar, alphaStar
