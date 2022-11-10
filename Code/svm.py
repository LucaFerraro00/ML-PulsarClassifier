# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:57:08 2022

@author: lucaf
"""
import numpy
import scipy
import validate
import matplotlib.pyplot as plt


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
def train_SVM_linear(DTR, LTR, C, piT, K=1):
    
    #first we simulate the bias by extending our data with ones, this allow us to simulate the effect of a bias
    DTREXT=numpy.vstack([DTR, numpy.ones((1,DTR.shape[1]))])
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
    
    DTR0 = DTR[:, LTR==0]
    DTR1 = DTR[:, LTR==1]
    pi_emp_F = (DTR0.shape[1] / DTR.shape[1])
    pi_emp_T = (DTR1.shape[1] / DTR.shape[1])
    CT = C * piT/pi_emp_T 
    CF = C * piT/pi_emp_F
    
    #bounds: a list of DTR.shape[1] elements. Each element is a tuple (0,CT) or (0,CF)
    bounds = [(0,1)]*DTR.shape[1] #Initialize the array with random values 
    for i in range(DTR.shape[1]):
        if LTR[i]==0:
            bounds[i]=(0,CF)
        else:
            bounds[i]=(0,CT)
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, #function to minimze
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = bounds,
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation  
                                                      )
    
    wStar = numpy.dot(DTREXT, mcol(alphaStar)*mcol(Z))

    return wStar, alphaStar

K=1
K_fold=5
def compute_score_linear(DTE,DTR,LTR, Options):
    if Options['C']== None:
        Options['C'] = 0
    
    if Options['piT'] == None:
        Options['piT']=0.5
    w,_alfa= train_SVM_linear(DTR, LTR, Options['C'], Options['piT'])
    DTE_EXT=numpy.vstack([DTE, numpy.ones((1,DTE.shape[1]))])
    score = numpy.dot(w.T,DTE_EXT)
    return score.ravel()
    
def plot_linear_minDCF_wrt_C(D,L,gaussianize):
    min_DCFs=[]
    for pi in [0.1, 0.5, 0.9]:
        C_array = numpy.logspace(-5,2, num = 8)
        for C in C_array:
                Options= {'C': C,
                          'piT':0.5}
                min_dcf_kfold = validate.kfold(D, L, K_fold, pi, compute_score_linear, Options ) 
                min_DCFs.append(min_dcf_kfold)

    min_DCFs_p0 = min_DCFs[0:8] #min_DCF results with prior = 0.1
    min_DCFs_p1 = min_DCFs[8:16] #min_DCF results with prior = 0.5
    min_DCFs_p2 = min_DCFs[16:24] #min_DCF results with prior = 0.9

    plt.figure()
    plt.plot(C_array, min_DCFs_p0, label='prior=0.1')
    plt.plot(C_array, min_DCFs_p1, label='prior=0.5')
    plt.plot(C_array, min_DCFs_p2, label='prior=0.9')
    plt.legend()
    plt.semilogx()
    plt.xlabel("C")
    plt.ylabel("min_DCF")
    if gaussianize:
        plt.savefig("../Images/min_DCF_C_linearSVM_gaussianized.jpg")
    else:
        plt.savefig("../Images/min_DCF_C_linearSVM_raw.jpg")
    plt.show()
    return min_DCFs

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
