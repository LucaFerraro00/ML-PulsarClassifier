# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:03:44 2022

@author: lucaf
"""

import numpy
import scipy
import validate
import matplotlib.pyplot as plt


K=5

def logreg_obj_wrap(DTR, LTR, l, piT): #l is lamda (used for regularization)
    #compute the labels Z
    Z= LTR * 2.0 -1.0
    M=DTR.shape[0]
    def logreg_obj(v):
        #v is a vector that contains [w,b]
        #extract b and w
        w = v [0:M]
        b = v[-1]
        
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        Z0 = Z[LTR==0]
        Z1 = Z[LTR==1]
        
        S1=numpy.dot(w.T, DTR1)+b
        S0=numpy.dot(w.T, DTR0)+b #S= score= exponent of e = wTxi +b
        cxe=numpy.logaddexp(0,-S1*Z1).mean()* piT #cxe is the cross entropy = log [e^0 + e^(-sz)]
        cxe= cxe + numpy.logaddexp(0,-S0*Z0).mean()* (1-piT)
        return cxe+0.5*l*numpy.linalg.norm(w)**2 #I add also the regularization term
    return logreg_obj

def train_log_reg(DTR, LTR, lambdaa, piT):
    logreg_objective=logreg_obj_wrap(DTR, LTR, lambdaa, piT)
    _v,j,_d = scipy.optimize.fmin_l_bfgs_b(logreg_objective, numpy.zeros(DTR.shape[0]+1) , approx_grad=True) #numpy.zeros(DTR.shape[0]+1) is the starting point. I am providing w and b equal to 0 as starting point
    #I can recover optimal w* and b* from _v:
    _w=_v[0:DTR.shape[0]]
    _b=_v[-1]
    return _w,_b

def compute_score(DTE,DTR,LTR, Options):
    #when this function is called for plotting min_dcf lambdaa != None
    #when this function is called for evaluation purpose we set lamdaa = 0 (no regularization)
    if Options['lambdaa']== None:
        Options['lambdaa'] = 0
    
    if Options['piT'] == None:
        Options['piT']=0.5
    
    _w,_b= train_log_reg(DTR, LTR, Options['lambdaa'], Options['piT'])
    scores = numpy.dot(_w.T,DTE)+_b
    return scores

def plot_minDCF_wrt_lamda(D,L, gaussianize):
    min_DCFs=[]
    for pi in [0.1, 0.5, 0.9]:
        lambdas = numpy.logspace(-6,3, num = 10)
        for l in lambdas:
                Options= {'lambdaa':l,
                          'piT':0.5}
                min_dcf_kfold = validate.kfold(D, L, K, pi, compute_score, Options ) 
                min_DCFs.append(min_dcf_kfold)

    min_DCFs_p0 = min_DCFs[0:10] #min_DCF results with prior = 0.1
    min_DCFs_p1 = min_DCFs[10:20] #min_DCF results with prior = 0.5
    min_DCFs_p2 = min_DCFs[20:30] #min_DCF results with prior = 0.9

    plt.figure()
    plt.plot(lambdas, min_DCFs_p0, label='prior=0.1')
    plt.plot(lambdas, min_DCFs_p1, label='prior=0.5')
    plt.plot(lambdas, min_DCFs_p2, label='prior=0.9')
    plt.legend()
    plt.semilogx()
    plt.xlabel("λ")
    plt.ylabel("min_DCF")
    if gaussianize:
        plt.savefig("../Images/min_DCF_lamda_log_reg_gaussianized.jpg")
    else:
        plt.savefig("../Images/min_DCF_lamda_log_reg_raw.jpg")
    plt.show()
    return min_DCFs


    
    
    
    
