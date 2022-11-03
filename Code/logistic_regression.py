# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:03:44 2022

@author: lucaf
"""

import numpy
import scipy

Lambda = 0.0002

def logreg_obj_wrap(DTR, LTR, l): #l is lamda (used for regularization)
    #compute the labels Z
    Z= LTR * 2.0 -1.0
    M=DTR.shape[0]
    def logreg_obj(v):
        #v is a vector that contains [w,b]
        #extract b and w
        w = v [0:M]
        b = v[-1]
        S=numpy.dot(w.T, DTR)+b #S= score= exponent of e = wTxi +b
        cxe=numpy.logaddexp(0,-S*Z).mean() #cxe is the cross entropy = log [e^0 + e^(-sz)]
        return cxe+0.5*l*numpy.linalg.norm(w)**2 #I add also the regularization term
    return logreg_obj

def train_log_reg(DTR, LTR):
    logreg_objective=logreg_obj_wrap(DTR, LTR, Lambda)
    _v,j,_d = scipy.optimize.fmin_l_bfgs_b(logreg_objective, numpy.zeros(DTR.shape[0]+1) , approx_grad=True) #numpy.zeros(DTR.shape[0]+1) is the starting point. I am providing w and b equal to 0 as starting point
    #I can recover optimal w* and b* from _v:
    _w=_v[0:DTR.shape[0]]
    _b=_v[-1]
    return _w,_b

def compute_score(DTE,DTR,LTR):
    _w,_b= train_log_reg(DTR, LTR)
    score = numpy.dot(_w.T,DTE)+_b
    print(score)
    return score
