# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:38:12 2022

@author: lucaf
"""

import numpy



def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th= - numpy.log(pi*Cfn) + numpy.log((1-pi)*Cfp)
    P= scores > th
    return numpy.int32(P)
        

def compute_confusion_matrix_binary(Pred, Labels):
    C=numpy.zeros((2,2))
    C[0,0] = ((Pred==0) * (Labels==0)).sum()
    C[0,1] = ((Pred==0) * (Labels==1)).sum()
    C[1,0] = ((Pred==1) * (Labels==0)).sum()
    C[1,1] = ((Pred==1) * (Labels==1)).sum()
    return C

def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr= CM[0,1] / (CM[0,1]+CM[1,1])
    fpr= CM[1,0] / (CM[0,0]+CM[1,0])
    return pi*Cfn*fnr + (1-pi) * Cfp * fpr

def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes= compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes/ min(pi*Cfn, (1-pi)*Cfp)
    

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred= assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM= compute_confusion_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)

def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf]) ])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return numpy.array(dcfList).min()

def bayes_error_plot(pArray, scores, labels, minCost=False):
    y=[]
    for p in pArray:
        pi=1.0/(1.0+numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1)) 
        return numpy.array(y)    


    
    
