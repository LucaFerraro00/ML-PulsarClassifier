# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:33:19 2022

@author: lucaf
"""

import validate
import numpy
import matplotlib.pyplot as plt
import gaussian_classifiers as gauss
import logistic_regression as log_reg
import dataset_analysis as analys

def min_vs_act(D,L):
    pi_array = numpy.linspace(-4, 4, 20)

    Options={}
    _ , scores_gauss, labels_gauss = validate.kfold(D, L, 5, 0.5, gauss.compute_score_tied_full  , Options)
    y_min1, y_act1 = validate.bayes_error(pi_array, scores_gauss, labels_gauss)
    for pi in [0.1, 0.5, 0.9]:
        act_dcf= validate.compute_act_DCF(scores_gauss, labels_gauss, pi, 1, 1)
        print ('tied full-cov MVG: pi = %f --> act_DCF = %f'%(pi,act_dcf))
    
    D = analys.pca(7, D)
    Options={
    'lambdaa' : 1e-06,
    'piT' : 0.1,
    }
    _ , scores_logReg, labels_logReg = validate.kfold(D, L, 5, 0.5, log_reg.compute_score , Options)
    y_min2, y_act2= validate.bayes_error(pi_array, scores_logReg, labels_logReg)
    for pi in [0.1, 0.5, 0.9]:
        act_dcf= validate.compute_act_DCF(scores_logReg, labels_logReg, pi, 1, 1)
        print ('linear logisic regression: pi = %f --> act_DCF = %f'%(pi,act_dcf))
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r--',  label='MVG min_DCF')
    plt.plot(pi_array, y_act1, 'r', label='MVG act_DCF')
    plt.plot(pi_array, y_min2, 'b--',  label='log-reg min_DCF')
    plt.plot(pi_array, y_act2, 'b', label='log-reg act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig("../Images/ScoreCalibration/actVSmin.pdf")
    

def optimal_threshold (D,L):
    #best model 1
    Options={}
    _, scores, labels = validate.kfold(D, L, 5, 0.5, gauss.compute_score_tied_full  , Options)
    scores_TR, LTR, scores_TE, LTE = split_scores(scores, labels) #split and shuffle scores
    t_array = numpy.arange (-10, 10, 0.1) #To be changed ??
    for pi in [0.1, 0.5, 0.9]:
        threshold_act_DCF = {}
        for t in t_array:
            act_dcf= validate.compute_act_DCF(scores_TR, LTR, pi, 1, 1, th=t)
            threshold_act_DCF[t]=act_dcf
        best_t = min(threshold_act_DCF, key=threshold_act_DCF.get)
        best_cost_evaluation = validate.compute_act_DCF(scores_TE, LTE, pi, 1, 1, th=best_t)
        print('full tied cov MVG: prior pi= % f - best threshold = %f' %(pi, best_t))
        print('full tied cov MVG: prior pi= % f - act_dcf computed on evaluation scores set for best threshold = %f' %(pi,best_cost_evaluation))
   
    #best model 2
    D = analys.pca(7, D)
    Options={
    'lambdaa' : 1e-06,
    'piT' : 0.1,
    }
    _, scores, labels = validate.kfold(D, L, 5, 0.5, log_reg.compute_score , Options)
    scores_TR, LTR, scores_TE, LTE = split_scores(scores, labels) #split and shuffle scores
    t_array = numpy.arange (-100, 100, 0.1) #To be changed ??
    for pi in [0.1, 0.5, 0.9]:
        threshold_act_DCF = {}
        for t in t_array:
            act_dcf= validate.compute_act_DCF(scores_TR, LTR, pi, 1, 1, th=t)
            threshold_act_DCF[t]=act_dcf
        best_t = min(threshold_act_DCF, key=threshold_act_DCF.get)
        best_cost_evaluation = validate.compute_act_DCF(scores_TE, LTE, pi, 1, 1, th=best_t)
        print('linear log reg Lambda = 1e-06 - piT=0.1: prior pi = %f- best threshold = %f' %(pi, best_t))
        print('linear log reg Lambda = 1e-06 - piT=0.1: prior pi= % f - act_dcf computed on evaluation scores set for best threshold = %f' %(pi,best_cost_evaluation))
        

def split_scores(D,L, seed=0):
    nTrain = int(D.shape[0]*8.0/10.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[0]) #idx contains N numbers from 0 to N (where is equals to number of training samples) in a random  order
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[idxTrain] 
    DTE = D[idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return DTR, LTR, DTE, LTE
