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

def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))


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
    
#first approach to calibrate the score: find optimal threshold on training scores samples, evaluate actual dcf with optimal threshold on evaluation scores samples
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
        best_cost_validation = validate.compute_act_DCF(scores_TE, LTE, pi, 1, 1, th=best_t)
        theoretical_cost_validation = validate.compute_act_DCF(scores_TE, LTE, pi, 1, 1, th=None) #if th=None, computed_act_DCF function will use theoretical threshold
        min_DCF_validation = validate.compute_min_DCF(scores_TE, LTE, pi, 1, 1)
        print('full tied cov MVG: prior pi= % f - act_dcf computed on evaluation scores set for best threshold = %f' %(pi,best_cost_validation))
        print('full tied cov MVG: prior pi= % f - act_dcf computed on evaluation scores set for theoretical threshold = %f' %(pi,theoretical_cost_validation))
        print('full tied cov MVG: prior pi= % f - min_DCF computed on evaluation scores set = %f' %(pi,min_DCF_validation))
   
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
        best_cost_validation = validate.compute_act_DCF(scores_TE, LTE, pi, 1, 1, th=best_t)
        theoretical_cost_validation = validate.compute_act_DCF(scores_TE, LTE, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        min_DCF_validation = validate.compute_min_DCF(scores_TE, LTE, pi, 1, 1)
        print('linear log reg Lambda = 1e-06 - piT=0.1: prior pi= % f - act_dcf computed on validation scores set for best threshold = %f' %(pi,best_cost_validation))
        print('linear log reg Lambda = 1e-06 - piT=0.1: prior pi= % f - act_dcf computed on validation scores set for theoretical threshold = %f' %(pi,theoretical_cost_validation))
        print('linear log reg Lambda = 1e-06 - piT=0.1: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))

def score_trasformation(scores_TR, LTR, scores_TE,pi):
    scores_TR= mrow(scores_TR)
    scores_TE= mrow(scores_TE)
    alfa, beta_prime = log_reg.train_log_reg(scores_TR, LTR, 1e-06, pi) #??this pi should be fixed or not????
    new_scores= numpy.dot(alfa.T,scores_TE)+beta_prime - numpy.log(pi/(1-pi))
    return new_scores
    
#second approach to calibrate score: trasform score so that theoretical threshold provide close to optimal values over different applications
def validate_score_trasformation(D,L):
    #best model 1
    D_pca = analys.pca(7, D)
    Options={
    'lambdaa' : 1e-06,
    'piT' : 0.1, 
    }
    for pi in [0.1, 0.5, 0.9]:
        _, scores, labels = validate.kfold(D_pca, L, 5, 0.5, log_reg.compute_score , Options)
        scores_TR, LTR, scores_TE, LTE = split_scores(scores, labels) #split and shuffle scores
        calibrated_scores = score_trasformation(scores_TR, LTR, scores_TE, pi)

        min_DCF_validation = validate.compute_min_DCF(scores_TE, LTE, pi, 1, 1)
        print('linear log reg Lambda = 1e-06 - piT=0.1: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        act_DCF_non_calibrated= validate.compute_act_DCF(scores_TE, LTE, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        print('act DCF computed on theoretical threshold (pi=%f) without calibration = %f'%(pi,act_DCF_non_calibrated))
    
        print('result of act_DCF computed on trasformed scores with log reg pi = %f reported below:'%pi)
        for pi in [0.1, 0.5, 0.9]:
            act_DCF= validate.compute_act_DCF(calibrated_scores, LTE, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            print('act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF))
        print('')
    
    #best model 2 !!!

def min_vs_act_after_calibration(D,L):
    pi_array = numpy.linspace(-4, 4, 20)
    
    D = analys.pca(7, D)
    Options={
    'lambdaa' : 1e-06,
    'piT' : 0.1,
    }
    _ , scores, labels = validate.kfold(D, L, 5, 0.5, log_reg.compute_score , Options)
    scores_TR, LTR, scores_TE, LTE = split_scores(scores, labels) #split and shuffle scores
    calibrated_scores = score_trasformation(scores_TR, LTR, scores_TE, 0.5)
    y_min2, y_act2= validate.bayes_error(pi_array, calibrated_scores, LTE)
    for pi in [0.1, 0.5, 0.9]:
        act_dcf= validate.compute_act_DCF(calibrated_scores, LTE, pi, 1, 1, th=None)
        print ('linear logisic regression: pi = %f --> act_DCF after calibrationi = %f'%(pi,act_dcf))
    
    plt.figure()
    #plt.plot(pi_array, y_min1, 'r--',  label='MVG min_DCF')
    #plt.plot(pi_array, y_act1, 'r', label='MVG act_DCF')
    plt.plot(pi_array, y_min2, 'b--',  label='log-reg min_DCF')
    plt.plot(pi_array, y_act2, 'b', label='log-reg act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig("../Images/ScoreCalibration/actVSmin_after_calibration.pdf")
    

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
