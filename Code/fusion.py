# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:06:38 2022

@author: lucaf
"""
import validate
import gaussian_classifiers as gauss
import dataset_analysis as analys
import logistic_regression as log_reg
import score_calibration as calibration
import numpy

def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))


def fused_scores(D,L):
    #best model 
    Options = {} 
    _ , scores1, labels = validate.kfold(D, L, 5, 0.5, gauss.compute_score_tied_full, Options)
    scores1_TR, LTR, scores1_TE, LTE = calibration.split_scores(scores1, labels) #split and shuffle scores
    scores1_TR = mrow(scores1_TR)
    scores1_TE = mrow(scores1_TE)

    #best model 2
    D = analys.pca(7, D)
    Options = {'lambdaa':1e-06 ,
               'piT': 0.1 }
    _ , scores2, labels = validate.kfold(D, L, 5, 0.5, log_reg.compute_score, Options) #labels returned are the same of best model 1
    scores2_TR, LTR, scores2_TE, LTE = calibration.split_scores(scores2, labels) #same split used for best model 1 above. Labels returned are the same of best model 1
    scores2_TR = mrow(scores2_TR)
    scores2_TE = mrow(scores2_TE)
    
    all_scores_TR = numpy.vstack((scores1_TR, scores2_TR))
    all_scores_TE = numpy.vstack((scores1_TE, scores2_TE))
    pi=0.5
    alfa, beta_prime = log_reg.train_log_reg(all_scores_TR, LTR, 1e-06, pi) #??this pi should be fixed or not????
    new_scores_TE= numpy.dot(alfa.T,all_scores_TE)+beta_prime - numpy.log(pi/(1-pi))
    return new_scores_TE, LTE

def validate_fused_scores(D,L):
    for pi in [0.1, 0.5, 0.9]:
        fused_scores_TE, LTE = fused_scores(D,L)

        min_DCF_validation = validate.compute_min_DCF(fused_scores_TE, LTE, pi, 1, 1)
        print('Fusion: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        act_DCF_non_calibrated= validate.compute_act_DCF(fused_scores_TE, LTE, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        print('act DCF computed on theoretical threshold (pi=%f) without calibration = %f'%(pi,act_DCF_non_calibrated))

    
    
    