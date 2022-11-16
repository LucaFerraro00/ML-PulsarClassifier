# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:25:14 2022

@author: lucaf
"""

import dataset_analysis as analys
import gaussian_classifiers as gauss
import validate
import logistic_regression as log_reg
import svm

def evaluation(DTR,LTR):
    print('')
    print ("##########################################################")
    print ("####Results of selected models on evaluation set: ########")
    print ("##########################################################")

    Options={}
    DEV, LEV = analys.loda_evaluation_set('../Data/Test.txt')
    DEV= analys.scale_ZNormalization(DEV)
    
    #Train models on all the training data and compute scores for the evaluation dataset
    scores_full = gauss.compute_score_full(DEV,DTR,LTR,Options) 
    scores_diag = gauss.compute_score_diag(DEV,DTR,LTR,Options) 
    scores_full_tied = gauss.compute_score_tied_full(DEV,DTR,LTR,Options) 
    scores_full_diag = gauss.compute_score_tied_diag(DEV,DTR,LTR,Options) 
    
    Options={
    'lambdaa' : 0,
    'piT': 0.1,
    }  
    scores_linear_log_reg = log_reg.compute_score(DEV, DTR, LTR, Options)
    
    Options={
    'C' : 1,
    'piT': 0.1,
    'rebalance':True
    }  
    scores_linear_svm = svm.compute_score_linear(DEV, DTR, LTR, Options)
    
    for pi in [0.1, 0.5, 0.9]:
        min_DCF = validate.compute_min_DCF(scores_full, LEV, pi, 1, 1)
        print("full covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        
        min_DCF = validate.compute_min_DCF(scores_diag, LEV, pi, 1, 1)
        print("diag covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        
        min_DCF = validate.compute_min_DCF(scores_full_tied, LEV, pi, 1, 1)
        print("full-tied covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        
        min_DCF = validate.compute_min_DCF(scores_full_diag, LEV, pi, 1, 1)
        print("full-diag covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        
        min_DCF = validate.compute_min_DCF(scores_linear_log_reg, LEV, pi, 1, 1)
        print("linear log reg -lamda = 0 -piT=0.1  gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        
        min_DCF = validate.compute_min_DCF(scores_linear_svm, LEV, pi, 1, 1)
        print("linear SVM -C=1 -piT=0.1  gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))