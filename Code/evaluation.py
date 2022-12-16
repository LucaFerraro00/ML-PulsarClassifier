# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:25:14 2022

@author: lucaf
"""

import dataset_analysis as analys
import gaussian_classifiers as gauss
import gaussian_mixture_models as gmm
import validate
import logistic_regression as log_reg
import svm
import numpy
import score_calibration as calibration
import matplotlib.pyplot as plt



def evaluation():

    DTR,LTR = analys.loda_training_set('../Data/Train.txt')
    DEV, LEV = analys.loda_evaluation_set('../Data/Test.txt')
    DTR,DEV= analys.scale_ZNormalization(DTR, DEV, normalize_ev = True) #Normalize evaluation samples using mean and covariance from training set
    DTR_gaussianized= analys.gaussianize_training(DTR)
    DEV_gaussianized = analys.gaussianize_evaluation(DEV, DTR) #Gaussianzize evaluation samples comparing them with training set
    
    print('-----------EVALUATION WITH RAW FEATURES STARTED...-----------------')
    gaussianize=False
    #evaluation_MVG(DTR, LTR, DEV, LEV)
    #evaluation_log_reg(DTR, LTR, DEV, LEV, gaussianize)
    evaluation_SVM(DTR, LTR, DEV, LEV, gaussianize)
    
    print('-----------EVALUATION ON GAUSSIANIZED FEATURES STARTED...-----------------')
    gaussianize=True
    #evaluation_MVG(DTR_gaussianized, LTR, DEV_gaussianized, LEV)
    #evaluation_log_reg(DTR_gaussianized, LTR, DEV_gaussianized, LEV, gaussianize)
    #evaluation_SVM(DTR_gaussianized, LTR, DEV_gaussianized, LEV,gaussianize )
    
    # evaluation_gmm(DTR, DTR_gaussianized, LTR, DEV, DEV_gaussianized, LEV)
    
    # calibration.min_vs_act(DTR,LTR, DEV=DEV, LEV=LEV, evaluation=True)
    # calibration.optimal_threshold(DTR, LTR, DEV=DEV, LEV=LEV, evaluation=True)
    # calibration.validate_score_trasformation(DTR,LTR, DEV=DEV, LEV=LEV, evaluation=True)
    
    # evaluation_fusion(DTR, LTR, DEV, LEV)


def evaluation_MVG(DTR, LTR, DEV, LEV):
    DTR_copy = DTR
    DEV_copy = DEV
    Options={ }  
    m = 8
    while m>=5:
        if m < 8:
            DTR, P = analys.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### Gaussian classifiers with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Gaussian classifiers with NO PCA ####")
            print ("##########################################")
            
        #Train models on all the training data and compute scores for the evaluation dataset
        scores_full = gauss.compute_score_full(DEV,DTR,LTR,Options) 
        scores_diag = gauss.compute_score_diag(DEV,DTR,LTR,Options) 
        scores_full_tied = gauss.compute_score_tied_full(DEV,DTR,LTR,Options) 
        scores_tied_diag = gauss.compute_score_tied_diag(DEV,DTR,LTR,Options) 
        for pi in [0.1, 0.5, 0.9]:
            #compute min DCF on evaluation set
            min_DCF = validate.compute_min_DCF(scores_full, LEV, pi, 1, 1)
            print("full covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_diag, LEV, pi, 1, 1)
            print("diag covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_full_tied, LEV, pi, 1, 1)
            print("full-tied covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_tied_diag, LEV, pi, 1, 1)
            print("tied-diag covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        
        m=m-1
        DEV = DEV_copy
        DTR = DTR_copy
        

def evaluation_log_reg(DTR, LTR, DEV, LEV, gaussianize):
    log_reg.plot_minDCF_wrt_lamda(DTR,LTR, gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    log_reg.quadratic_plot_minDCF_wrt_lamda(DTR, LTR, gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    DTR_copy = DTR
    DEV_copy = DEV
    m = 8
    while m>=5:
        if m < 8:
            DTR, P = analys.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### Logistic Regression with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Logistic Regression with NO PCA ####")
            print ("##########################################")
            
        Options={
        'lambdaa' : 1e-6,
        'piT': None,
        }              
        for piT in [0.1, 0.5, 0.9]:
            Options['piT']=piT
            scores_linear_log_reg = log_reg.compute_score(DEV, DTR, LTR, Options)
            scores_quadratic_log_reg = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
            for pi in [0.1, 0.5, 0.9]:
                min_DCF = validate.compute_min_DCF(scores_linear_log_reg, LEV, pi, 1, 1)
                print("linear log reg -lamda = 10^-6 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
                min_DCF = validate.compute_min_DCF(scores_quadratic_log_reg, LEV, pi, 1, 1)
                print("Quadratic log reg -lamda = 10^-6 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        m=m-1
        DEV = DEV_copy
        DTR = DTR_copy
        
def evaluation_SVM(DTR, LTR, DEV, LEV, gaussianize):
    svm.plot_linear_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    # svm.plot_quadratic_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    #svm.plot_RBF_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    Options={
        'C' : None,
        'piT': None,
        'gamma':None,
        'rebalance':None
        }  
    DTR_copy = DTR
    DEV_copy = DEV
    m = 8
    while m>=5:
        if m < 8:
            DTR, P = analys.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
        #     print ("##########################################")
        #     print ("##### SVM with m = %d ####" %m)
        #     print ("##########################################")
        # else:
        #     print ("##########################################")
        #     print ("#### SVM with NO PCA ####")
        #     print ("##########################################")
            
        # for piT in [0.1, 0.5, 0.9]:
        #     Options['C']=1
        #     Options['piT']=piT
        #     Options['rebalance']=True
        #     scores_linear_svm = svm.compute_score_linear(DEV, DTR, LTR, Options)
        #     for pi in [0.1, 0.5, 0.9]:
        #         min_DCF = validate.compute_min_DCF(scores_linear_svm, LEV, pi, 1, 1)
        #         print("linear SVM -C =1 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        # Options['C']=1
        # Options['rebalance']=False
        # scores_linear_svm = svm.compute_score_linear(DEV, DTR, LTR, Options)
        # for pi in [0.1, 0.5, 0.9]:
        #     min_DCF = validate.compute_min_DCF(scores_linear_svm, LEV, pi, 1, 1)
        #     print("linear SVM -C =1 -No rebalancing - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
        if m < 8:
            DTR, P = analys.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### SVM Quadratci with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### SVM Quadratic with NO PCA ####")
            print ("##########################################")
            
        # for piT in [0.1, 0.5, 0.9]:
        #     Options['C']=0.1
        #     Options['piT']=piT
        #     Options['rebalance']=True
        #     scores_quadratic_svm = svm.compute_score_quadratic(DEV, DTR, LTR, Options)
        #     for pi in [0.1, 0.5, 0.9]:
        #         min_DCF = validate.compute_min_DCF(scores_quadratic_svm, LEV, pi, 1, 1)
        #         print("linear SVM -C =1 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        Options['C']=0.1
        Options['rebalance']=False
        scores_quadratic_svm = svm.compute_score_quadratic(DEV, DTR, LTR, Options)
        for pi in [0.1, 0.5, 0.9]:
            min_DCF = validate.compute_min_DCF(scores_quadratic_svm, LEV, pi, 1, 1)
            print("linear SVM -C =1 -No rebalancing - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
        if m < 8:
            DTR, P = analys.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### SVM RBF with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### SVM RBF with NO PCA ####")
            print ("##########################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['C']=10
            Options['piT']=piT
            Options['gamma']=0.01
            Options['rebalance']=True
            scores_rbf_svm = svm.compute_scoreRBF(DEV, DTR, LTR, Options)
            for pi in [0.1, 0.5, 0.9]:
                min_DCF = validate.compute_min_DCF(scores_rbf_svm, LEV, pi, 1, 1)
                print("linear SVM -C =1 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        Options['C']=10
        Options['rebalance']=False
        scores_rbf_svm = svm.compute_scoreRBF (DEV, DTR, LTR, Options)
        for pi in [0.1, 0.5, 0.9]:
            min_DCF = validate.compute_min_DCF(scores_rbf_svm, LEV, pi, 1, 1)
            print("linear SVM -C =1 -No rebalancing - pi = %f --> min_DCF= %f" %(pi,min_DCF))    
        
        m=m-1
        DEV = DEV_copy
        DTR = DTR_copy


def evaluation_gmm(DTR, DTR_gaussianized, LTR, DEV, DEV_gaussianized, LEV):
    gmm.plot_minDCF_wrt_components(DTR, DTR_gaussianized, LTR, DEV=DEV, DEV_gaussianized=DEV_gaussianized, LEV=LEV, evaluation=True  )


def evaluation_fusion(DTR, LTR, DEV, LEV):
    fused_scores, LTE = fusion_on_evaluation(DTR, LTR, DEV, LEV)
    for pi in [0.1, 0.5, 0.9]:
        min_DCF = validate.compute_min_DCF(fused_scores, LTE, pi, 1, 1)
        print("Fusion - pi = %f --> min_DCF= %f" %(pi,min_DCF))
    
    ROC_with_fusion_evaluation(DTR, LTR, DEV, LEV)
    
    

def fusion_on_evaluation(DTR, LTR, DEV, LEV):
    Options={
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }   
    scores1 = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
    scores1_TR, _, scores1_TE, LTE = calibration.split_scores(scores1, LEV) #split and shuffle scores
    scores1_TR = analys.mrow(scores1_TR)
    scores1_TE = analys.mrow(scores1_TE)
 
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    scores2 = svm.compute_score_RBF(DEV, DTR, LTR, Options)
    scores2_TR, LTR, scores2_TE, LTE = calibration.split_scores(scores2, LEV) #same split used for best model 1 above. Labels returned are the same of best model 1
    scores2_TR = analys.mrow(scores2_TR)
    scores2_TE = analys.mrow(scores2_TE)
    
    all_scores_TR = numpy.vstack((scores1_TR, scores2_TR))
    all_scores_TE = numpy.vstack((scores1_TE, scores2_TE))
    pi=0.5
    alfa, beta_prime = log_reg.train_log_reg(all_scores_TR, LTR, 1e-06, pi) #??this pi should be fixed or not????
    new_scores_TE= numpy.dot(alfa.T,all_scores_TE)+beta_prime - numpy.log(pi/(1-pi))
    return new_scores_TE, LTE


def ROC_with_fusion_evaluation(DTR, LTR, DEV, LEV):
    Options={
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }    
    scores1 = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
    scores1_TR, LTR1, scores1_TE, LTE = calibration.split_scores(scores1, LEV) #split and shuffle scores
    calibrated_scores = calibration.score_trasformation(scores1_TR, LTR1, scores1_TE, 0.5)
    FPR1, TPR1 =validate.ROC (calibrated_scores, LTE)
   
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    scores2 = svm.compute_score_RBF(DEV, DTR, LTR, Options)
    scores2_TR, LTR2, scores2_TE, _ = calibration.split_scores(scores2, LEV) #same split used for best model 1 above. Labels returned are the same of best model 1
    calibrated_scores = calibration.score_trasformation(scores2_TR, LTR2, scores2_TE, 0.5)
    FPR2, TPR2 =validate.ROC (calibrated_scores, LTE)
    
    fused_scores_TE, _ = fusion_on_evaluation(DTR, LTR, DEV, LEV)
    FPR3, TPR3 =validate.ROC (fused_scores_TE, LTE)

    plt.figure()
    plt.plot(FPR1,TPR1, 'r', label = 'Quadratic Log Reg', )
    plt.plot(FPR2,TPR2, 'b', label = 'RBF SVM')
    plt.plot(FPR3,TPR3, 'g', label = 'Fusion')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Negative Rate')
    plt.legend()
    plt.savefig("../Images/fusion_ROC_evaluaion.pdf" )
    plt.show()
    

   
    
    
   