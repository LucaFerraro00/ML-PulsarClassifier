# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:43:38 2022

@author: lucaf
"""

import dataset_analysis as analys
import gaussian_classifiers as gauss
import validate
import logistic_regression as log_reg
import svm

#D,L = analys.load('../Data/Train.txt')

k=3
D,L = analys.load_pulsar_dataset('../Data/Train.txt')
gaussianize= False 

def main():
    D,L = analys.load_pulsar_dataset('../Data/Train.txt')
    D_normal= analys.scale_ZNormalization(D)
    gaussianize=False
    plot(D_normal, L, gaussianize) #plot raw features before gaussianization
    D_gaussianized= analys.gaussianize_training(D_normal)
    gaussianize= True
    plot(D_gaussianized, L, gaussianize) #plot gaussianized features
    
    """
    DTR,LTR,DTE,LTE = analys.split_db_2to1(D, L)
    DTR = analys.scale_ZNormalization(DTR)
    DTE = analys.scale_ZNormalization(DTE)
    
    
    #evaluate without gaussianization
    print("EVALUATION WITHOUT GAUSSIANIZATION")
    #train_evaluate_gaussian_models(DTR, LTR, DTE, LTE)
    #train_evaluate_log_reg(DTR, LTR, DTE, LTE)
    train_evaluate_svm(DTR, LTR, DTE, LTE)
    
    
    print("")
    print("EVALUATION WITH GAUSSIANIZATION")
    #evaluate with gaussianization
    gaussianize=True
    train_evaluate_gaussian_models(DTR, LTR, DTE, LTE)
    train_evaluate_log_reg(DTR, LTR, DTE, LTE)
    """
def plot(DTR, LTR, gaussianize):
    #save histograms of the distribution of all the features in '../Images' folder. E
    analys.plot_histograms(DTR, LTR,gaussianize)

    #compute correlation of pearce for the features
    analys.pearce_correlation_map(DTR, LTR, gaussianize)


def train_evaluate_gaussian_models(DTR,LTR,DTE,LTE):
    print("-------------------FULL GAUSSIAN CLASSIFIER-----------------")
    for pi in [0.1, 0.5, 0.9]:
        scores = gauss.compute_score_full(DTE,DTR,LTR)  
        min_dcf = validate.compute_min_DCF(scores, LTE, pi, 1, 1)
        print ("Full aussian with single fold")
        print("- pi = %f  minDCF = %f" %(pi,min_dcf))
        min_dcf_kfold = validate.kfold(D, L, k, pi, gauss.compute_score_full, gaussianize)
        print ("Gaussian with Kfold")
        print("- pi = %f  minDCF = %f" %(pi,min_dcf_kfold))
        
    print("-------------------NAIVE GAUSSIAN CLASSIFIER-----------------")
    for pi in [0.1, 0.5, 0.9]:
        scores = gauss.compute_score_naive(DTE,DTR,LTR)  
        min_dcf = validate.compute_min_DCF(scores, LTE, pi, 1, 1)
        print ("Full aussian with single fold")
        print("- pi = %f  minDCF = %f" %(pi,min_dcf))
        min_dcf_kfold = validate.kfold(D, L, k, pi, gauss.compute_score_naive, gaussianize)
        print ("Gaussian with Kfold")
        print("- pi = %f  minDCF = %f" %(pi,min_dcf_kfold))
    
    print("-------------------TIED GAUSSIAN CLASSIFIER-----------------")
    for pi in [0.1, 0.5, 0.9]:
         scores = gauss.compute_score_tied(DTE,DTR,LTR)  
         min_dcf = validate.compute_min_DCF(scores, LTE, pi, 1, 1)
         print ("Full aussian with single fold")
         print("- pi = %f  minDCF = %f" %(pi,min_dcf))
         min_dcf_kfold = validate.kfold(D, L, k, pi, gauss.compute_score_tied, gaussianize)
         print ("Gaussian with Kfold")
         print("- pi = %f  minDCF = %f" %(pi,min_dcf_kfold))
    
    
def train_evaluate_log_reg(DTR,LTR,DTE,LTE):
    print("-------------------LOGISTIC REGRESSION-----------------")
    for pi in [0.1, 0.5, 0.9]:
        scores = log_reg.compute_score(DTE,DTR,LTR)
        min_dcf = validate.compute_min_DCF(scores, LTE, pi, 1, 1)
        print ("Logistic regression with single fold")
        print("- pi = %f  minDCF = %f" %(pi,min_dcf))
        min_dcf_kfold = validate.kfold(D, L, k, pi, log_reg.compute_score, gaussianize)
        print ("Logistic regression with k-fold")
        print("- pi = %f  minDCF = %f" %(pi,min_dcf_kfold))


def train_evaluate_svm(DTR,LTR,DTE,LTE):
    print("-------------------LINEAR SVM-----------------")
    for pi in [0.1, 0.5, 0.9]:
        scores = svm.compute_score_linear(DTE, DTR, LTR)
        min_dcf = validate.compute_min_DCF(scores, LTE, pi, 1, 1)
        print ("LinearSVM with single fold")
        print("- pi = %f  minDCF = %f" %(pi,min_dcf))
        #min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_linear, gaussianize)
        #print ("Logistic regression with k-fold")
        #print("- pi = %f  minDCF = %f" %(pi,min_dcf_kfold))
    
main()
