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
k=5
  
def main():
    D,L = analys.load_pulsar_dataset('../Data/Train.txt')
    D= analys.scale_ZNormalization(D)
    gaussianize= False 
    plot(D, L, gaussianize) #plot raw features before gaussianization
    D_gaussianized= analys.gaussianize_training(D)
    gaussianize= True
    plot(D_gaussianized, L, gaussianize) #plot gaussianized features    
    
    #evaluate without gaussianization
    print("EVALUATION WITHOUT GAUSSIANIZATION")
    gaussianize=False
    #train_evaluate_gaussian_models(D, L)
    
    #log_reg.plot_minDCF_wrt_lamda(D, L, gaussianize)
    #train_evaluate_log_reg(D, L)
    
    #svm.plot_linear_minDCF_wrt_C(D, L, gaussianize)
    svm.plot_quadratic_minDCF_wrt_C(D, L, gaussianize)
    svm.plot_RBF_minDCF_wrt_C(D, L, gaussianize)
    train_evaluate_svm(D,L)
    
    #evaluate with gaussianization
    print('\n')
    print("EVALUATION WITH GAUSSIANIZATION")
    gaussianize=True
    #train_evaluate_gaussian_models(D_gaussianized, L)
    
    #log_reg.plot_minDCF_wrt_lamda(D_gaussianized, L, gaussianize)
    #train_evaluate_log_reg(D_gaussianized, L)
    
    #svm.plot_linear_minDCF_wrt_C(D_gaussianized, L, gaussianize)
    #svm.plot_quadratic_minDCF_wrt_C(D_gaussianized, L, gaussianize)
    #svm.plot_RBF_minDCF_wrt_C(D_gaussianized, L, gaussianize)
    #train_evaluate_svm(D_gaussianized,L)
    
    
    
def plot(DTR, LTR, gaussianize):
    #save histograms of the distribution of all the features in '../Images' folder. E
    analys.plot_histograms(DTR, LTR,gaussianize)

    #compute correlation of pearce for the features
    analys.pearce_correlation_map(DTR, LTR, gaussianize)


def train_evaluate_gaussian_models(D,L):
    Options={ }  
    m = 8
    while m>=5:
        if m < 8:
            D = analys.pca(m, D)
            print ("##########################################")
            print ("##### Gaussian classifiers with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Gaussian classifiers with NO PCA ####")
            print ("##########################################")
        
        for pi in [0.1, 0.5, 0.9]:
            min_dcf_full = validate.kfold(D, L, k, pi, gauss.compute_score_full, Options)
            print(" Full-Cov - pi = %f -> minDCF = %f" %(pi,min_dcf_full))
            min_dcf_diag = validate.kfold(D, L, k, pi, gauss.compute_score_diag, Options)
            print(" Diag-cov - pi = %f -> minDCF = %f" %(pi,min_dcf_diag))
            min_dcf_tied_full = validate.kfold(D, L, k, pi, gauss.compute_score_tied_full, Options)
            print(" Tied full-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_full))
            min_dcf_tied_diag = validate.kfold(D, L, k, pi, gauss.compute_score_tied_diag, Options)
            print(" Tied diag-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_diag))

        m=m-1

        
def train_evaluate_log_reg(D,L):
    Options={
    'lambdaa' : None,
    'piT': None,
    }  
    m = 8
    while m>=5:
        if m < 8:
            D = analys.pca(m, D)
            print ("##########################################")
            print ("### Logistic regression with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Logistic regression with NO PCA ####")
            print ("##########################################")
            
        print("-------------------LOGISTIC REGRESSION-----------------")
        for piT in [0.1, 0.5, 0.9]:
            Options['lambdaa']=0
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, log_reg.compute_score, Options)
                print(" Logistic reggression -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))    
            
        m = m-1
    

def train_evaluate_svm(D,L):
    Options={
        'C' : None,
        'piT': None,
        'gamma':None
        }  
    m = 8
    while m>=5:
        if m < 8:
            D = analys.pca(m, D)
            print ("##########################################")
            print ("############# SVM LINEAR with m = %d ##############" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("##############SVM LINEAR with NO PCA ##############")
            print ("##########################################")
        print("-------------------LINEAR SVM-----------------")
        for piT in [0.1, 0.5, 0.9]:
            Options['C']=1
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_linear, Options)
                print(" SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))      
                
        if m < 8:
            D = analys.pca(m, D)
            print ("##########################################")
            print ("############# SVM QUADRATIC with m = %d ##############" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("##############SVM QUADRATIC with NO PCA ##############")
            print ("##########################################")
        print("-------------------LINEAR SVM-----------------")
        for piT in [0.1, 0.5, 0.9]:
            Options['C']=1
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_quadratic, Options)
                print(" SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))      
           
        if m < 8:
            D = analys.pca(m, D)
            print ("##########################################")
            print ("############# SVM RBF with m = %d ##############" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("##############SVM RBF with NO PCA ##############")
            print ("##########################################")
        print("-------------------LINEAR SVM-----------------")
        for piT in [0.1, 0.5, 0.9]:
            Options['C']=1
            Options['piT']=piT
            Options['gamma']=0.1
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_RBF, Options)
                print(" SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))      
                
        m = m-1
    
main()
