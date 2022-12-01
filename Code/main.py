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
import evaluation
import gaussian_mixture_models as gmm
import score_calibration 
import fusion

k=5 #kfold
  
def main():
    D,L = analys.loda_training_set('../Data/Train.txt')
    D= analys.scale_ZNormalization(D)[0]
    # D_gaussianized= analys.gaussianize_training(D)

    # gaussianize= False 
    # plot(D, L, gaussianize) #plot raw features before gaussianization
    # #D_gaussianized= analys.gaussianize_training(D)
    # gaussianize= True
    # #plot(D_gaussianized, L, gaussianize) #plot gaussianized features    
    
    # #evaluate models on raw data
    print("VALIDATION WITHOUT GAUSSIANIZATION")
    gaussianize=False
    # #train_evaluate_gaussian_models(D, L)
    
    # log_reg.plot_minDCF_wrt_lamda(D, L, gaussianize)
    # log_reg.quadratic_plot_minDCF_wrt_lamda(D, L, gaussianize)
    # train_evaluate_log_reg(D, L)
    
    # svm.plot_linear_minDCF_wrt_C(D, L, gaussianize)
    # svm.plot_quadratic_minDCF_wrt_C(D, L, gaussianize)
    # svm.plot_RBF_minDCF_wrt_C(D, L, gaussianize)
    #train_evaluate_svm(D,L)
    
    # gmm.plot_minDCF_wrt_components(D, D_gaussianized, L)
    # train_evaluate_gmm(D, L)
    
    # evaluate models on gaussianized data  
    # print('\n')
    print("VALIDATION WITH GAUSSIANIZATION")
    gaussianize=True
    # train_evaluate_gaussian_models(D_gaussianized, L)
    
    # log_reg.plot_minDCF_wrt_lamda(D_gaussianized, L, gaussianize)
    # log_reg.quadratic_plot_minDCF_wrt_lamda(D_gaussianized, L, gaussianize)
    # train_evaluate_log_reg(D_gaussianized, L)
    
    # svm.plot_linear_minDCF_wrt_C(D_gaussianized, L, gaussianize)
    # svm.plot_quadratic_minDCF_wrt_C(D_gaussianized, L, gaussianize)
    # svm.plot_RBF_minDCF_wrt_C(D_gaussianized, L, gaussianize)
    # train_evaluate_svm(D_gaussianized,L)
    
    # train_evaluate_gmm(D_gaussianized, L)
    
    # validate.two_bests_roc(D, L) #model selection
    
    # calibration(D, L) #score calibration
    
    # validate_fusion(D,L)
    
    # evaluation.evaluation()
    
    

    
def plot(DTR, LTR, gaussianize):
    #save histograms of the distribution of all the features in '../Images' folder. E
    analys.plot_histograms(DTR, LTR,gaussianize)

    #compute correlation of pearce for the features
    analys.pearce_correlation_map(DTR, LTR, gaussianize)


def train_evaluate_gaussian_models(D,L):
    Options={ }  
    m = 5
    while m>=5:
        if m < 8:
            D = analys.pca(m, D)[0]
            print ("##########################################")
            print ("##### Gaussian classifiers with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Gaussian classifiers with NO PCA ####")
            print ("##########################################")
        
        for pi in [0.1, 0.5, 0.9]:
            min_dcf_full = validate.kfold(D, L, k, pi, gauss.compute_score_full, Options)[0]
            print(" Full-Cov - pi = %f -> minDCF = %f" %(pi,min_dcf_full))
            min_dcf_diag = validate.kfold(D, L, k, pi, gauss.compute_score_diag, Options)[0]
            print(" Diag-cov - pi = %f -> minDCF = %f" %(pi,min_dcf_diag))
            min_dcf_tied_full = validate.kfold(D, L, k, pi, gauss.compute_score_tied_full, Options)[0]
            print(" Tied full-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_full))
            min_dcf_tied_diag = validate.kfold(D, L, k, pi, gauss.compute_score_tied_diag, Options)[0]
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
            D = analys.pca(m, D)[0]
            print ("############################################")
            print ("### Linear Logistic regression with m = %d ##" %m)
            print ("###########################################")
        else:
            print ("################################################")
            print ("#### Linear Logistic regression with NO PCA ####")
            print ("###############################################")
            
        # for piT in [0.1, 0.5, 0.9]:
        #     Options['lambdaa']=1e-06
        #     Options['piT']=piT
        #     for pi in [0.1, 0.5, 0.9]:
                # min_dcf_kfold = validate.kfold(D, L, k, pi, log_reg.compute_score, Options)[0]
                # print(" Logistic reggression -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))
             
        if m < 8:
            D = analys.pca(m, D)[0]
            print ("################################################")
            print ("### Quadratic Logistic regression with m = %d ##" %m)
            print ("###############################################")
        else:
            print ("###################################################")
            print ("#### Quadratic Logistic regression with NO PCA ####")
            print ("##################################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['lambdaa']=1e-06
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, log_reg.compute_score_quadratic, Options)[0]
                print(" Quadratic Log Reg -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))  
        
        
        m = m-1
    

def train_evaluate_svm(D,L):
    D_copy = D
    Options={
        'C' : None,
        'piT': None,
        'gamma':None,
        'rebalance':None
        }  
    m = 7
    while m>=5:
        
        # if m < 8:
        #     D = analys.pca(m, D)[0]
        #     print ("##########################################")
        #     print ("############ SVM LINEAR with m = %d #######" %m)
        #     print ("##########################################")
        # else:
        #     print ("##########################################")
        #     print ("######## SVM LINEAR with NO PCA ##########")
        #     print ("##########################################")
            
        # Options['C']=1
        # for piT in [0.1, 0.5, 0.9]:
        #     for pi in [0.1, 0.5, 0.9]:
        #         Options['piT']=piT
        #         Options['rebalance']=True
        #         min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_linear, Options)[0]
        #         print("Linear SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))
                
        # Options['rebalance']=False
        # for pi in [0.1, 0.5, 0.9]:
        #         min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_linear, Options)[0]
        #         print("Linear SVM without rebalancing -C=%f - pi = %f - minDCF = %f" %(Options['C'], pi,min_dcf_kfold)) 
          
        if m < 8:
            D = analys.pca(m, D)[0]
            print ("##########################################")
            print ("########## SVM QUADRATIC with m = %d ######" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("########SVM QUADRATIC with NO PCA ########")
            print ("##########################################")
        # for piT in [0.5, 0.9]:
        #     for pi in [0.1, 0.5, 0.9]:
        #         Options['C']=0.1
        #         Options['piT']=piT
        #         Options['rebalance']=True
                #min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_quadratic, Options)[0]
                # print("Quadratric SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))
                
        Options['rebalance']=False
        Options['C']=0.1
        for pi in [0.1]:
            min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_quadratic, Options)[0]
            print("Quadratic SVM without rebalancing -C=%f - pi = %f - minDCF = %f" %(Options['C'], pi,min_dcf_kfold))
           
        # if m < 8:
        #     D = analys.pca(m, D)[0]
        #     print ("##########################################")
        #     print ("######### SVM RBF with m = %d #############" %m)
        #     print ("##########################################")
        # else:
        #     print ("##########################################")
        #     print ("##########SVM RBF with NO PCA ############")
        #     print ("##########################################")
        # for piT in [0.1, 0.5, 0.9]:
        #     for pi in [0.1, 0.5, 0.9]:
        #         Options['C']=10
        #         Options['piT']=piT
        #         Options['gamma']=0.01
        #         Options['rebalance']=True
        #         min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_RBF, Options)[0]
        #         print("RBF SVM -piT = %f -gamma =%f -C=%f - pi = %f -> minDCF = %f" %(piT, Options['gamma'], Options['C'], pi,min_dcf_kfold))      
            
        # Options['rebalance']=False
        # for pi in [0.1, 0.5, 0.9]:
        #     pass
        #     min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_RBF, Options)[0]
        #     print("RBF SVM without rebalancing -gamma =%f -C=%f - pi = %f -> minDCF = %f" %(Options['gamma'], Options['C'], pi,min_dcf_kfold))
              
        m = m-1
        D=D_copy
        
def train_evaluate_gmm(D,L):
    Options={ 
        'Type':None,
        'iterations':None #components will be 2^iterations
        }  
    m = 8
    while m>=5:
        if m < 8:
            D = analys.pca(m, D)[0]
            print ("##########################################")
            print ("############# GMM with m = %d ##############" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("########## GMM LINEAR with NO PCA ########")
            print ("##########################################")
            for pi in [0.1, 0.5, 0.9]:
                Options['Type']='full'
                Options['iterations']= 4
                min_dcf_kfold = validate.kfold(D, L, k, pi, gmm.compute_score, Options)[0]
                print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
                
                Options['Type']='diag'
                Options['iterations']= 5
                min_dcf_kfold = validate.kfold(D, L, k, pi, gmm.compute_score, Options)[0]
                print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
                
                Options['Type']='full-tied'
                Options['iterations']= 6
                min_dcf_kfold = validate.kfold(D, L, k, pi, gmm.compute_score, Options)[0]
                print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
                
                Options['Type']='diag-tied'
                Options['iterations']= 3
                min_dcf_kfold = validate.kfold(D, L, k, pi, gmm.compute_score, Options)[0]
                print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
            
            
def calibration(D,L):
    score_calibration.min_vs_act(D, L)
    score_calibration.optimal_threshold(D,L)
    score_calibration.validate_score_trasformation(D, L)
    score_calibration.min_vs_act_after_calibration(D, L)
        
def validate_fusion(D,L):
    fusion.validate_fused_scores(D,L)
    fusion.ROC_with_fusion(D,L)
    fusion.bayes_plot_with_fusion(D,L)

main()
