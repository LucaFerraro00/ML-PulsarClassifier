# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:43:38 2022

@author: lucaf
"""

import dataset_analysis as analys
import gaussian_classifiers as gauss
import validate


def main():
    D,L = analys.load('../Data/Train.txt')
    #plot(D, L)
    DTR,LTR,DTE,LTE = load_and_scale()
    
    classifier_type =1 #change
    train_evaluate_gaussian(DTR, LTR, DTE, LTE, classifier_type)
    
    

def plot(DTR, LTR):
    #save histograms of the distribution of all the features in '../Images' folder. E
    analys.plot_histograms(DTR, LTR)

    #compute correlation of pearce for the features
    analys.pearce_correlation_map(DTR, LTR)


def load_and_scale():
    #load the training set and slit Data (DTR) and labels (LTR)
    D,L = analys.load('../Data/Train.txt')
    DTR,LTR,DTE,LTE = analys.split_db_2to1(D, L)

    #compute meand and std
    mu=analys.compute_mean(DTR)
    sigma=analys.compute_variance(DTR)

    #compute Z-normalization on training and evaluation data
    scaled_DTR=analys.scale_ZNormalization(DTR, mu, sigma)
    scaled_DTE=analys.scale_ZNormalization(DTE, mu, sigma)
    #TBR: to test the Z-normalization has been perfomed correctly compute again mean and variance of scaled_DTR anc check mean= [0,0...0]; sigma 0 [1,1.....1]

    gaussianized_DTR = analys.gaussianize_training(scaled_DTR)
    gaussianized_DTE = analys.gaussianize_evaluation(scaled_DTE, scaled_DTR)
    #analys.pca(1, gaussianized_DTR)

    return gaussianized_DTR, LTR, gaussianized_DTE, LTE

def train_evaluate_gaussian(DTR,LTR,DTE,LTE,classifier_type):
    mu0,C0,mu1,C1 = gauss.train_gaussian_classifier(DTR, LTR, classifier_type)
    scores = gauss.compute_score(DTE, mu0, C0, mu1, C1)  
    min_dcf = validate.compute_min_DCF(scores, LTE, 0.1, 1, 1)
    print(min_dcf)
    
main()
