# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:57:08 2022

@author: lucaf
"""
import numpy
import scipy
import validate
import matplotlib.pyplot as plt


def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))
"""
----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------LINEAR SVM-------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""
#we are considering a binary problem, we're supposing that LTR is labels 0 or 1
#C and K are hyperparameters
def train_SVM_linear(DTR, LTR, C, piT, rebalance, K=1):
    
    #first we simulate the bias by extending our data with ones, this allow us to simulate the effect of a bias
    DTREXT=numpy.vstack([DTR, numpy.ones((1,DTR.shape[1]))])
    #DTREXt is the original data matrix  wich contains the original training data x1...xn. Each x has an added feature equals to K=1 appendend
    
    #compute Z to be used to do compute: zi(wTx+b)
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    #compute matrix H which is inside the dual formula
    #Hij=zizjxiTxj (xi and xj are vectors)
    G=numpy.dot(DTREXT.T,DTREXT)
    H=mcol(Z) * mrow(Z) * G #mcol(Z) * mrow(Z) result in a matrix of the same dimension of G?
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad
        
    if rebalance:
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        pi_emp_F = (DTR0.shape[1] / DTR.shape[1])
        pi_emp_T = (DTR1.shape[1] / DTR.shape[1])
        CT = C * piT/pi_emp_T 
        CF = C * (1-piT)/pi_emp_F
        
        #bounds: a list of DTR.shape[1] elements. Each element is a tuple (0,CT) or (0,CF)
        bounds = [(0,1)]*DTR.shape[1] #Initialize the array with random values 
        for i in range(DTR.shape[1]):
            if LTR[i]==0:
                bounds[i]=(0,CF)
            else:
                bounds[i]=(0,CT)
    else:#no re-balancing
        bounds = [(0,C)]*DTR.shape[1]
        
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, #function to minimze
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = bounds,
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation  
                                                      )
    
    wStar = numpy.dot(DTREXT, mcol(alphaStar)*mcol(Z))

    return wStar, alphaStar

K=1
K_fold=5
def compute_score_linear(DTE,DTR,LTR, Options):
    if Options['C']== None:
        Options['C'] = 0
    if Options['piT'] == None:
        Options['piT']=0.5
    if Options['rebalance'] == None:
        Options['rebalance']=True

    w,_alfa= train_SVM_linear(DTR, LTR, Options['C'], Options['piT'], Options['rebalance'])
    DTE_EXT=numpy.vstack([DTE, numpy.ones((1,DTE.shape[1]))])
    score = numpy.dot(w.T,DTE_EXT)
    return score.ravel()
    
def plot_linear_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=None, LEV=None, evaluation=False):
    # print('Linear SVM: computation for plotting min_cdf wrt C started...')
    # min_DCFs=[]
    # for pi in [0.1, 0.5, 0.9]:
    #     C_array = numpy.logspace(-4,3, num = 8)
    #     for C in C_array:
    #             Options= {'C': C,
    #                       'piT':0.5,
    #                       'rebalance':True}
    #             min_dcf_kfold = validate.kfold(DTR, LTR, K_fold, pi, compute_score_linear, Options )[0] 
    #             min_DCFs.append(min_dcf_kfold)
    #             print ("Linear SVM min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_dcf_kfold))
    # min_DCFs_p0 = min_DCFs[0:8] #min_DCF results with prior = 0.1
    # min_DCFs_p1 = min_DCFs[8:16] #min_DCF results with prior = 0.5
    # min_DCFs_p2 = min_DCFs[16:24] #min_DCF results with prior = 0.9
    
    C_array = numpy.logspace(-4,3, num = 8)
    min_DCFs_p0 = [0.28179464765598977, 0.2623780620508301, 0.24202992041343174, 0.25202498457927575, 0.25188159889764367, 0.24912315296997736, 0.2441076601019872, 0.3292985945197211]
    min_DCFs_p1 = [0.156195, 0.145005, 0.125764, 0.119765, 0.120497, 0.119550, 0.118670, 0.134182]
    min_DCFs_p2=[0.5994239923623829, 0.5558852457636595, 0.5501256202858575, 0.521894873345654, 0.5160083951264499, 0.5177359370810403, 0.5186231547735709, 0.5720969960549399]

    
   
    if evaluation==False: #plot only result for validation set
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, label='prior=0.1')
        plt.plot(C_array, min_DCFs_p1, label='prior=0.5')
        plt.plot(C_array, min_DCFs_p2, label='prior=0.9')
        plt.legend()
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("../Images/min_DCF_C_linearSVM_gaussianized.pdf")
        else:
            plt.savefig("../Images/min_DCF_C_linearSVM_raw.pdf")
        plt.show()
    
    else: #compare plot of validation and evaluation set
        # min_DCFs=[]
        # for pi in [0.1, 0.5, 0.9]:
        #     C_array = numpy.logspace(-4,3, num = 8)
        #     for C in C_array:
        #             Options= {'C': C,
        #                       'piT':0.5,
        #                       'rebalance':True}
        #             scores_linear_svm = compute_score_linear(DEV, DTR, LTR, Options)
        #             min_DCF_ev = validate.compute_min_DCF(scores_linear_svm, LEV, pi, 1, 1)
        #             min_DCFs.append(min_DCF_ev)
        #             print ("computed min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_DCF_ev))
        # min_DCFs_p0_ev = min_DCFs[0:8] #min_DCF results with prior = 0.1
        # min_DCFs_p1_ev = min_DCFs[8:16] #min_DCF results with prior = 0.5
        # min_DCFs_p2_ev = min_DCFs[16:24] #min_DCF results with prior = 0.9
        
        C_array = numpy.logspace(-4,3, num = 8)
        min_DCFs_p0_ev = [0.2865733545826198, 0.2502143976214237, 0.23684615474603893, 0.2310877900892056, 0.22631704931280272, 0.2288803719764986, 0.22998453097539442, 0.22694786875715972]
        min_DCFs_p1_ev = [0.14878400028316385, 0.13566547551877625, 0.11879158031519375, 0.11183336887879418, 0.11441408932079372, 0.11442278820994559, 0.11490917609821226, 0.11490482665363633]
        min_DCFs_p2_ev = [0.5343322657696613, 0.5132305604574297, 0.4887328388164832, 0.47924175082841924, 0.4819747018305763, 0.4779261188346249, 0.47608585383646507, 0.48087054283168046]
        
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, '--b',  label='prior=0.1-val')
        plt.plot(C_array, min_DCFs_p1, '--r', label='prior=0.5-val')
        plt.plot(C_array, min_DCFs_p2, '--g', label='prior=0.9-val')
        plt.plot(C_array, min_DCFs_p0_ev, 'b', label='prior=0.1-eval')
        plt.plot(C_array, min_DCFs_p1_ev, 'r', label='prior=0.5-eval')
        plt.plot(C_array, min_DCFs_p2_ev, 'g', label='prior=0.9-eval')
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        #plt.legend()
        plt.show()
        if gaussianize:
            plt.savefig("../Images/min_DCF_C_linearSVM_evaluation_gaussianized.pdf")
        else:
            plt.savefig("../Images/min_DCF_C_linearSVM_evaluation_raw.pdf")
        
    #return min_DCFs

"""
----------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------RBF SVM ------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""

def train_SVM_RBF(DTR, LTR, C, piT, gamma, rebalance, K=1):
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    Dist= mcol((DTR**2).sum(0)) +mrow((DTR**2).sum(0)) - 2 * numpy.dot(DTR.T,DTR)
    kernel= numpy.exp(-gamma* Dist) + K #K account for byas term
    H=mcol(Z) *mrow(Z) * kernel
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad
    
    
    if rebalance:
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        pi_emp_F = (DTR0.shape[1] / DTR.shape[1])
        pi_emp_T = (DTR1.shape[1] / DTR.shape[1])
        CT = C * piT/pi_emp_T 
        CF = C * (1-piT)/pi_emp_F
        
        #bounds: a list of DTR.shape[1] elements. Each element is a tuple (0,CT) or (0,CF)
        bounds = [(0,1)]*DTR.shape[1] #Initialize the array with random values 
        for i in range(DTR.shape[1]):
            if LTR[i]==0:
                bounds[i]=(0,CF)
            else:
                bounds[i]=(0,CT)
    else:#no re-balancing
        bounds = [(0,C)]*DTR.shape[1]
        
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, #function to minimze
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = bounds,
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation  
                                                      )
    
    return alphaStar,Z



def compute_score_RBF(DTE,DTR,LTR, Options, K=1):
    if Options['C']== None:
        Options['C'] = 0
    if Options['piT'] == None:
        Options['piT']=0.5
    if Options['gamma']==None:
        Options['gamma']=0.1
    if Options['rebalance']==None:
        Options['rebalance']=True
    Dist= mcol((DTR**2).sum(0)) +mrow((DTE**2).sum(0)) - 2 * numpy.dot(DTR.T,DTE)
    kernel= numpy.exp(-Options['gamma']* Dist) + (K**2) #K account for byas term
    alphaStar, Z= train_SVM_RBF(DTR, LTR, Options['C'], Options['piT'] ,Options['gamma'],Options['rebalance'], K)
    score= numpy.dot(alphaStar * Z, kernel)
    return score.ravel()
        
   
    
def plot_RBF_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=None, LEV=None, evaluation=False):
    print('RBF SVM: computation for plotting min_cdf wrt C started...')
    min_DCFs=[]
    # pi=0.5
    # gamma_array= [0.0001, 0.001, 0.01, 0.1]    
    # for gamma in gamma_array:
    #     C_array = numpy.logspace(-4,3, num = 8)
    #     for C in C_array:
    #             Options= {'C': C,
    #                       'piT':0.5,
    #                       'gamma':gamma,
    #                       'rebalance':True}
    #             min_dcf_kfold = validate.kfold(DTR, LTR, K_fold, pi, compute_score_RBF, Options )[0] 
    #             min_DCFs.append(min_dcf_kfold)
    #             print ("computed min_dcf for pi=%f -gamma=%f -C=%f - results min_dcf=%f "%(pi, gamma, C,min_dcf_kfold))
    #     print (min_DCFs)
    # min_DCFs_g0 = min_DCFs[0:8] #min_DCF results with gamma = 0.0001
    # min_DCFs_g1 = min_DCFs[8:16] #min_DCF results with gamma = 0.001
    # min_DCFs_g2 = min_DCFs[16:24] #min_DCF results with gamma = 0.01
    # min_DCFs_g3 = min_DCFs[24:32] #min_DCF results with gamma = 0.1
    
    C_array = numpy.logspace(-4,3, num = 8)
    min_DCFs_g0=[0.16331553988166023, 0.16331553988166023, 0.16331553988166023, 0.16354639984705527, 0.1549246218165072, 0.13891502429440417, 0.12392851517363193, 0.17740671513924344]
    min_DCFs_g1=[ 0.16319214402776092, 0.16319214402776092, 0.16329960813925665, 0.15119494080005003, 0.14014898283339727, 0.12235623255774807, 0.12072833914290833, 0.14109241252271026]
    min_DCFs_g2= [ 0.16417931085895537, 0.16479629012845193, 0.15281501864314462, 0.14274029576528274, 0.12436242970494413, 0.12016697067236762, 0.11735667143216144, 0.11236116839190403]
    min_DCFs_g3= [0.164004, 0.161675, 0.147151, 0.13328661024224667, 0.11643323157058123, 0.11347173107699782, 0.1119909808302061, 0.13086243030614197]



    if evaluation == False:
        plt.figure()
        plt.plot(C_array, min_DCFs_g0, label='gamma=0.0001')
        plt.plot(C_array, min_DCFs_g1, label='gamma=0.001')
        plt.plot(C_array, min_DCFs_g2, label='gamma=0.01')
        plt.plot(C_array, min_DCFs_g3, label='gamma=0.1')
        plt.legend()
        plt.tight_layout() # TBR: Use with non-default font size to keep axis label inside the figure
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("../Images/min_DCF_C_RBF_SVM_gaussianized.pdf")
        else:
            plt.savefig("../Images/min_DCF_C_RBF_SVM_raw.pdf")
        plt.show()
        
    else:#compare validation and evaluation
        min_DCFs=[]
        # pi=0.5
        # gamma_array= [0.0001, 0.001, 0.01, 0.1]
        # for gamma in gamma_array:
        #     C_array = numpy.logspace(-4,3, num = 8)
        #     for C in C_array:
        #             Options= {'C': C,
        #                       'piT':0.5,
        #                       'gamma':gamma,
        #                       'rebalance':True}
        #             scores_rbf_svm = compute_score_RBF(DEV, DTR, LTR, Options)
        #             min_DCF_ev =validate.compute_min_DCF(scores_rbf_svm, LEV, pi, 1, 1)
        #             min_DCFs.append(min_DCF_ev)
        #             print ("computed min_dcf for pi=%f -gamma=%f -C=%f - results min_dcf=%f "%(pi, gamma, C,min_DCF_ev))
        # min_DCFs_g0_ev = min_DCFs[0:8] #min_DCF results with gamma = 0.0001
        # min_DCFs_g1_ev = min_DCFs[8:16] #min_DCF results with gamma = 0.001
        # min_DCFs_g2_ev = min_DCFs[16:24] #min_DCF results with gamma = 0.01
        # min_DCFs_g3_ev = min_DCFs[24:32] #min_DCF results with gamma = 0.1
        
        min_DCFs_g0_ev = [0.162906, 0.162906, 0.162906, 0.159725, 0.145717, 0.131669, 0.116833, 0.111347]
        min_DCFs_g1_ev = [0.163155, 0.163155, 0.159725, 0.146330, 0.131669, 0.115615, 0.113065, 0.111347]
        min_DCFs_g2_ev = [0.163151, 0.161670, 0.148420, 0.132269, 0.116360, 0.108779, 0.104739, 0.103521]
        min_DCFs_g3_ev = [0.164413, 0.160128, 0.140213, 0.117976, 0.099968, 0.098859, 0.111474, 0.135788]

        
        plt.figure()
        plt.plot(C_array, min_DCFs_g0, '--b', label='??=0.0001-val')
        plt.plot(C_array, min_DCFs_g1, '--g', label='??=0.001-val')
        plt.plot(C_array, min_DCFs_g2, '--r', label='??=0.01-val')
        plt.plot(C_array, min_DCFs_g3, '--y', label='??=0.1-val')
        plt.plot(C_array, min_DCFs_g0_ev, 'b', label='??=0.0001-eval')
        plt.plot(C_array, min_DCFs_g1_ev, 'g', label='??=0.001-eval')
        plt.plot(C_array, min_DCFs_g2_ev, 'r', label='??=0.01-eval')
        plt.plot(C_array, min_DCFs_g3_ev, 'y', label='??=0.1-eval')
        # plt.legend()
        plt.tight_layout() # TBR: Use with non-default font size to keep axis label inside the figure
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("../Images/min_DCF_C_RBF_SVM_eval_gaussianized.pdf")
        else:
            plt.savefig("../Images/min_DCF_C_RBF_SVM__eval_raw.pdf")
        plt.show()
    
    return min_DCFs

"""
----------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------QUADRATRIC SVM------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""

def train_SVM_quadratic(DTR, LTR, C, piT, rebalance, K=1):
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    kernel= ((numpy.dot(DTR.T,DTR)+1)**2)+K #K account for byas term
    H=mcol(Z) *mrow(Z) * kernel
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad

    if rebalance:
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        pi_emp_F = (DTR0.shape[1] / DTR.shape[1])
        pi_emp_T = (DTR1.shape[1] / DTR.shape[1])
        CT = C * piT/pi_emp_T 
        CF = C * (1-piT)/pi_emp_F
        
        #bounds: a list of DTR.shape[1] elements. Each element is a tuple (0,CT) or (0,CF)
        bounds = [(0,1)]*DTR.shape[1] #Initialize the array with random values 
        for i in range(DTR.shape[1]):
            if LTR[i]==0:
                bounds[i]=(0,CF)
            else:
                bounds[i]=(0,CT)
    else:#no re-balancing
        bounds = [(0,C)]*DTR.shape[1]
            
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, #function to minimze
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = bounds,
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation  
                                                      )

    return  alphaStar, Z

K_fold=5
def compute_score_quadratic(DTE,DTR,LTR, Options, c=1, d=2, K=1):
    if Options['C']== None:
        Options['C'] = 0
    if Options['piT'] == None:
        Options['piT']=0.5
    if Options['rebalance']==None:
        Options['rebalance']=True
    kernel= ((numpy.dot(DTR.T,DTE)+c)**d)+K #K account for byas term
    alphaStar, Z= train_SVM_quadratic(DTR, LTR, Options['C'], Options['piT'], Options['rebalance'] )
    score= numpy.dot(alphaStar * Z, kernel)
    return score.ravel()
    
def plot_quadratic_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=None, LEV=None, evaluation=False):
    print('Quadratic SVM: computation for plotting min_cdf wrt C started...')
    min_DCFs=[]
    # for pi in [0.1, 0.5, 0.9]:
    #     C_array = numpy.logspace(-4,3, num = 8)
    #     for C in C_array:
    #             Options= {'C': C,
    #                       'piT':0.5,
    #                       'rebalance':True}
    #             min_dcf_kfold = validate.kfold(DTR, LTR, K_fold, pi, compute_score_quadratic, Options )[0] 
    #             min_DCFs.append(min_dcf_kfold)
    #             print ("computed min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_dcf_kfold))
    min_DCFs_p0 = min_DCFs[0:8] #min_DCF results with prior = 0.1
    min_DCFs_p1 = min_DCFs[8:16] #min_DCF results with prior = 0.5
    min_DCFs_p2 = min_DCFs[16:24] #min_DCF results with prior = 0.9
    
    C_array = numpy.logspace(-4,3, num = 8)
    min_DCFs_p0= [0.260658, 0.229814, 0.232716, 0.233898, 0.247618, 0.245756, 0.248192, 0.698612]
    min_DCFs_p1= [0.154606, 0.134704, 0.124637, 0.123606, 0.123467, 0.122109, 0.124175, 0.264217]
    min_DCFs_p2= [0.715637, 0.574091, 0.533394, 0.528295, 0.521198, 0.527551, 0.560418, 0.851759]
    
    
    
    if evaluation == False:
        #setup visualization font
        plt.rc('font', size=16)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, label='prior=0.1')
        plt.plot(C_array, min_DCFs_p1, label='prior=0.5')
        plt.plot(C_array, min_DCFs_p2, label='prior=0.9')   
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("../Images/min_DCF_C_QuadraticSVM_gaussianized.pdf")
        else:
            plt.savefig("../Images/min_DCF_C_QuadraticSVM_raw.pdf")
        plt.show()
        
        
    else:#compare evaluation and validation
        print("Evaluation samples")
        min_DCFs=[]
        # for pi in [0.1, 0.5, 0.9]:
        #     C_array = numpy.logspace(-4,3, num = 8)
        #     for C in C_array:
        #             Options= {'C': C,
        #                       'piT':0.5,
        #                       'rebalance':True}
        #             scores_svm_quad = compute_score_quadratic(DEV, DTR, LTR, Options)
        #             min_DCF_ev = validate.compute_min_DCF(scores_svm_quad, LEV, pi, 1, 1)
        #             min_DCFs.append(min_DCF_ev)
        #             print ("computed min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_DCF_ev))
        #     print(min_DCFs)
        # min_DCFs_p0_eval = min_DCFs[0:8] #min_DCF results with prior = 0.1
        # min_DCFs_p1_eval = min_DCFs[8:16] #min_DCF results with prior = 0.5
        # min_DCFs_p2_eval = min_DCFs[16:24] #min_DCF results with prior = 0.9
        
        min_DCFs_p0_eval = [0.25624857705670984, 0.22891951697768198, 0.2210320242105083, 0.22347701198556946, 0.2292344767573181, 0.22592199976063057, 0.6405813977555066, 0.5866350866994285]
        min_DCFs_p1_eval = [ 0.15150480283667778, 0.12310292975587017, 0.11157930132322103, 0.11428270609843123, 0.11561918543002059, 0.11489177831990854, 0.2760172525968434, 0.33241994997238855]
        min_DCFs_p2_eval = [0.625920, 0.523061, 0.508529, 0.501529, 0.49257519814719664, 0.49493484682006117, 0.8323377604679882, 0.9973009446693657]

    
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, '--b', label='prior=0.1-val')
        plt.plot(C_array, min_DCFs_p1, '--r', label='prior=0.5-val')
        plt.plot(C_array, min_DCFs_p2, '--g', label='prior=0.9-val')   
        plt.plot(C_array, min_DCFs_p0_eval, 'b', label='prior=0.1-eval')
        plt.plot(C_array, min_DCFs_p1_eval, 'r', label='prior=0.5-eval')
        plt.plot(C_array, min_DCFs_p2_eval, 'g', label='prior=0.9-eval')  
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("../Images/min_DCF_C_QuadraticSVM_eval_gaussianized.pdf")
        else:
            plt.savefig("../Images/min_DCF_C_QuadraticSVM_eval_raw.pdf")
        plt.show()
        
    return min_DCFs
