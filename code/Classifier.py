# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:05:09 2023

@author: Jens
"""
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from MultiLayerPerceptron import MultiLayerPerceptron
from DNN_flex import DeepNeuralNet

class Classifier:
    def __init__(self,C,class_names):
        self.C = C
        self.class_names = class_names
    
    def train(self,X,T):
        pass

    def classify(self,x):
        return -1

    def confusion_matrix(self,class_names,predict,true_labels):
        cf_mat = pd.DataFrame(np.zeros((len(class_names),len(class_names))),index=class_names,columns=class_names)
        
        print("sklearn CF_Mat: ", skm.confusion_matrix(true_labels,predict))
        print("sklearn Precision: ", skm.precision_score(true_labels,predict,average=None))
        print("sklearn Recall: ", skm.recall_score(true_labels,predict,average=None))
        print("sklearn F1: ", skm.f1_score(true_labels,predict,average=None))
        
        for i in range(len(true_labels)):
            cf_mat[predict[i]][true_labels[i]] += 1
        
        return cf_mat
    
    def crossvalidation(self,S,X,T):
        n=len(T)
        cl_names = np.array(np.unique(T))#,dtype=str)
        cv_err = 0
        print("Class Names: ",cl_names)
        
        
        data_split = [range(i*n//S,(i+1)*n//S) for i in range(S)]
        
        for ds in data_split:
            test_data = X[ds,:]
            print("test_data = ",test_data)
            test_labels=np.array(T[ds],dtype=str)
            #test_labels=T[ds]
            idxtrain = [i for i in range(n) if i not in ds]
            train_data = X[idxtrain,:]
            train_labels =T[idxtrain]
            # go for training
            self.train(train_data,train_labels)
            
            # prep for testing
            miscl = 0
            #print("test_labels: ",test_labels)
            dt = test_labels.dtype.name # in order to get correct data typ of labels
            print("dt = ",dt)
            if dt.startswith("str"):
                pred_class = np.array(range(len(ds)),dtype=str)
            else:
                pred_class = np.array(range(len(ds)))
            # test loop & confusion matrix
            print("pred_class = ",pred_class)
            for i in range(len(ds)):
                #print("self.classify(test_data[i]) = ", self.classify(test_data[i]))
                pred_class[i] = self.classify(test_data[i])    
                if pred_class[i] != test_labels[i]: miscl += 1
            cl_err = miscl/len(ds)
            print("pred_class = ",pred_class)
            print("test_labels = ",test_labels)
            print("Classify Error: ", cl_err)
            print("Confusion Matrix: Split(",ds ,") ---> ", self.confusion_matrix(cl_names,pred_class,test_labels))
            cv_err = cv_err + cl_err
        cv_err = cv_err/S
        print("Crossval Error: ",cv_err)

class MLP2Classifier(Classifier,MultiLayerPerceptron):
    def __init__(self,C=2,class_names=[],M=3,eta0=0.1,eta_inc=1.05,eta_dec=0.7,eta_fade=0
                 ,nMicroEpochs=1,maxEpochs=50,bshuffle=False,nExplore=0,nTrails=1,eps=0.01,debug=0):
        Classifier.__init__(self,C,class_names) 
        
        MultiLayerPerceptron.__init__(self,M,eta0,eta_inc,eta_dec,eta_fade
                 ,nMicroEpochs,maxEpochs,bshuffle,nExplore,nTrails,eps,debug)
        
    def train(self,X,T):
        # get number of classes:
        C = len(np.unique(T))
        # in order to get correct Dimensions
        if C >2:
            one_hot = pd.get_dummies(T)
            Toh = np.array(one_hot)
        else:
            dt = T.dtype.name
            if dt.startswith("int"):
                T[T==0]=-1
            elif dt.startswith("float"):
                T[T==0.0]=-1.0
            Toh = np.array(T)
        
        MultiLayerPerceptron.doTraining(self,X, Toh)
        
    def classify(self,X):
        #N = X.shape[0]
        #self.X = np.concatenate((np.ones((1,1)),X),axis=None)
        #print("in Classify: X = ", X)
        return MultiLayerPerceptron.doPrediction(self,X)

class DNN_Classifier(Classifier,DeepNeuralNet):
    def __init__(self,L=3,C=2,class_names=[],eta0=0.1,eta_inc=1.05,eta_dec=0.7,eta_fade=0,beta1=0.9,beta2=0.999,epsilon=1e-8,optim="SGD"
                 ,err_fkt="SSE",nMicroEpochs=1,maxEpochs=50,bshuffle=False,nExplore=0,nTrails=1,eps=0.01,debug=0):
        Classifier.__init__(self,C,class_names)
        
        
        DeepNeuralNet.__init__(self,L,eta0,eta_inc,eta_dec,eta_fade,beta1,beta2,epsilon,optim
                 ,err_fkt,nMicroEpochs,maxEpochs,bshuffle,nExplore,nTrails,eps,debug)        


    def train(self,X,T):
        # get number of classes:
        C = len(np.unique(T))
        # in order to get correct Dimensions
        if C >2:
            one_hot = pd.get_dummies(T)
            Toh = np.array(one_hot)
        else:
            dt = T.dtype.name
            if dt.startswith("int"):
                T[T==0]=-1
            elif dt.startswith("float"):
                T[T==0.0]=-1.0
            Toh = np.array(T)
        
        DeepNeuralNet.doTraining(self,X, Toh)
        
    def classify(self,X):

        #print("in Classify: X = ", X)
        return DeepNeuralNet.doPrediction(self,X)




#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    #Data matrices
    X1 = np.array([[-2,-1], [-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]])  # class 1 data
    N1,D1 = X1.shape
    X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]])       # class 2 data
    N2,D2 = X2.shape
    
    T1 = np.ones(N1)
    T2 = np.ones(N2)*-1
    #T2 = np.ones(N2)
    
    X=np.concatenate((X1,X2))
    T=np.concatenate((T1,T2))
   # Tcol1=np.concatenate((T1,T2*0)).reshape(-1,1)
    #Tcol2=np.concatenate((T1*0,T2)).reshape(-1,1)
    #T=np.concatenate((Tcol1,Tcol2),axis=1)
    #print("T= ",T)
    # ---- Start Training
    #mlp = MultiLayerPerceptron(eta_fade=0,nMicroEpochs=len(T),maxEpochs=50,bshuffle=False)
    
    
    #mlp.doTraining(X,T)


    mlp_object = MLP2Classifier()
    mlp_object.__init__(eta_fade=0,nMicroEpochs=len(T),maxEpochs=50,bshuffle=False)
    mlp_object.train(X, T)
    x_input = np.array([-1.5,0])
    mlp_object.classify(x_input)