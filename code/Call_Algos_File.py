# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:27:50 2023

@author: Jens
"""
import numpy as np
import pandas as pd
from Classifier import MLP2Classifier,DNN_Classifier

# F means which case should be considered
F=4
# should raw data be used? (1=yes)
raw=0


if F==2:
    #Train C=2 Data matrices
    X1 = np.array([[-2,-1], [-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]])  # class 1 data
    N1,D1 = X1.shape
    X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]])       # class 2 data
    N2,D2 = X2.shape
        
    T1 = np.ones(N1)
    T2 = np.ones(N2)*-1.0
        #T2 = np.ones(N2)
        
    X=np.concatenate((X1,X2))
    T=np.concatenate((T1,T2))
    
    C_names = np.unique(T)
    C = len(C_names)
    
elif F==3:
    #Train C=3 Data matrices
    X1 = np.array([[-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]])  # class 1 data
    N1,D1 = X1.shape
    X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]])       # class 2 data
    N2,D2 = X2.shape
    X3 = np.array([[-2,-1], [-2,-3],[-3,0],[-3,-2],[-4,-1],[-4,-3],[-4,-1]])       # class 2 data
    N3,D3 = X3.shape
    
    
    T1 = np.ones(N1)
    T2 = np.ones(N2)*2
    T3 = np.ones(N2)*3
        
    X=np.concatenate((X1,X2,X3))
    T=np.concatenate((T1,T2,T3))

    C_names = np.unique(T)
    C = len(C_names)

elif F==4:
    # read Forest Data
    data = pd.read_csv("ForestTypesData.csv")
    T = data["class"]
    X_raw = np.array(data.iloc[:,1:])
    C_names = np.unique(T)
    C = len(C_names)
    X = np.ones((X_raw.shape))
    for i in range(X_raw.shape[0]):
        X_norm = (X_raw[i]-np.min(X_raw[i]))/(np.max(X_raw[i])-np.min(X_raw[i]))
        X[i] = X_norm

elif F == 5:
    # Read complete Data
    data = pd.read_excel("CTG_C2.xls",index_col=0)
    T = data["NSP"]
    X_raw = np.array(data.iloc[:,:-1])
    C_names = np.unique(T)
    C = len(C_names)
    if raw == 1:
        X = X_raw
    else:
        X = np.ones((X_raw.shape))
        for i in range(X_raw.shape[0]):
            X_norm = (X_raw[i]-np.min(X_raw[i]))/(np.max(X_raw[i])-np.min(X_raw[i]))
            X[i] = X_norm

    
#------------------------------------------------------------------------------
# Call MLP
#------------------------------------------------------------------------------

#obj = MLP2Classifier()
#obj.__init__(C,C_names,M=6,eta0=0.1,eta_dec=0.7,eta_fade=0,nMicroEpochs=len(T),bshuffle=True,nExplore=1,maxEpochs=50,nTrails=1,debug=0)
#obj.train(X,T)

#obj.doPrediction(X[1])
#obj.__init__(C,C_names,M=15,eta0=0.001,nMicroEpochs=len(T),bshuffle=True,nExplore=1,maxEpochs=50,nTrails=1,debug=0)
#S=3
#obj.crossvalidation(S, X, T)

#------------------------------------------------------------------------------
# Call DNN
#------------------------------------------------------------------------------
#F=2 
#layer_spec = [2,3,3,1]
#activ=["lin","tanh","tanh","lin"]
#F=4
layer_spec = [27,12,12,4]
activ=["lin","tanh","tanh","sigmoid"]

#----------------
obj = DNN_Classifier()
obj.__init__(len(layer_spec),C,C_names,err_fkt="CCE",eta0=0.01,eta_dec=0.9,eta_fade=0,beta1=0.9,beta2=0.999,epsilon=1e-8,optim="ADAM",nMicroEpochs=len(T),bshuffle=True,nExplore=1,maxEpochs=100,nTrails=1,debug=0)
obj.init_net(layer_spec,activ)
obj.train(X,T)

#S=3
#obj.crossvalidation(S, X, T)

"""############################################################################
### To do next:
#    #implement in DNN xavier / hu initialization
#    #implement in DNN CNN Layer and maxpool layer
"""

# so far best result F=4 Forest Data 
"""
layer_spec = [27,12,12,4]
activ=["lin","tanh","tanh","lin"]

in get_Error: error =  61.311231348250566
Anzahl Misclass:  33
in get_Error: error =  61.16464426684009
End of trail: ---------------------
Error =  30.582322133420046
"""
#so far best result for 4.10
"""
Epoche:  999
Anzahl Misclass:  65
Fehler:  81.14160941158444
eta:  9.563373770030702e-05
Error decreased: eta_new=  0.00010041542458532238
End of trail: ---------------------
Error =  81.14160941158444


sklearn CF_Mat:  [[141   1   2  15]
                 [  0  77   0   9]
                 [ 18   1  61   3]
                 [  5   8   1 181]]
sklearn Precision:  [0.8597561  0.88505747 0.953125   0.87019231]
sklearn Recall:  [0.88679245 0.89534884 0.73493976 0.92820513]
sklearn F1:  [0.87306502 0.89017341 0.82993197 0.89826303]
Confusion Matrix:         d     h     o      s 
                    d   141.0   1.0   2.0   15.0
                    h     0.0  77.0   0.0    9.0
                    o    18.0   1.0  61.0    3.0
                    s     5.0   8.0   1.0  181.0


"""



    