# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:59:43 2023

@author: Jens
"""
# import libraries -----------------
import numpy as np




# -----------------------------------
class Classifier:
    def __init__(self,C,class_names):
        self.C = C
        self.class_names = class_names
    
    def train(self,X,T):
        pass

    def classify(self,x):
        return -1


class MultiLayerPerceptron(Classifier):
    def __init__(self,M=3,C=2,class_names=[],eta0=0.1,eta_inc=1.05,eta_dec=0.7,eta_fade=0
                 ,nMicroEpochs=1,maxEpochs=50,bshuffle=False,nExplore=0,nTrails=1,eps=0.01,debug=0):
        self.M = M
        self.eta0 = eta0
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec
        self.eta_fade = eta_fade
        self.nMicroEpochs = nMicroEpochs
        self.maxEpochs = maxEpochs
        self.bshuffle = bshuffle
        self.nExplore = nExplore
        self.nTrails = nTrails
        self.eps = eps
        self.debug = debug
        self.eta = eta0
        Classifier.__init__(self,C,class_names) # only for calling from main in this file
        
        
        
#    def setParameters(self,M=None,eta0=None,eta_inc=None,eta_dec=None,eta_fade=None
#                 ,nMicroEpochs=None,maxEpochs=None,nExplore=None,nTrails=None,eps=None,debug=None):
        


    def setTrainingData(self,X,T):
        N,D = X.shape
        
        if len(T.shape)==1: # otherwise T.shape = (len(T),) i.e. without second entry
            self.T = np.reshape(T,(len(T),1))
        else:
            self.T = T
        self.X = np.concatenate((np.ones((N,1)),X),1)
        
        if self.debug==1:
            print("X : ",self.X)
            print("T : ",self.T)
        N,D = self.X.shape
        N,K = self.T.shape
        self.K = K        
        self.N = N
        self.D = D
    
        #return X,T
    
    def setRandomWeights(self):
        #self.W1 = np.random.rand(self.M,self.D)-0.5
        #self.W2 = np.random.rand(self.K,self.M)-0.5
        # for tests weight matrices
        self.W1 = np.array([[0.190695,  0.43476544, 0.62971617]
               ,[0.68506212, 0.42229838, 0.26503337]
               ,[0.19631766, 0.88303398, 0.86517736]])
        self.W2 = np.array([[0.5746371, 0.4267539, 0.8519658]])
        #print("self.W1 = ",self.W1)
        #print("self.W2 = ",self.W2)
        return self.W1, self.W2
    
    def propagateActivity(self,x,W1=None,W2=None):
    
        if W1 is None:
            W1 = self.W1
        if W2 is None:
            W2 = self.W2
        #print("x: ", x)
            
        a = np.dot(W1,x)
        #print("a: ", a)
        z = np.tanh(a)
        #print("z: ", z)
        #print("W2", W2)
        y = np.dot(W2,z)
        self.yn = y
        self.zn = z
    
        return self.yn


    def getError(self,W1=None,W2=None,idx=None):
        
        if W1 is None:
            W1 = self.W1
        if W2 is None:
            W2 = self.W2
            
        #y = np.arange(len(T))
        #z = np.arange(len(T))
        #print("idx",idx)
        if idx:
            r=idx # only certain amount of training data
        else:
            r = self.N # all training data
         
        error = 0
        err_count = 0
        #print("Range for Error Calc: ", r)
        for i in range(r):
            #print("propActiv: ",propagateActivity(X[i], W1, W2))
            #print("W1 = ",W1)
            #print("W2 = ",W2)
            y = self.propagateActivity(self.X[i], W1, W2)#[0]
            #print("y= ",y , "; t= ",self.T[i])
            #error = error + (y-self.T[i])**2
            error = error + np.dot((y-self.T[i]),(y-self.T[i]))
            
            if self.C == 2:
            # classification if C=2:
            # misclassification if label is >=0 but y says <0 (here )
                t = -1 # t is assigned class
                if (y>=0): t=1
                if t*self.T[i]<0:         
                #if ((y_decs < 0 and T[n] > 0) or (y_decs > 0 and T[n] < 0)):
                    err_count = err_count + 1
            #Classification if C>2
            else:
                t = np.argmax(y)
                if t != np.argmax(self.T[i]):
                    err_count = err_count +1

        print("Anzahl Misclass: ",err_count)
        print("in get_Error: error = ",error)    
        return 0.5*error

    def getGradientPart_backprop(self,xn,tn,W1=None,W2=None):
        
        self.yn = self.propagateActivity(xn, W1, W2)
        #print("xn= ",xn)
        #print("yn= ",self.yn)
        #print("zn= ",self.zn)
        
        #print("------ in part backprop ---------------------------")
        delta_out = self.yn-tn
        #print("delta_out= ", delta_out)
    
        delta_n = (1-self.zn**2)*np.dot(W2.T,delta_out) # check if still good in C=2 case!!!
        #print("delta_n= ", delta_n)
    
        D_W1 = np.outer(delta_n,xn)
        #print("in backprop part D_W1= ", D_W1)
        D_W2 = np.outer(delta_out,self.zn)
        #print("in backprop part D_W2= ", D_W2)
        #print("------ in part backprop ---------------------------")
        return D_W1,D_W2

    def getGradient_backprop(self,idx=None,W1=None,W2=None):
        
        if W1 is None:
            W1 = self.W1
        if W2 is None:
            W2 = self.W2
            
        #y = np.arange(len(T))
        #z = np.arange(len(T))
        #print("idx in backprop",idx)
        if idx is not None:
            r=idx # idx is range! only certain amount of training data
            #print("r durch idx", r)
        else:
            r = range(self.N) # all training data
            #print("r durch self.N range", r)
            
        nabla1_ges = 0
        nabla2_ges = 0
        #print("Range for backprop: ", r)
        for i in r:
            #print("i for backprop ", i)
            nabla1,nabla2 = self.getGradientPart_backprop(self.X[i], self.T[i],W1,W2)
            #print("nabla1 = ", nabla1)
            #print("nabla2 = ", nabla2)
            
            nabla1_ges = nabla1_ges + nabla1
            nabla2_ges = nabla2_ges + nabla2

        return nabla1_ges,nabla2_ges
  
    def getGradient_numeric(self,idx=None,W1=None,W2=None,epsilon=1e-5):
        
        if(W1 is None): W1=self.W1 
        if(W2 is None): W2=self.W2
        
        idim = W1.shape[0]
        jdim = W1.shape[1]
        eps_mat = np.zeros((idim,jdim))
        grad1_num = np.zeros((idim,jdim))
        for i in range(idim):
            for j in range(jdim):
                eps_mat[i,j]=epsilon
                E1_peps = self.getError(W1+eps_mat,W2,idx)
                E1_meps = self.getError(W1-eps_mat,W2,idx)
                grad1_num[i,j] = (E1_peps-E1_meps)/(2*epsilon)
                eps_mat[i,j]=0
        
        
        idim = W2.shape[0]
        jdim = W2.shape[1]
        eps_mat = np.zeros((idim,jdim))
        grad2_num = np.zeros((idim,jdim))
        for i in range(idim):
            for j in range(jdim):
                eps_mat[i,j]=epsilon
                E2_peps = self.getError(W1,W2+eps_mat,idx)
                E2_meps = self.getError(W1,W2-eps_mat,idx)
                grad2_num[i,j] = (E2_peps-E2_meps)/(2*epsilon)
                eps_mat[i,j]=0
        
        
        #print("Grad1_num: ",grad1_num)
        #print("Grad2_num: ",grad2_num)
        
        return grad1_num,grad2_num
    
    def checkGradient(self,idx=None,W1=None,W2=None,epsilon=1e-5):
        
        num_grad1,num_grad2 = self.getGradient_numeric(idx,W1,W2,epsilon)
        ana_grad1,ana_grad2 = self.getGradient_backprop(idx,W1,W2)
        
        # put into vector
        ana_vec = np.hstack((ana_grad1.flatten(),ana_grad2.flatten()))                # gradient as flat vector
        num_vec = np.hstack((num_grad1.flatten(),num_grad2.flatten()))   
        
        #print("ana_vec: ",ana_vec)
        #print("num_vec: ",num_vec)
        #relative error:
        rel_err = np.linalg.norm(num_vec-ana_vec)/((np.linalg.norm(num_vec)+np.linalg.norm(ana_vec))/2) 
        
        #print("Gradient-Check relative Error: ", rel_err)
        return rel_err
    
#    def doLearningStep(self,W1,W2,xn,tn,eta=None):
#            
#        #print("W1= ",W1ls)
#        #print("W2= ",W2ls)
#        self.yn = self.propagateActivity(xn, W1, W2)
#        #print("xn= ",xn)
#        #print("yn= ",self.yn)
#        #print("zn= ",self.zn)
#        
#    
#        delta_out = self.yn-tn
#        #print("delta_out= ", delta_out)
#    
#        delta_n = (1-self.zn**2)*(W2*delta_out)
#        #print("delta_n= ", delta_n)
#    
#        D_W1 = np.outer(delta_n,xn)
#        #print("D_W1= ", D_W1)
#        D_W2 = np.outer(delta_out,self.zn)
#        #print("D_W2= ", D_W2)
#    
#        W1_new = W1 - eta*D_W1
#        W2_new = W2 - eta*D_W2
#    
#        return W1_new,W2_new
#

    def doLearningEpoche(self,W1=None,W2=None,eta=None):
        
        if W1 is None:
            W1 = self.W1
        if W2 is None:
            W2 = self.W2
        #print("W1 = ",W1)
        #print("W2 = ",W2)
        
        nME = self.nMicroEpochs # aka Mini-Batches!!
        #print("nMicroEpochs",self.nMicroEpochs)
        if self.bshuffle==True:
            # permutation of numbers of data points
            perm = np.random.permutation(self.N)
            #print("Permutation ", perm)
        else:
            perm = np.arange(self.N)
            #print("Pseudo Permutation ", perm)
        # devide training set into mini batches, if nME=N one loop over all training data follows
        idxME_list = [range(i*self.N//nME,(i+1)*self.N//nME) for i in range(nME)]
        #print("idxME_list: ",idxME_list)
        
        # "shuffle" mini batch for better learning
        idxME_list_perm = [perm[j]   for j in idxME_list]
        #print("idxME_list_perm: ",idxME_list_perm)
        # loop over mini batches:
        for idxME in idxME_list_perm:
            D_W1,D_W2 = self.getGradient_backprop(idxME,W1,W2)
            #print("D_W1 = ",D_W1)
            #print("D_W2 = ",D_W2)
            W1 = W1 - eta*D_W1
            W2 = W2 - eta*D_W2
            #print("--- in epoche pro X--------")
            #print("W1 = ",W1)
            #print("W2 = ",W2)
            #print("---------------------------")
        # compute new error and missclassifications (within getError())
        err = self.getError(W1, W2)
        # gradienten check
        if self.debug == 2:
            r_err = self.checkGradient(None,W1,W2)
            #print("Check gradient relative error: ", r_err)
        else:
            r_err = "none"
            
        return W1,W2,err,r_err


    def doLearningTrail(self):
       

        # initial error
        error = self.getError(self.W1, self.W2)
        print("Initial Fehler: ",error)

        # loop over epochs
        for epo in range(self.maxEpochs):
            #print("Epoche: ",epo)
            
            # adjust eta if "fade out" is non-zero, set via parameter
            self.eta = self.eta/(1+self.eta_fade*epo)
            
    
            if self.nExplore == 1: # dann "Finalization" mode
                err_old = error # for comparison
                W1_old = self.W1
                W2_old = self.W2
    
            self.W1,self.W2,error,r_error = self.doLearningEpoche(None,None,self.eta)
            #print("---------------------------------------")
            #print("after Lepo W1 = ",self.W1)
            #print("after Lepo W2 = ",self.W2)
            #print("---------------------------------------")
            if self.debug==3:
                print("Epoche: ",epo)
                print("Fehler: ",error)
                print("CheckGrad Rel-Fehler: ",r_error)
                print("eta: ",self.eta)
            if self.nExplore == 1: # dann "Finalization" mode
                #only take new weights if error decreased
                if error < err_old:
                    self.eta = self.eta*self.eta_inc           
                    #print("Error decreased: eta_new= ",self.eta)
                else:  # take the old weights if error did not decrease
                    self.eta = self.eta*self.eta_dec
                    self.W1 = W1_old
                    self.W2 = W2_old
                    #print("Error increased: eta_new= ",self.eta)

        return self.W1,self.W2,error


    def doTraining(self,X,T):
        
        self.setTrainingData(X, T)
        # for each training new weights should be used
        self.setRandomWeights()

        for n in range(self.nTrails):
            print("Start Trail ",n," --------------")
            self.W1,self.W2,err = self.doLearningTrail()
            if err <= self.eps: # cutting criterium
                print("End of trail. Error small enough:", err)
                print("Final weights:")
                print("W1 = ",self.W1)
                print("W2 = ",self.W2)
                return 0
            
        print("End of trail: ---------------------")
        #print("W1 = ",self.W1)
        #print("W2 = ",self.W2)
        print("Error = ",err)
        #print("Final y = ", self.yn)



    def doPrediction(self,xtest):
        xtest = np.concatenate(([1],xtest))
        ytest = self.propagateActivity(xtest,self.W1,self.W2)
        if self.C == 2:
            if ytest >=0: pred=1.0
            else: pred=-1.0
        else:
            t = np.argmax(ytest)
            pred = self.class_names[t]
        
        #print("Prediction Class: ",pred)
        return pred


#--------------------------------------------------------------
# MAIN 
#--------------------------------------------------------------

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
    cl_names = np.unique(T)
    #print("T= ",T)
    # ---- Start Training
    mlp = MultiLayerPerceptron(M=3,C=2,class_names=cl_names,eta_fade=0,nMicroEpochs=len(T),maxEpochs=50,bshuffle=False,debug=1)
    
    
    mlp.doTraining(X,T)
    
    #xinput=np.array([1,1.5,0])
    #mlp.doPrediction(xinput)

    
"""
Korrektes Ergebnis im Vergleich zu Knoblauch-Lösung
End of trail, final weights: ---------------------
W1 =  [[ 1.17066274 -0.00208039 -0.20151125]
 [ 1.1838708   1.49247581 -0.64355327]
 [-2.00423215  0.96073423  0.97643644]]
W2 =  [[ 1.38980713 -1.26792726  1.26510412]]
Error =  0.1625924952701836
"""
   

""" micro epoches len(T)//2:
Anzahl Misclass:  2
in get_Error: error =  5.052396772712838
End of trail: ---------------------
Error =  2.526198386356419
"""