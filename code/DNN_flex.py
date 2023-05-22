# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:48:37 2023

@author: Jens
"""


# import libraries -----------------
import numpy as np




# -----------------------------------
class Classifier: # not really neccessary here
    def __init__(self,C,class_names):
        self.C = C
        self.class_names = class_names
    
    def train(self,X,T):
        pass

    def classify(self,x):
        return -1


class Layer():
    def __init__(self,M,activation="lin"):
    #    pass
    #
    #def build(self,M,activation="lin"):
        #self.L           =L
        self.M           =M
        self.b           =np.zeros(M,dtype="float")
        self.a           =np.zeros(M,dtype="float")
        self.alpha       =np.zeros(M,dtype="float")
        self.z           =np.zeros(M,dtype="float")
        self.delta       =np.zeros(M,dtype="float")
        self.activation  =activation
        
            
# here vectors of dendritic potential, fire rate, error pot and error signal  
    #def dendritic_pot(self):

    def firerate(self,a_pot,act):
        
        if act == "lin":
            self.z = a_pot #should be something else, just for test =a
        elif act == "tanh":
            self.z = np.tanh(a_pot) 
        elif act == "sigmoid":
            self.z = np.divide(1.0,1+np.exp(-a_pot))
        
        return self.z
    
    def jacobi(self,z,act):
        
        if act=="lin":
            return 1
        elif act=="tanh":
            return (1 - z*z)
        elif act == "sigmoid":
            return np.multiply(z,1.0-z)
        
 
class Connection:
    def __init__(self,lin,lout):
        self.lin=lin
        self.lout=lout
        self.W=[]
        
    def weights(self,l,lp):
        #li = self.layer[l].M
        #lo = self.layer[lp].M
        weight=np.random.rand(l,lp)-0.5 # -0.5 in order to have random vars around zero and not in [0,1]
        
        return weight

        #,C=2,class_names=[-1,1]
class DeepNeuralNet(Layer,Classifier):
    def __init__(self,L=3,eta0=0.1,eta_inc=1.05,eta_dec=0.7,eta_fade=0,beta1=0.9,beta2=0.999,epsilon=1e-8,optim="SGD"
                 ,err_fkt="SSE",nMicroEpochs=1,maxEpochs=50,bshuffle=False,nExplore=0,nTrails=1,eps=0.01,debug=0):
        self.L = L
        self.eta0 = eta0
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec
        self.eta_fade = eta_fade
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.nMicroEpochs = nMicroEpochs
        self.maxEpochs = maxEpochs
        self.bshuffle = bshuffle
        self.nExplore = nExplore
        self.nTrails = nTrails
        self.eps = eps
        self.debug = debug
        self.eta = eta0
        self.optim = optim
        self.err_fkt = err_fkt
        self.layer=[]
        self.conn=[]
        self.W=[]
        #Classifier.__init__(self,C,class_names) # only for calling from main in this file 
        #Layer.__init__(self)    
        
        
#    def setParameters(self,M=None,eta0=None,eta_inc=None,eta_dec=None,eta_fade=None
#                 ,nMicroEpochs=None,maxEpochs=None,nExplore=None,nTrails=None,eps=None,debug=None):
        
    def init_net(self,lu,activ):
        # lu contains number of units in each layer respectively
        #l1 = Layer(lu[1],"lin")
        #l2 = Layer(lu[2],"tanh")
        #print("l1=",l1)
        for i in range(len(lu)):
            
            self.conn.append([i] * len(lu))
            self.W.append([0] * len(lu))
            self.layer.append([i])
            #self.DW
        for i in range(len(lu)):
            self.layer[i]=Layer(lu[i],activ[i])
            #print("layer",i,":  ",self.layer[i])
            #print("layer",i,": M= ",self.layer[i].M)
            #print("layer",i,": z= ",self.layer[i].z)
            #print("layer",i,": e.g. firerate= ",self.layer[i].firerate(4))
            #if i == 1:
            self.layer[i].b = np.random.rand(self.layer[i].M)
                #self.layer[i].b = np.ones(self.layer[i].M,dtype='float')
                #self.layer[i].b = np.array([0.190695,0.68506212,0.19631766])
            #print("layer",i,"= ",self.layer[i])    
        for i in range(len(lu)-1):        
            print("i=",i)
            #self.conn[i][i+1]=Connection.__init__(self,i,i+1)
            #print("ConnWeights: ",Connection.weights(self, lu[i], lu[i+1]))
            self.W[i][i+1]=Connection.weights(self, lu[i], lu[i+1]).T
            # for tests weight matrices
            #if i==0:
            #    self.W[i][i+1] = np.array([[  0.43476544, 0.62971617]
            #                               ,[ 0.42229838, 0.26503337]
            #                               ,[ 0.88303398, 0.86517736]])
            #if i==1:
            #    self.W[i][i+1] = np.array([[0.5746371, 0.4267539, 0.8519658]])
            
            print("W[",i,",",i+1,"] = ",self.W[i][i+1])
            print("b[",i,"]",self.layer[i].b)
            
    def setTrainingData(self,X,T):
        N,D = X.shape
        
        # init list of data vectors z for each layer step
        #self.z=[]
        #for i in range(self.L):
        #    self.z.append([i]*1)
        
        
        # set input data to z0
        #self.z[0] = np.concatenate((np.ones((N,1)),X),1)
        #for i in range(1,self.L):
        self.layer[0].z = X[0]
        print("z : ",self.layer[0].z)
        #print("z00 : ",self.z[0][1])
        
        
        
        if len(T.shape)==1: # otherwise T.shape = (len(T),) i.e. without second entry
            self.T = np.reshape(T,(len(T),1))
        else:
            self.T = T
        self.X = X
        
        N,D = self.X.shape
        N,K = self.T.shape
        self.K = K        
        self.N = N
        self.D = D
    
        if self.debug==1:
            print("X : ",self.X)
            print("T : ",self.T)
            print("K : ",self.K)
            print("N : ",self.N)
            print("D : ",self.D)
        #return X,T
    
#    def setRandomWeights(self):
#       self.W1 = np.random.rand(self.M,self.D)#-0.5
#        self.W2 = np.random.rand(self.K,self.M)#-0.5
#        # for tests weight matrices
#        #self.W1 = np.array([[0.190695,  0.43476544, 0.62971617]
#        #       ,[0.68506212, 0.42229838, 0.26503337]
#        #       ,[0.19631766, 0.88303398, 0.86517736]])
#        #self.W2 = np.array([[0.5746371, 0.4267539, 0.8519658]])
#        
#        return self.W1, self.W2
    
    def forward_pass(self,zprop,w):
        
        
        if w is None:
             w = self.W

        # z only for forward pass internally relevant and backward pass internally
        self.layer[0].z=zprop

        #print("zprop: ", zprop)
        #print("w: ", w)
        
        for il in range(1,self.L):
            wz=0
            for i in range(il,0,-1): # do we need here a -1 in upper range as well???
                #print("w[",il-1,"][",i,"]= ",w[il-1][i])
                #print("layer[",i-1,"].z= ",self.layer[i-1].z)
                #print("type(w[il-1][i])",type(w[il-1][i]))
                if type(w[il-1][i]).__name__ == "ndarray":
                    wz = wz + np.dot(w[il-1][i],self.layer[i-1].z) # z=x for input
                    #print("wz(",il,",",i,")= ",wz)
                else:
                    wz = wz
                    #print("im else wz(",il,",",i,")= ",wz)
            self.layer[il].a = self.layer[il].b + wz
            #print("layer[",il ,"].a: ", self.layer[il].a)
            #print("layer[",il ,"].b: ", self.layer[il].b)
            self.layer[il].z = self.layer[il].firerate(self.layer[il].a,self.layer[il].activation)
        
            #print("after one l: w =", w)

      
        self.y = self.layer[self.L-1].z # output
    
        return self.y


    def getError(self,W,err_fkt,idx=None):
        
        if W is None:
            W = self.W
        #if W2 is None:
        #    W2 = self.W2
            
        #y = np.arange(len(T))
        #z = np.arange(len(T))
        #print("idx",idx)
        #print("W = ",W)
        if idx:
            r=idx # only certain amount of training data
        else:
            r = self.N # all training data
         
        error = 0
        err_count = 0
        #print("Range for Error Calc: ", r)
        for i in range(r):
            #print("propActiv: ",propagateActivity(X[i], W1, W2))
            #print("W = ",W)
            y = self.forward_pass(self.X[i], W)#[0]
            #print("y= ",y , "; t= ",self.T[i])
            #error = error + (y-self.T[i])**2
            if self.err_fkt=="SSE":
                error = error + 0.5*np.dot((y-self.T[i]),(y-self.T[i]))
            elif self.err_fkt=="BCE":
                error = error - (np.dot(self.T[i],np.log(y))+np.dot((1-self.T[i]),np.log(1-y)))
            elif self.err_fkt=="CCE":
                error = error - np.dot(self.T[i],np.log(y))
            
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
        return error

    def getGradientPart_backprop(self,xn,tn,W):
        
        yn = self.forward_pass(xn, W)
        #print("tn= ",tn)
        #print("yn= ",yn)
        #print("in backprop: W= ",W)

        
        
        self.layer[self.L-1].delta = yn-tn
        #print("delta_out",[self.L-1],"= ", self.layer[self.L-1].delta)
        
        #print("----------- in part backprop ------------------------------------")
        for l in range(self.L-2,-1,-1): # minus one in upper range in order to get 0 as well
            Wd=0    
            for i in range(l,self.L): 
                #print("W[",l,"][",i,"]",W[l][i])
                #print("type(self.W[i][l]).__name__: ",type(W[l][i]).__name__)
                if type(W[l][i]).__name__ == "ndarray":
                    Wd = Wd + np.dot(W[l][i].T,self.layer[i].delta)
                    #print("self.layer[",i,"].delta: ",self.layer[i].delta)
                else:
                    pass#self.layer[l].alpha = self.layer[l].alpha
                    
            #G = np.outer(self.layer[l].delta,self.layer[i].z)
            self.layer[l].alpha = Wd 
            self.layer[l].delta = self.layer[l].jacobi(self.layer[l].z,self.layer[l].activation)*self.layer[l].alpha # jacobi in case of lin activation equal to 1
               
            
            #print("self.layer[",l,"].alpha = ",self.layer[l].alpha)
            #print("self.layer[",l,"].delta = ",self.layer[l].delta)
        
    
        #print("----------- ENDE part backprop ------------------------------------")
        #print("in backprop part D_W1= ", D_W1)
        

        

    def getGradient_backprop(self,idx=None,W=None):
        
        if W is None:
            W = self.W
            
        
        #print("idx in backprop",idx)
        if idx is not None:
            r=idx # idx is range! only certain amount of training data
            #print("r durch idx", r)
        else:
            r = range(self.N) # all training data
            #print("r durch self.N range", r)
            
        # create lists for updating weights
        delta_ges = [0 for i in range(self.L)]
        DW = []
        for i in range(self.L):
            DW.append([0] * self.L)
        #print("DW= ",DW)
        #print("Range for backprop: ", r)
        
        for i in r:
            #print("i for backprop ", i)
            self.getGradientPart_backprop(self.X[i], self.T[i],W)
            
            #-----------------------------------------------------------------
            for l in range(self.L): 
                for lp in range(self.L):
                    if lp==l+1:
                        #print("-------- UPDATE Weights ------------------------------")
                        #print("self.layer[",l,"].delta = ",self.layer[l].delta)
                        #print("self.layer[",lp,"].z = ",self.layer[lp].z)
                        DW[l][lp] = DW[l][lp] + np.outer(self.layer[lp].delta,self.layer[l].z)
                        #W[l][lp] = W[l][lp] - eta*np.outer(delta[lp],self.layer[l].z)
                
                #if l==1: #this "if" only for comparison with exercixe 4.5
                delta_ges[l] = delta_ges[l] + self.layer[l].delta # if-cond only for comparison with Ubung 4.x MLP
                    #self.layer[l].b = self.layer[l].b - eta*delta[l]
            
            #-----------------------------------------------------------------
            #for l in range(self.L): # check for other than len(T) micro epoches
            #    
            #    delta_ges[l] = delta_ges[l] + self.layer[l].delta

                #print("delta_ges[",l,"] = ", delta_ges[l])
            
            
        return DW,delta_ges
  
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
    


    def doLearningEpoche(self,W=None,eta=None,tau=None):
        
        if W is None:
            W = self.W
        #print("W = ",W)
        # for optimization methods:
        m = []
        v = []
        #nabla_ges=[]
        for i in range(self.L):
            m.append([0] * self.L)
            v.append([0] * self.L)
            #nabla_ges.append([0] * self.L)
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
            deltaW,delta = self.getGradient_backprop(idxME,W)
            #print("delta = ", delta)
            #for l in range(self.L): # DO THIS MULTIPLICATION of delta AND z in backprop function!!!
             #   for lp in range(self.L):
              #      if lp==l+1:
                        #print("-------- UPDATE Weights ------------------------------")
                        #print("self.layer[",l,"].delta = ",self.layer[l].delta)
                        #print("self.layer[",lp,"].z = ",self.layer[lp].z)
                        #W[l][lp] = W[l][lp] - eta*np.outer(self.layer[lp].delta,self.layer[l].z)
            #W = W - np.dot(eta,nablaW) # here np.dot otherwise no componentwise multiplication
            
            #W = np.array(W,dtype=object) - eta*np.array(deltaW,dtype=object)
            #nabla_ges = np.array(nabla_ges,dtype=object) + np.array(deltaW,dtype=object)
            # this update must be here because the adjusting of gradients (with one version 
            # of the weights W) happens in
            # function getGradient_backprop and here the adjustment for the WHOLE BATCH
            # needs to be done, next iteration is another (mini) batch where for all data
            # points in that batch the gradient adjustments are done and here again only for
            # the final correction of the weights for EACH (mini) batch.
            # if this would be done only once after the (mini) batch loop, there would
            # be only one single update of the weights W!!! So, no possibility of
            # stochastic gradient descent method
            # optimization:
            if self.optim == "SGD":
                W = np.array(W,dtype=object) - eta*np.array(deltaW,dtype=object)
            elif self.optim == "MOM":
                m = self.beta1*np.array(m,dtype=object) + np.array(deltaW,dtype=object)
                W = np.array(W,dtype=object)-eta*m
            elif self.optim == "ADAM":
                #print("beta1,beta2,tau,beta**tau",self.beta1,self.beta2,tau,self.beta2**tau)
                m = self.beta1*np.array(m,dtype=object) + (1-self.beta1)*np.array(deltaW,dtype=object)
                v = self.beta2*np.array(v,dtype=object) + (1-self.beta2)*np.array(deltaW,dtype=object)*np.array(deltaW,dtype=object)                
                m_hat= np.array(m/(1.0-self.beta1**(tau+1)),dtype=object)
                v_hat= np.array(v/(1.0-self.beta2**(tau+1)),dtype=object)
                #print("m_hat = ",m_hat)
                #print("v_hat = ",v_hat)
                for i in range(self.L):
                    for j in range(self.L):
                        W[i][j] = W[i][j]-eta*m_hat[i][j]/(np.sqrt(v_hat[i][j])+self.epsilon)
            #print("W = ",W)
               # if l==1: 
                    #self.layer[l].b = self.layer[l].b - eta*self.layer[l].delta # if-cond only for comparison with Ubung 4.x MLP
            for l in range(self.L):
                self.layer[l].b = self.layer[l].b - eta*delta[l]
                #print("self.layer[",l,"].b",self.layer[l].b)
            #W2 = W2 - eta*D_W2
            #print("---- in epoche pro X ---------------")
            #print("W = ", W)
            #print("b = ",self.layer[1].b)
            #print("-------- END ----------------------------")
        #if self.optim == "SGD":
        #    W = np.array(W,dtype=object) - eta*np.array(nabla_ges,dtype=object)
        #elif self.optim == "MOM":
        #    m = self.beta*np.array(m,dtype=object) + np.array(nabla_ges,dtype=object)
        #    W = np.array(W,dtype=object)-eta*m 
            
        # compute new error and missclassifications (within getError())
        err = self.getError(W,self.err_fkt)
        # gradienten check
        if self.debug == 2:
            pass
            #r_err = self.checkGradient(None,W1,W2)
            #print("Check gradient relative error: ", r_err)
        else:
            r_err = "none"
            
        return W,err,r_err


    def doLearningTrail(self):
       

        # initial error
        error = self.getError(self.W,self.err_fkt)
        print("Initial Fehler: ",error)
        
        
        # loop over epochs
        for epo in range(self.maxEpochs):
            #print("Epoche: ",epo)
            
            # adjust eta if "fade out" is non-zero, set via parameter
            self.eta = self.eta/(1+self.eta_fade*epo)
            
    
            if self.nExplore == 1: # dann "Finalization" mode
                err_old = error # for comparison
                W_old = self.W
    
    
            # only here update of "self.weights"!!!
            self.W,error,r_error = self.doLearningEpoche(None,self.eta,epo)
            # optimization:
            #if self.optim == "SGD":
            #    self.W = np.array(self.W,dtype=object) - self.eta*np.array(nablaW,dtype=object)
            #elif self.optim == "MOM":
            #    m = self.beta1*np.array(m,dtype=object) + np.array(nablaW,dtype=object)
            #    self.W = np.array(self.W,dtype=object)-self.eta*m
            #elif self.optim == "ADAM":
            #    m = self.beta1*np.array(m,dtype=object) + (1-self.beta1)*np.array(nablaW,dtype=object)
            #    v = self.beta2*np.array(v,dtype=object) + (1-self.beta2)*np.array(nablaW,dtype=object)*np.array(nablaW,dtype=object)
            
                
            
            #print("---------------------------------------")
            #print("after Lepo W = ",self.W)
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
                    self.W = W_old
                    #print("Error increased: eta_new= ",self.eta)

        return self.W,error


    def doTraining(self,X,T):
        
        self.setTrainingData(X, T)
        # for each training new weights should be used
        #self.setRandomWeights()

        for n in range(self.nTrails):
            print("Start Trail ",n," --------------")
            self.W,err = self.doLearningTrail()
            if err <= self.eps: # cutting criterium
                print("End of trail. Error small enough:", err)
                print("Final weights:")
                print("W = ",self.W)
                return 0
            
        print("End of trail: ---------------------")
        #print("W1 = ",self.W1)
        #print("W2 = ",self.W2)
        print("Error = ",err)
        #print("Final y = ", self.yn)



    def doPrediction(self,xtest):
        
        ytest = self.forward_pass(xtest,self.W)
        if self.C == 2:
            if ytest >=0: pred=1
            else: pred=-1 
        else:
            t = np.argmax(ytest)
            pred = self.class_names[t]
        
        print("Prediction Class: ",pred)
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
    
    #Data matrices for debuggin
    #X1 = np.array([[-2,-1], [-2,2], [-1.5,1]])  # class 1 data
    #N1,D1 = X1.shape
    #X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5]])       # class 2 data
    #N2,D2 = X2.shape
    
    T1 = np.ones(N1)
    T2 = np.ones(N2)*-1
    #T2 = np.ones(N2)
    
    X=np.concatenate((X1,X2))
    T=np.concatenate((T1,T2))
    #Tcol1=np.concatenate((T1,T2*0)).reshape(-1,1)
    #Tcol2=np.concatenate((T1*0,T2)).reshape(-1,1)
    #T=np.concatenate((Tcol1,Tcol2),axis=1)
    cl_names = np.unique(T)
    #print("T= ",T)
    
    layer_units=[2,3,1]
    activ=["lin","tanh","lin"]
    # ---- Start Training
    # L=number of layers
    mlp = DeepNeuralNet(L=3,C=2,class_names=cl_names,eta_fade=0,nMicroEpochs=len(T),maxEpochs=50,bshuffle=False,debug=3)
    mlp.init_net(layer_units,activ)
    
    mlp.doTraining(X,T)
    
    #xinput=np.array([1,1.5,0])
    #mlp.doPrediction(xinput)


"""############################################################################
### To do next:
#    #implement xavier / hu initialization
#    #implement CNN Layer and maxpool layer
"""
