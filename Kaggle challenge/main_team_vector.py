#!/usr/bin/env python
# coding: utf-8

# In[1]:


#@title Installation



# @title Imports

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
np.random.seed(54321)

from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import OneHotEncoder


# In[3]:


X_train=pd.read_csv('./data/Xtr.csv', sep=',') #we use this dataset to train our model
Y_train=pd.read_csv('./data/Ytr.csv', sep=',') #we use this dataset to train our model
X_test=pd.read_csv('./data/Xte.csv', sep=',') #we will use this data set later to validate our 



print('The shape of the X_train dataset is:',X_train.shape)
print('The shape of the Y_train dataset is:',Y_train.shape)



# Some models

# Ridge Regression (RR)

class solveRR():
    def __init__(self, X, y, lam=0.1):
        self.beta = None
        self.X = X
        self.y = y
        self.lam = lam
            
    def fit(self):
        
        X = self.X
        y = self.y
        lam = self.lam 
        
        n, p = X.shape
        assert (len(y) == n)

        A = (X.T.dot(X)) + np.eye(p)*lam*n
        b = X.T.dot(y)
        
        self.beta = np.linalg.solve(A, b)
        
        return self.beta
    
        
    def predict(self, X, threshold=.5):
        return np.where(X.dot(self.beta) >= threshold, 1, 0)
        
          
    def Accuracy_check(self,X, y, threshold=.5):
        return np.mean(self.predict(X, threshold)==y)
    

# Weighted Ridge Regression (WRR)
class solveWRR():
    def __init__(self, X, y, w, lam=0.1):
        self.beta = None
        self.X = X
        self.y = y
        self.lam = lam
        self.w = w
    
    def fit(self):
        
        X = self.X
        y = self.y
        lam = self.lam 
        w = self.w
        
        n, p = X.shape
        assert (len(y) == len(w) == n)

        y1 = np.sqrt(w) * y
        X1 = (np.sqrt(w) * X.T).T
        
        # Hint:
        # Find y1 and X1 such that:
        
        self.beta = solveRR(X1, y1, lam).fit()
                
        return self.beta
    
        
    def predict(self, X, threshold):
        return np.where(X.dot(self.beta) >= threshold, 1, 0)
        
          
    def Accuracy_check(self,X, y, threshold=.5):
        return np.mean(self.predict(X, threshold)==y)
    

# Logistic Ridge Regression (LRR)
class solveLRR():
    def __init__(self, X, y, lam=0.1):
        self.beta = None
        self.X = X
        self.y = y
        self.lam = lam
    
    def fit(self):
        
        X = self.X
        y = self.y
        
        n, p = X.shape
        assert (len(y) == n)
    
        lam = self.lam 
        max_iter = 50
        eps = 1e-3
        sigmoid = lambda a: 1/(1 + np.exp(-a))
        
        
        
        # Initialize
        self.beta = np.zeros(p)

        # Hint: Use IRLS
        for i in range(max_iter):
            beta_old = self.beta
            f = X.dot(beta_old)
            w = sigmoid(f) * sigmoid(-f)
            z = f + y / sigmoid(y*f)
            self.beta = solveWRR(X, z, w, 2*lam).fit()
            # Break condition (achieved convergence)
            #if np.sum((beta-beta_old)**2) < eps:
            #    break                
        return self.beta
    
        
    def predict(self, X, threshold):
        return np.where(X.dot(self.beta) >= threshold, 1, 0)
        
          
    def Accuracy_check(self,X, y, threshold=.5):
        return np.mean(self.predict(X, threshold)==y)


# Kernel Models

# In[12]:


def rbf_kernel(X1, X2, sigma=10):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    # For loop with rbf_kernel_element works but is slow in python
    # Use matrix operations!
    X2_norm = np.sum(X2 ** 2, axis=-1)
    X1_norm = np.sum(X1 ** 2, axis=-1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K


# In[13]:

class ksolveRR():
    def __init__(self, X, y, lam= 0.0001, sigma=0.5, kernel=rbf_kernel):
        self.beta = None
        self.X = X
        self.y = y
        self.lam = lam
        self.sigma = sigma
        self.kernel = kernel
            
    
    def fit(self):
        X = self.X
        y = self.y
        lam = self.lam 
        
        n, p = X.shape
        assert (len(y) == n)
        
        if self.sigma is None:
            self.sigma = sigma_from_median(X)
            
        A = self.kernel(X, X, self.sigma)+n*self.lam*np.eye(n)
        self.alpha = np.linalg.solve(A, y)
        
        return self.beta
    
        
    def predict(self, X, threshold=.5):
        K_x = self.kernel(X, self.X, self.sigma)
        return np.where(K_x.dot(self.alpha) >= threshold, 1, 0)
        
          
    def Accuracy_check(self,X, y, threshold=.5):
        return np.mean(self.predict(X, threshold)==y)
    
class ksolveRR_2():
    def __init__(self, X, y, lam= 0.0001, sigma=0.5, sample_weights = None, kernel = None):
        self.alpha = None
        self.X = X
        self.y = y
        self.lam = lam
        self.sigma = sigma
        self.kernel = kernel
        self.sample_weights = sample_weights
            
    
    def fit(self):
        if self.sample_weights is not None:
            self.X *= self.sample_weights[:, None]
        
        X = self.X
        y = self.y
        lam = self.lam
        
        n, p = X.shape
        assert (len(y) == n)
        
        A = self.kernel(X, X, self.sigma)+n*self.lam*np.eye(n)
        self.alpha = np.linalg.solve(A, y)
        
        return self
    
        
    def predict(self, X, threshold=.5):
        K_x = self.kernel(X, self.X, self.sigma)
        return np.where(K_x.dot(self.alpha) >= threshold, 1, 0)
        
          
    def Accuracy_check(self,X, y, threshold=.5):
        return np.mean(self.predict(X, threshold)==y)


# Logistic Ridge Regression (LRR)
class ksolveLRR():
    def __init__(self, X, y, lam = 0.1, sigma = 4, max_iter=100, tol=1e-5, kernel=None):
        self.alpha = None
        self.X = X
        self.y = y
        self.lam = lam
        
        self.kernel = kernel
        
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        
    
    def fit(self):
        
        X = self.X
        y = self.y
        
        n, p = X.shape
        assert (len(y) == n)

        sigmoid = lambda a: 1/(1 + np.exp(-a))
        
        K = self.kernel(X, X)

        # Initialize
        alpha = np.zeros(n)
        
        # Hint: Use IRLS
        for n_iter in range(self.max_iter):
            alpha_old = alpha
            f = K.dot(alpha_old)
            w = sigmoid(f) * sigmoid(-f)
            z = f + y / sigmoid(y*(f))
            
            alpha = ksolveRR_2(X, y, lam = 2*self.lam,\
                               sigma=self.sigma, sample_weights = w).fit().alpha
            
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < self.tol:
                break  
                
                
        self.n_iter = n_iter
        self.alpha = alpha
        
        return self
    
        
    def predict(self, X, threshold):
        K_x = self.kernel(X, self.X)
        return np.where(K_x.dot(self.alpha) >= threshold, 1, 0)
        
          
    def Accuracy_check(self,X, y, threshold=.5):
        return np.mean(self.predict(X, threshold)==y)

solver='cvxopt'

import cvxopt

def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    #cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
    return np.array(solution['x']).flatten()

solve_qp = cvxopt_qp

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = - G.T
        qp_b = - h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    solution = cvxopt.solvers.qp(*cvx_matrices)
    return np.array(solution['x']).flatten()

solve_qp = {'quadprog': quadprog_solve_qp, 'cvxopt': cvxopt_qp}[solver]

def svm_dual_soft_to_qp_kernel(K, y, C=1):
    n = K.shape[0]
    assert (len(y) == n)
        
    # Dual formulation, soft margin
    P = np.diag(y).dot(K).dot(np.diag(y))
    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    P += eps * np.eye(n)
    q = - np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    A = y[np.newaxis, :]
    A = A.astype('float')
    b = np.array([0.])
    return P, q, G, h, A, b

# SVM primal soft
class KernelSVM():
    def __init__(self, X, y, C=0.1, lam = 0.1, sigma = 4, tol = 1e-1, kernel=None):
        self.alpha = None
        self.X = X
        self.y = y
        self.w = None
        self.b = None
        self.C = C
        self.kernel = kernel
        self.lam = lam        
        self.sigma = sigma
        self.tol = tol
    
    def fit(self):
        
        X = self.X
        y = self.y
        C = self.C
        
        n, p = X.shape
        assert (len(y) == n)
        K = self.kernel(X, X)
        
        # Solve dual problem
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.C))
        
        
       # Compute support vectors and bias b
        sv = np.logical_and((self.alpha>self.tol), (self.C - self.alpha > self.tol))
        self.bias = y[sv] - K[sv].dot(self.alpha*y)
        self.bias =  self.bias.mean()

        self.support_vector_indices = np.nonzero(sv)[0]
        
        return self
        
        
        
    def predict(self, X, threshold):
#         y_pred = self.kernel(X, self.X_).dot(self.alphas* seld.y_)
        K_x = self.kernel(X, self.X)
        return np.where((K_x.dot(self.alpha * self.y) +  self.bias) >= threshold, 1, 0)
        
          
    def Accuracy_check(self,X, y, threshold=.5):
        return np.mean(self.predict(X, threshold)==y)


# # Cross Validation

X_train_ = pd.read_csv('./data/Xtr.csv', sep=',') #we use this dataset to train our model
Y_train_ = pd.read_csv('./data/Ytr.csv', sep=',') #we use this dataset to train our model
X_test_ = pd.read_csv('./data/Xte.csv', sep=',')

kfold=KFold(n_splits=5)

def spectrum_kernal(X_train, y, X_test, n=2, encoder=4, one_hot = True, normalise = False):
    
    d = {'A': 1, 'C':2, 'G':3, 'T':4}
    
    for i in range(0, 101-n+1, 1):
        X_train['seq_'+str(i)] = X_train.seq.apply(lambda x :x[i:i+n])
        X_test['seq_'+str(i)] = X_test.seq.apply(lambda x :x[i:i+n])
        
        X_train['seq_'+str(i)] = X_train['seq_'+str(i)].apply(lambda x : sum([d[x[ii]]*encoder**(ii+1) for ii in range(n)]))
        X_test['seq_'+str(i)] = X_test['seq_'+str(i)].apply(lambda x : sum([d[x[ii]]*encoder**(ii+1) for ii in range(n)]))
        
    X = X_train.drop(['seq', 'Id'], axis=1)
    X_t = X_test.drop(['seq', 'Id'], axis=1)
    y = Y_train.Bound

    if one_hot:
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')

        X_cross = X.values
        X_t = X_t.values
        
        enc = onehot_encoder.fit(X)
        X_cross = enc.transform(X)
        X_t_enc = enc.transform(X_t)
        
    elif normalise:
        scaler = MinMaxScaler()#MinMaxScaler() # StandardScaler()
        scaler.fit(X)
        
        X_cross = scaler.transform(X)
        X_t_enc = scaler.transform(X_t)
        
    else :
        
        X_cross = X.values
        X_t_enc = X_t.values
    
    y_cross = y.values
    
    return X_cross, y_cross, X_t_enc




X_cross, y_cross, X_t_enc = spectrum_kernal(X_train_, Y_train_, X_test_, n=1, encoder=8, one_hot = True, normalise = False)


sigma  = 4.119788517147901
kenel = rbf_kernel
lam = 1.2752298700618223e-14

    
models = {ksolveRR : 'k Ridge Reg'}

for model in models:
    accuracy = []
    for i, (train_index, validate_index) in enumerate(kfold.split(X_cross)):

        X_train, y_train = X_cross[train_index], y_cross[train_index]
        X_valid, y_valid = X_cross[validate_index], y_cross[validate_index]

        if models[model] == 'weigh Ridge Reg':
            sample_weights = np.random.rand(len(y_train))
            model_curr = model(X_train, y_train, lam = lam, sigma = sigma, sample_weights = sample_weights, kernel = kenel)
        elif models[model] == 'k Logistic Ridge Reg':
            model_curr = model(X_train, y_train, lam = lam, sigma = sigma, max_iter=500, tol = tol, kernel = kenel)
        
        elif models[model] == 'k Ridge Reg':
            model_curr = model(X_train, y_train, lam = lam, sigma = sigma, kernel = kenel)
        else:
            model_curr = model(X_train, y_train, C=c, lam = lam, sigma = sigma, tol= tol, kernel = kenel)
            
        model_curr.fit()

        accuracy.append(model_curr.Accuracy_check(X_valid, y_valid, threshold=0.5))
        print(f'accurracy fold {i}: {accuracy[i]}')
    
    print(f'\nAverage accuracy {models[model]} is : {np.mean(accuracy)}\n')
    
# # Predictions


model = ksolveRR(X_cross, y_cross, lam = lam, sigma = sigma, kernel = rbf_kernel)
model.fit()
y_pred = model.predict(X_t_enc, 0.5)


X = np.arange(1000).reshape(-1, 1)
sample = pd.DataFrame(data=X, columns=['Id'])
sample.head()


sample['Bound'] = y_pred


sample.to_csv('./ksolveRR_accuracy'+str(np.mean(accuracy))+'_cv_rbf_kernel_sigma_'+str(sigma)+'_lam_'+str(lam)+'.csv', index=False)

print("Predictions on test data saved in your current repository")





