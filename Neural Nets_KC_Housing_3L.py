#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import keras as kr
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers import Dense
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
#import random
#random.seed(10)


# In[3]:


KCHousingData = pd.read_csv("DataSetsForNeuralNets/kc_house_data.csv")
KCHousingData


# In[4]:


KCHousingArray = KCHousingData.values
KCHousingArray


# In[5]:


X = KCHousingArray[:,1:11]
Y = KCHousingArray[:,0]


# In[6]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = Y.reshape(-1,1)
Y = scaler.fit_transform(Y)


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)


# In[8]:


train_features = X_train.copy()
test_features = X_test.copy()


# In[9]:


#normalizer = tf.keras.layers.Normalization(axis=-1)
#normalizer.adapt(np.array(train_features))


# In[10]:


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    r2 = 1 - SS_res/(SS_tot + K.epsilon())
    return 1 - (((1 - r2) * (X_train.shape[0] - 1 ))/ (X_train.shape[0] - X_train.shape[1] - 1)) 


# In[11]:


total_k = X_train.shape[1]
n = X_train.shape[0]
ks = list(range(1,total_k + 1))
adjr2 = np.array([])
cvr2 = np.array([])
i_added_array = np.array([])
i_not_added_array = np.array([list(range(total_k))])
i_not_added_array = i_not_added_array.reshape(-1)


adjr2_to_add = -1000
cvr2_to_add = -1000
i_added = -1

for k in range(total_k):
    adjr2_to_add = -1000
    cvr2_to_add = -1000
    i_added = -1
    for i in range(total_k):
        i_this_it = np.array([])
        if np.any(i_not_added_array == i):
            i_this_it = np.append(i_added_array, i)
            i_this_it = i_this_it.astype(int)
            print(i_this_it)
            model = Sequential([
                Dense(64, activation = "relu"),
                Dense(64, activation = "relu"),
                Dense(1, activation = "relu")
          ])
            model.compile(loss='mean_absolute_error',
                optimizer = tf.keras.optimizers.Adam(0.1),
                metrics = ['mse', coeff_determination])
            model.fit(X_train[:,i_this_it],Y_train,batch_size=32, epochs = 100,verbose = 0)
            if model.evaluate(X_test[:,i_this_it], Y_test)[2] * 100 > adjr2_to_add:
                adjr2_to_add = model.evaluate(X_test[:,i_this_it], Y_test)[2] * 100
                i_added = i

    adjr2 = np.append(adjr2, adjr2_to_add )
    i_added_array = np.append(i_added_array, i_added)
    i_added_array = i_added_array.astype(int)
    i_to_remove = np.array([i_added])
    i_not_added_array = np.setdiff1d(i_not_added_array, i_added_array)

    kfold = KFold(n_splits=5)
    cvscores = []
    for train, test in kfold.split(X, Y):

        model.fit(X[train][:,i_added_array], Y[train], epochs=100, batch_size=32, verbose=0)
        scores = model.evaluate(X[test][:,i_added_array], Y[test], verbose=0)
        cvscores.append(scores[2] * 100)
    cvr2_to_add = np.mean(cvscores)
    cvr2 = np.append(cvr2, cvr2_to_add)
    
plot.plot(ks, adjr2, label = "AdjR2")
plot.plot(ks, cvr2, label = "CVR2")
plot.legend()
plot.show()


# In[12]:


total_k = X_train.shape[1]
n = X_train.shape[0]
ks = list(range(1,total_k + 1))
adjr2 = np.array([])
cvr2 = np.array([])
i_added_array = np.array([])
i_not_added_array = np.array([list(range(total_k))])
i_not_added_array = i_not_added_array.reshape(-1)


adjr2_to_add = -1000
cvr2_to_add = -1000
i_added = -1

for k in range(total_k):
    adjr2_to_add = -1000
    cvr2_to_add = -1000
    i_added = -1
    for i in range(total_k):
        i_this_it = np.array([])
        if np.any(i_not_added_array == i):
            i_this_it = np.append(i_added_array, i)
            i_this_it = i_this_it.astype(int)
            print(i_this_it)
            model = Sequential([
                Dense(64, activation = "sigmoid"),
                Dense(64, activation = "sigmoid"),
                Dense(1, activation = "sigmoid")
          ])
            model.compile(loss='mean_absolute_error',
                optimizer = tf.keras.optimizers.Adam(0.1),
                metrics = ['mse', coeff_determination])
            model.fit(X_train[:,i_this_it],Y_train,batch_size=32, epochs = 100,verbose = 0)
            if model.evaluate(X_test[:,i_this_it], Y_test)[2] * 100 > adjr2_to_add:
                adjr2_to_add = model.evaluate(X_test[:,i_this_it], Y_test)[2] * 100
                i_added = i

    adjr2 = np.append(adjr2, adjr2_to_add )
    i_added_array = np.append(i_added_array, i_added)
    i_added_array = i_added_array.astype(int)
    i_to_remove = np.array([i_added])
    i_not_added_array = np.setdiff1d(i_not_added_array, i_added_array)

    kfold = KFold(n_splits=5)
    cvscores = []
    for train, test in kfold.split(X, Y):

        model.fit(X[train][:,i_added_array], Y[train], epochs=100, batch_size=32, verbose=0)
        scores = model.evaluate(X[test][:,i_added_array], Y[test], verbose=0)
        cvscores.append(scores[2] * 100)
    cvr2_to_add = np.mean(cvscores)
    cvr2 = np.append(cvr2, cvr2_to_add)
    
plot.plot(ks, adjr2, label = "AdjR2")
plot.plot(ks, cvr2, label = "CVR2")
plot.legend()
plot.show()


# In[13]:


total_k = X_train.shape[1]
n = X_train.shape[0]
ks = list(range(1,total_k + 1))
adjr2 = np.array([])
cvr2 = np.array([])
i_added_array = np.array([])
i_not_added_array = np.array([list(range(total_k))])
i_not_added_array = i_not_added_array.reshape(-1)


adjr2_to_add = -1000
cvr2_to_add = -1000
i_added = -1

for k in range(total_k):
    adjr2_to_add = -1000
    cvr2_to_add = -1000
    i_added = -1
    for i in range(total_k):
        i_this_it = np.array([])
        if np.any(i_not_added_array == i):
            i_this_it = np.append(i_added_array, i)
            i_this_it = i_this_it.astype(int)
            print(i_this_it)
            model = Sequential([
                Dense(64, activation = "tanh"),
                Dense(64, activation = "tanh"),
                Dense(1, activation = "tanh")
          ])
            model.compile(loss='mean_absolute_error',
                optimizer = tf.keras.optimizers.Adam(0.1),
                metrics = ['mse', coeff_determination])
            model.fit(X_train[:,i_this_it],Y_train,batch_size=32, epochs = 100,verbose = 0)
            if model.evaluate(X_test[:,i_this_it], Y_test)[2] * 100 > adjr2_to_add:
                adjr2_to_add = model.evaluate(X_test[:,i_this_it], Y_test)[2] * 100
                i_added = i

    adjr2 = np.append(adjr2, adjr2_to_add )
    i_added_array = np.append(i_added_array, i_added)
    i_added_array = i_added_array.astype(int)
    i_to_remove = np.array([i_added])
    i_not_added_array = np.setdiff1d(i_not_added_array, i_added_array)

    kfold = KFold(n_splits=5)
    cvscores = []
    for train, test in kfold.split(X, Y):

        model.fit(X[train][:,i_added_array], Y[train], epochs=100, batch_size=32, verbose=0)
        scores = model.evaluate(X[test][:,i_added_array], Y[test], verbose=0)
        cvscores.append(scores[2] * 100)
    cvr2_to_add = np.mean(cvscores)
    cvr2 = np.append(cvr2, cvr2_to_add)
    
plot.plot(ks, adjr2, label = "AdjR2")
plot.plot(ks, cvr2, label = "CVR2")
plot.legend()
plot.show()


# In[ ]:




