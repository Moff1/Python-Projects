#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install imbalanced-learn')


# In[1]:


import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

allData = pd.read_csv("creditcard.csv")
#allData[["Class"]] = allData[["Class"]].astype(str)



##nonFraud = allData[allData["Class"] == 0]
from typing import TypeVar, Callable

##noFraudTrain = nonFraud.sample(frac=.00173)

##fraud = allData[allData["Class"]==1]


##trainingData = noFraudTrain.append(fraud)




X = allData.loc[:, allData.columns!= 'Class']
y = allData["Class"]
counter = Counter(y)

oversample = SMOTE()
newX, newY = oversample.fit_resample(X,y)

counterY = Counter(newY)


allTrueData = pd.concat([newX,newY], axis = 1)
fraud = allTrueData[allTrueData["Class"]==1]
nonFraud = allTrueData[allTrueData["Class"]==0]

fraudTest, fraudTrain = train_test_split(fraud, test_size = 0.75)
nonFraudTest, nonFraudTrain = train_test_split(nonFraud, test_size = 0.75)

trainData = fraudTrain.append(nonFraudTrain)
testData = fraudTest.append(nonFraudTest)




print(fraudTest)
print(fraudTrain)
print(nonFraudTest)
print(nonFraudTrain)

print(trainData)
print(testData)



# In[2]:


y = trainData["Class"]
trainData1 = trainData.drop("Class", axis = 1)

yt = testData["Class"]
testData = testData.drop("Class",axis = 1)


#print(yt)
#print(TestingC)


# In[3]:


#Classifiers and Models (k = 1)

model = KNeighborsClassifier(n_neighbors=1)
y3 = model.fit(trainData1,y)

expected = y
predicted = model.predict(trainData1)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[4]:


#KNN Models (k = 10)
#y = trainData["Class"]
#trainData1 = traindata.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=10)
y3 = model.fit(trainData1,y)

expected = y
predicted = model.predict(trainData1)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[5]:


#KNN Models (k = 100)
y = trainData["Class"]
trainData1 = trainData.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=100)
y3 = model.fit(trainData1,y)

expected = y
predicted = model.predict(trainData1)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[6]:


#KNN Models (k = 1000)
y = trainData["Class"]
trainData1 = trainData.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=1000)
y3 = model.fit(trainData1,y)

expected = y
predicted = model.predict(trainData1)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[9]:


#Testing Data
#Classifiers and Models (k = 1)
#y1 = testData["Class"]
#testData1 = testData.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=1)
y3 = model.fit(testData,yt)

expected = yt
predicted = model.predict(testData)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[10]:


#Testing Data
#Classifiers and Models (k = 10)
#y = testData["Class"]
#testData1 = testData.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=10)
y3 = model.fit(testData,yt)

expected = yt
predicted = model.predict(testData)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[11]:


#Testing Data
#Classifiers and Models (k = 100)
#y = testData["Class"]
#testData1 = testData.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=100)
y3 = model.fit(testData,yt)

expected = yt
predicted = model.predict(testData)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[12]:


#Testing Data
#Classifiers and Models (k = 1000)
#y = testData["Class"]
#testData1 = testData.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=1000)
y3 = model.fit(testData,yt)

expected = yt
predicted = model.predict(testData)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[ ]:




