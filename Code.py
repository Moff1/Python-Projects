
import imblearn
import numpy
import statistics
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import imblearn
from typing import TypeVar, Callable
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

allData = pd.read_csv("creditcard.csv")
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

#Classifiers and Models (k = 1)
y = trainData["Class"]
trainData1 = trainData.drop("Class", axis = 1)
model = KNeighborsClassifier(n_neighbors=1)
y3 = model.fit(trainData1,y)

expected = y
predicted = model.predict(trainData1)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))