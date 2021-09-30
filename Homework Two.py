#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# first fill in your name
first_name = "Joseph"
last_name  = "Moffatt"

print("****************************************************************")
print("CSCI 3360 Homework 2")
print(f"completed by {first_name} {last_name}")
print(f"""
I, {first_name} {last_name}, certify that the following code
represents my own work. I have neither received nor given 
inappropriate assistance. I have not consulted with another
individual, other than a member of the teaching staff, regarding
the specific problems in this homework. I recognize that any 
unauthorized assistance or plagiarism will be handled in 
accordance with the University of Georgia's Academic Honesty
Policy and the policies of this course.
""")
print("****************************************************************")


# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
import numpy
import statistics
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from numpy import percentile
from statistics import mean

# 'r' means read-only, it's assumed if you leave it out
Iris = open('iris.txt') 


#Question1 - The data set has 150 rows/ instances.
#Question2 - The dataset has 4 columns: sepal length(cm), sepal width (cm),
#petal length (cm), petal width (cm), and the 'class' of the iris.

IrisData = pd.read_csv('iris.txt', sep = ',', names = ["Sepal Length","Sepal Width","Petal Length","Petal Width","Class of Iris"])


# In[4]:


IrisDataMIN = IrisData.min()
print(IrisDataMIN)


# In[ ]:


IrisDataMAX = IrisData.max()
print(IrisDataMAX)


# In[ ]:


Median = IrisData.quantile(.5)
print(Median)


# In[5]:


print(IrisData.mean())


# In[ ]:


print(IrisData.var())


# In[ ]:


print(IrisData.std())


# In[6]:


IrisData.corr()
#Based on the correlation matrix below Petal length and petal width have the strongest postive correlation, while
#Sepal Length and Petal Length have the strongest negative correlation (means while one variable increases the other decreases and vice versa.)
#Correlations near 0 essentially means there is no correlation between those two particular variables. 


# In[31]:


IrisData.cov()


# In[2]:


#Iris-setosa
IrisDataIS = IrisData[IrisData['Class of Iris'] == 'Iris-setosa']

#Iris-versicolor
IrisDataIVer = IrisData[IrisData['Class of Iris'] == 'Iris-versicolor']

#Iris-virginica
IrisDataIVir = IrisData[IrisData['Class of Iris'] == 'Iris-virginica']


# In[5]:


#HISTOGRAMS
##Iris-setosa - Sepal Lengths

IrisDataISSL = IrisDataIS['Sepal Length']

histogram = plt.hist(IrisDataISSL)

plt.title("Iris-Setosa Sepal Length")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:



##Iris-Versicolor - Sepal Lengths

IrisDataIVSL = IrisDataIS['Sepal Length']

histogram = plt.hist(IrisDataIVSL)

plt.title("Iris-Versicolor Sepal Length")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


##Iris-Virginca - Sepal Lengths

IrisDataIVirSL = IrisDataIVir['Sepal Length']

histogram = plt.hist(IrisDataIVirSL)

plt.title("Iris-virginica Sepal Length")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15])


# In[4]:


#Iris-setosa - Sepal Width
IrisDataISSW = IrisDataIS['Sepal Width']

histogram = plt.hist(IrisDataISSL)

plt.title("Iris-Setosa Sepal Width")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


#Iris-Versicolor - Sepal Width

IrisDataIVSW = IrisDataIS['Sepal Width']

histogram = plt.hist(IrisDataIVSW)

plt.title("Iris-Versicolor Sepal Width")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


##Iris-Virginca - Sepal Width

IrisDataIVirSW = IrisDataIVir['Sepal Width']

histogram = plt.hist(IrisDataIVirSL)

plt.title("Iris-virginica Sepal Width")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


#Petal Length

##Iris-setosa - Petal Lengths

IrisDataISPL = IrisDataIS['Petal Length']

histogram = plt.hist(IrisDataISPL)

plt.title("Iris-Setosa Petal Length")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


##Iris-Versicolor - Petal Lengths

IrisDataIVPL = IrisDataIS['Petal Length']

histogram = plt.hist(IrisDataIVPL)

plt.title("Iris-Versicolor Petal Length")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


##Iris-Virginca - Petal Lengths

IrisDataIVirPL = IrisDataIVir['Petal Length']

histogram = plt.hist(IrisDataIVirPL)

plt.title("Iris-virginica Petal Length")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


#Petal Width

IrisDataISPW = IrisDataIS['Petal Width']

histogram = plt.hist(IrisDataISPW)

plt.title("Iris-Setosa Petal Width")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


##Iris-Versicolor - Petal Width

IrisDataIVPW = IrisDataIS['Petal Width']

histogram = plt.hist(IrisDataIVPW)

plt.title("Iris-Versicolor Petal Width")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[ ]:


##Iris-Virginca - Petal Width

IrisDataIVirPW = IrisDataIVir['Petal Width']

histogram = plt.hist(IrisDataIVirPW)

plt.title("Iris-virginica Petal Width")    
plt.ylabel("Frequency") 
plt.xlabel("Length") 
plt.axis([4, 6, 0, 15]) 


# In[7]:


#Iris-setosa

IrisDataISMIN = IrisDataIS.min()
print(IrisDataISMIN)


# In[6]:


IrisDataISMAX = IrisDataIS.max()
print(IrisDataISMAX)


# In[ ]:


Median = IrisDataIS.quantile(.5)
print(Median)


# In[8]:


Mean = IrisDataIS.mean()
print(Mean)


# In[9]:


Var = IrisDataIS.var()
print(Var)


# In[10]:


STD = IrisDataIS.std()
print(STD)


# In[10]:


#Iris-versicolor

IrisDataIVerMIN = IrisDataIVer.min()
print(IrisDataIVerMIN)

IrisDataIVerMAX = IrisDataIVer.max()
print(IrisDataIVerMAX)

Median = IrisDataIVer.quantile(.5)
print(Median)

Mean = IrisDataIVer.mean()
print(Mean)

Var = IrisDataIVer.var()
print(Var)

STD = IrisDataIVer.std()
print(STD)


# In[ ]:


#Iris-virginica

IrisDataIVirMIN = IrisDataIVir.min()
print(IrisDataIVerMIN)

IrisDataIVirMAX = IrisDataIVir.max()
print(IrisDataIVerMAX)

Median = IrisDataIVir.quantile(.5)
print(Median)

Mean = IrisDataIVir.mean()
print(Mean)

Var = IrisDataIVir.var()
print(Var)

STD = IrisDataIVir.std()
print(STD)


# In[4]:


IrisDataIS.corr()


# In[ ]:


IrisDataIS.cov()


# In[5]:


IrisDataIVer.corr()


# In[ ]:


IrisDataIVer.cov()


# In[ ]:


IrisDataIVir.cov() 


# In[6]:


IrisDataIVir.corr()


# In[ ]:


#9 -  
#10 - 

