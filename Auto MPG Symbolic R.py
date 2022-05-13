#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[2]:


get_ipython().run_line_magic('pylab', 'inline')
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[3]:


random.seed(10)


# In[4]:


auto = pd.read_csv("Documents/auto-mpg.csv")


# In[5]:


X = auto[["cylinders","displacement","horsepower","weight","acceleration","model_year","origin"]]


# In[6]:


Y = auto[auto.columns[-1]]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[8]:


est_gp = SymbolicRegressor(population_size=5000,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)


# In[9]:


y_train = y_train.values


# In[10]:


y_train = y_train.ravel()


# In[11]:


est_gp.fit(X_train,y_train)


# In[12]:


est_gp.score(X_test,y_test)


# In[13]:


print(est_gp._program)


# In[18]:


scores = cross_val_score(est_gp, X, Y, cv=3)


# In[19]:


sum(scores) / len(scores)


# In[ ]:




