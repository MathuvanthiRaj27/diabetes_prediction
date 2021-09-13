#!/usr/bin/env python
# coding: utf-8

# In[5]:


''' Importing the required libraries'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[6]:


''' Reading the csv file'''

data = pd.read_csv("diabetes2.csv") 
data.head()


# In[7]:


''' Printing the data types of the variables '''

data.dtypes


# In[8]:


''' Printing the shape of the dataset'''

data.shape


# In[9]:


''' Checking for null values '''

print(data.isna().sum())


# In[10]:


''' Printing the unique values in each columns '''

for i in data.columns:
    print(i + " has " + str(data[i].nunique())+" unique values")


# In[11]:


x = data.drop("Outcome",axis=1)  # Excluded the dependent variable and store the remaining to x
y = data["Outcome"]              # 'Outcome' column is dependent variable

''' Spliting the dataset for training and testing '''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 2)


# In[12]:


''' Initializing and fitting the model '''

model = RandomForestClassifier(n_estimators=100)


# In[13]:


model.fit(x_train,y_train)


# In[14]:


pr= model.predict(x_test)
print("Accuracy Score = {}%".format(round(accuracy_score(y_test, pr)*100,2)))


# In[15]:


pickle.dump(model,open("Diabetes_Prediction.pkl","wb"))


# In[ ]:




