#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


iris=pd.read_csv('iris.csv')
iris.head()


# In[3]:


iris.shape


# In[4]:


iris.columns


# In[5]:


iris.describe


# In[6]:


iris.dtypes


# In[7]:


# here species is object  and convert into numbers
iris.replace('Iris-setosa',0,inplace=True)
iris.replace('Iris-versicolor',1,inplace=True)
iris.replace('Iris-virginica',2,inplace=True)


# In[27]:


x=iris[['sepal_length','sepal_width','petal_length','petal_width']]
y=iris['species']


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


lr= LinearRegression()


# In[30]:


lr.fit(x,y)


# In[31]:


y_predict=lr.predict(x)
y_predict[:5]


# In[32]:


x[:5]


# In[33]:


from matplotlib import pyplot as plt


# In[34]:


x.head(2)


# In[35]:


y.head(2)


# In[37]:


x1=iris['sepal_length']
y1=iris['species']


# In[39]:


plt.scatter(x1,y1,color='hotpink',label='Actual Values')
plt.scatter(x1,y_predict,color='g',label='Predicted Values')
plt.legend()
plt.show()


# In[40]:


from sklearn.metrics import r2_score


# In[41]:


r2_score(y,y_predict)


# By using the LinearRegression we are getting a score of 93%
