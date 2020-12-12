#!/usr/bin/env python
# coding: utf-8

# # Nikhil Madane
# Task-1
# Linear Regression with Python Scikit Learn
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. 
# We will start with simple linear regression involving two variables.
# 
# Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# This is a simple linear regression task as it involves just two variables.

# In[6]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


data=pd.read_csv("Student_Scores_Dataset.csv")


# data.head()

# In[20]:


data.shape


# In[21]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[22]:



# Plotting the distribution of scores
plt.scatter(data['Hours'], data['Scores'],color='r')  
plt.title('Hours vs Score')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[24]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[26]:


model.fit(X_train,y_train)
print("Training complete")


# In[27]:


# Plotting the regression line
line = model.coef_*X+model.intercept_

# Plotting for the test data
plt.scatter(X, y,color='g')
plt.plot(X, line,color='r');
plt.show()


# In[ ]:


#Now that we have trained our algorithm, it's time to make some predictions.


# In[28]:


print(X_test) # Testing data - In Hours
y_pred = model.predict(X_test) # Predicting the scores


# In[29]:


y_test


# In[30]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[31]:


# You can also test with your own data
hours = [[9.25]]
own_pred = model.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[32]:


pred = model.predict(data[['Hours']])
pred


# In[33]:


from sklearn.metrics import r2_score
r2_score(data.Scores,pred)


# In[ ]:


#Evaluating the model
#The final step is to evaluate the performance of algorithm.
#This step is particularly important to compare how well different algorithms perform on a particular dataset.
#For simplicity here, we have chosen the mean square error. There are many such metrics.


# In[34]:


from sklearn.metrics import r2_score
accuracy_score = r2_score(y_test,y_pred)
print("The model has accuracy of {}%".format(accuracy_score*100))


# In[35]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




