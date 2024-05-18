#!/usr/bin/env python
# coding: utf-8

# In[150]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[151]:


# load the dataset

df=pd.read_csv('Iris.csv')


# In[152]:


df


# In[153]:


# Summarizing Dataset:

# The given dataset is about a particular species of flower (Iris-setosa,Iris-virginica)
#Iris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae.
# The dataset shows the measurements of the various flower parts such as sepal the outer covering and petal the actaul part of flower.
# Each rows shows the actual measure of each flower as per the species type.
# Our objective is to predict which flower species, does the data belong to by follwing steps of handling dataset.


# In[154]:


df.info()


# In[155]:


df.describe()


# In[156]:


len(df)


# In[157]:


df.dtypes


# In[158]:


# from the above we can see that there are 150 entries (rows) with min, maximum & mean values of each feature along with "std" values.


# In[159]:


# pre-processing the dataset


# In[160]:


# checking the null values

df.isnull().sum()


# In[161]:


df.isnull().sum().max


# In[162]:


# From the above it is clear that we do not have any null values in this dataset. so now we can proceed further.


# In[163]:


# We have a column "Id" with serial numbering for the entries in the dataset.

df['Id'].unique


# In[164]:


# We have to drop this column

df.drop('Id', axis=1, inplace=True)


# In[165]:


df


# In[166]:


df['Species'].unique()


# In[167]:


# There is one column which is in "object" data types, so we will hve to perform label encoding in this dataset


# In[168]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[169]:


label = LabelEncoder()
label


# In[170]:


# Column to be encoded
columns_to_encode = ['Species']


# In[171]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[172]:


#This is the final dataset with all integer values

df.head(5)


# In[173]:


df.dtypes


# In[174]:


df['Species'].unique()


# In[175]:


# Data Visualisation of the above dataset


# In[176]:


# Recalling our cleaned dataset

df


# In[223]:


sns.pairplot(df, diag_kind='hist')


# In[225]:


sns.pairplot(df, diag_kind='kde')


# In[178]:


# from the above chart we can see that there is co-relation between sepal features and petal features as per the type of species
# skewness is obsrved in the dataset for sepal and petal features.


# In[179]:


# Calculate correlation matrix for all the features.

corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[180]:


plt.figure(figsize=(15,8))
plt.title('Heatmap showing Correlation between all the features', fontsize=20)
sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f")


# In[181]:


# The above chart givees us clear picture about direct co-relation of sepal feature as per species types.

# Petal features vary because of numerous reason depending upon the upbringing of the flower in the nursery or in the field.


# In[182]:


#Segregating the Dataset into Input(x) and Output(y)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[183]:


x.shape, y.shape


# In[184]:


# Splitting the Dataset into Training and Testing Data


# In[185]:


# we import additional libraries
from sklearn.model_selection import train_test_split
print('importing is done')


# In[186]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[187]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[188]:


# Loading the Models


# In[189]:


# We import additional libraries

from sklearn.svm import SVC
print('importing is done')


# In[190]:


svc = SVC()


# In[191]:


# Training the Models

svc.fit(x_train, y_train)


# In[192]:


# Predicting the Result Using the Trained Models

y_pred = svc.predict(x_test)


# In[193]:


y_pred


# In[194]:


y_test


# In[195]:


# Calculating the Accuracy of the Trained Models


# In[196]:


# we import additional library for accuracy
from sklearn.metrics import accuracy_score, classification_report
print('importing is done')


# In[214]:


# Now we check for accuracy of test data vs training data
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[212]:


np.round(accuracy,2)*100


# In[198]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[199]:


# Results:
# The above result shows us that our model prediction is 100% accurate with 80/20 ratio.
# This can be due to the small size of the dataset (or it can be because of over fitting).
# So we try to change the random state for confirmation


# In[200]:


# If we keep split as 70/30
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=70)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
np.round(accuracy,2)*100


# In[201]:


# Still the results are same.


# In[202]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[203]:


x_test[2]


# In[204]:


x_test[2].shape


# In[205]:


x_test[2].reshape(1,4)


# In[206]:


x_test[2].reshape(1,4).shape


# In[207]:


svc.predict(x_test[2].reshape(1,4))


# In[208]:


y_test[2]


# In[209]:


# conclusion: Our model is well trained to execute 100% prediction each time.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




