#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[4]:


# load the dataset

df=pd.read_csv('data.csv')


# In[5]:


df


# In[6]:


# Summarizing Dataset:

# This dataset is all about various patients readings during test done for predicting the type of Breast Cancer.
# Many variables are taken into consideration such as radius, preimeter, area, concavity etc.
# This dataset has total of 569 patients readings summarizing to prediction of their respective Breast Cancer status.
# Our objective is to study these variables and liekwise do predictions about the positiveness of a particular type of cancer.


# In[7]:


df.info()


# In[8]:


df.describe().transpose()


# In[9]:


len(df)


# In[10]:


df.dtypes


# In[11]:


# From the above we can cofirm that this dataset has total 569 patients details.
# It also shows that many features are involved inorder to predict the cancer type accurately
# Also, except 'diagnosis' column rest all are in numeric data types.


# In[12]:


# Pre-processing the dataset


# In[13]:


# Checking the null values

df.isnull().sum()


# In[14]:


df.isnull().sum().max


# In[15]:


# From the above we can see that the column'Unnamed: 32' has 569 null values.
# We will have to drop this entire column as it serves no purpose in this dataset.
df.drop('Unnamed: 32', axis=1, inplace=True)


# In[16]:


# We have a column "Id" with serial numbering for the entries in the dataset.
# We will drop the 'Id' column since its not required for our perspective in determining the cancer types.
df['id'].unique


# In[17]:


# We have to drop this column

df.drop('id', axis=1, inplace=True)


# In[18]:


# After removing our dataset becomes better but still we need to clean it for better processing in further steps.
df


# In[19]:


df['diagnosis'].unique()


# In[20]:


# The above column is in "object" data types, so we will hve to perform label encoding in this dataset


# In[21]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[22]:


label = LabelEncoder()
label


# In[23]:


# Column to be encoded
columns_to_encode = ['diagnosis']


# In[24]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[25]:


#This is the final dataset with all integer values

df.head(5).transpose()


# In[26]:


df.dtypes


# In[27]:


df['diagnosis'].unique()


# In[28]:


# This is our clean dataset, which we will be using for further analysis and processing.
df


# In[29]:


# Data Visualisation of the above dataset


# In[71]:


df.columns.unique


# In[83]:


len(df.columns)


# In[84]:


# Plot scatterplots for each feature against the target column

target_column = 'diagnosis'
features = df.drop(columns=[target_column]).columns
num_cols = 3  # Number of columns for subplots
num_rows = (len(features) - 1) // num_cols + 1
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features, start=1):
    plt.subplot(num_rows, num_cols, i)
    plt.scatter(df[feature], df[target_column], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel(target_column)

plt.tight_layout()
plt.show()


# In[30]:


# Calculate correlation matrix for all the features.

corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[31]:


plt.figure(figsize=(20,12))
plt.title('Heatmap showing Correlation between all the features', fontsize=20)
sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f")


# In[ ]:


# From the above heatmap we can see that there is a direct co-relation of diagnosis with fractal dimensions
# Note: The actual fractal dimensions of the breast tissue samples are 1.5448 for the normal, 1.6126 for stage I, 1.6631 for stage II, and 1.7284 for stage III cancer.


# In[32]:


# Segregating the Dataset into Input(x) and Output(y)


# In[87]:


x = df.drop(columns=['diagnosis']).values
y =  df['diagnosis'].values


# In[88]:


x.shape, y.shape


# In[89]:


# Splitting the Dataset into Training and Testing Data


# In[90]:


# we import additional libraries
from sklearn.model_selection import train_test_split
print('importing is done')


# In[91]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[92]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[93]:


# Loading the Models


# In[94]:


# We import additional libraries

from sklearn.svm import SVR
print('importing is done')


# In[95]:


svr = SVR()


# In[96]:


# Training the Models

svr.fit(x_train, y_train)


# In[97]:


# Predicting the Result Using the Trained Models

y_pred = svr.predict(x_test)


# In[98]:


y_pred


# In[99]:


y_test


# In[100]:


# Calculating the Accuracy of the Trained Models


# In[101]:


# we import additional library for accuracy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('importing is done')


# In[102]:


mse = mean_squared_error(y_test, y_pred)


# In[103]:


print(mse)


# In[104]:


r2 = r2_score(y_test, y_pred)


# In[105]:


print(r2)


# In[106]:


# If we keep split as 70/30
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=70)
svr.fit(x_train, y_train)
y_pred = svr.predict(x_test)
print(mse)
print(r2)


# In[107]:


# From the above we can see that MSE and R2 score both gives good result compiling to the fact that our model works well.


# In[108]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[109]:


x_test[4]


# In[110]:


x_test[4].shape


# In[111]:


x_test[4].reshape(1,30)


# In[112]:


x_test[4].reshape(1,30).shape


# In[113]:


svr.predict(x_test[4].reshape(1,30))


# In[114]:


y_test[4]


# In[115]:


# conclusion: Our model is well trained to execute best possible accuracy prediction each time.


# In[ ]:




