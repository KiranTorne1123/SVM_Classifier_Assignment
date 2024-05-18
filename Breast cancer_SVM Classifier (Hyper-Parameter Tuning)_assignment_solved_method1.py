#!/usr/bin/env python
# coding: utf-8

# In[169]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[170]:


# load the dataset

df=pd.read_csv('data.csv')


# In[171]:


df


# In[172]:


# Summarizing Dataset:

# This dataset is all about various patients readings during test done for predicting the type of Breast Cancer.
# Many variables are taken into consideration such as radius, preimeter, area, concavity etc.
# This dataset has total of 569 patients readings summarizing to prediction of their respective Breast Cancer status.
# Our objective is to study these variables and liekwise do predictions about the positiveness of a particular type of cancer.


# In[173]:


df.info()


# In[174]:


df.describe().transpose()


# In[175]:


len(df)


# In[176]:


df.dtypes


# In[177]:


# From the above we can cofirm that this dataset has total 569 patients details.
# It also shows that many features are involved inorder to predict the cancer type accurately
# Also, except 'diagnosis' column rest all are in numeric data types.


# In[178]:


# Pre-processing the dataset


# In[179]:


# Checking the null values

df.isnull().sum()


# In[180]:


df.isnull().sum().max


# In[181]:


# From the above we can see that the column'Unnamed: 32' has 569 null values.
# We will have to drop this entire column as it serves no purpose in this dataset.
df.drop('Unnamed: 32', axis=1, inplace=True)


# In[182]:


# We have a column "Id" with serial numbering for the entries in the dataset.
# We will drop the 'Id' column since its not required for our perspective in determining the cancer types.
df['id'].unique


# In[183]:


# We have to drop this column

df.drop('id', axis=1, inplace=True)


# In[184]:


# After removing our dataset becomes better but still we need to clean it for better processing in further steps.
df


# In[185]:


df['diagnosis'].unique()


# In[186]:


# The above column is in "object" data types, so we will hve to perform label encoding in this dataset


# In[187]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[188]:


label = LabelEncoder()
label


# In[189]:


# Column to be encoded
columns_to_encode = ['diagnosis']


# In[190]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[191]:


#This is the final dataset with all integer values

df.head(5).transpose()


# In[192]:


df.dtypes


# In[193]:


df['diagnosis'].unique()


# In[194]:


# This is our clean dataset, which we will be using for further analysis and processing.
df


# In[195]:


# Data Visualisation of the above dataset


# In[196]:


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


# In[197]:


# Calculate correlation matrix for all the features.

corr = df.corr()
plt.figure(figsize=(20,12))
plt.title('Heatmap showing Correlation between all the features', fontsize=20)
sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f")


# In[198]:


# From the above heatmap we can see that there is a direct co-relation of diagnosis with fractal dimensions worst feature.
# Note: The actual fractal dimensions of the breast tissue samples are 1.5448 for the normal, 1.6126 for stage I, 1.6631 for stage II, and 1.7284 for stage III cancer.
# Moreover, most of the feature show some co-relation in estimating that the patient has breast cancer


# In[199]:


# Segregating the Dataset into Input(x) and Output(y)


# In[240]:


x = df.drop(columns=['diagnosis']).values
y =  df['diagnosis'].values


# In[241]:


x


# In[236]:


y


# In[242]:


x.shape, y.shape


# In[243]:


# Splitting the Dataset into Training and Testing Data


# In[244]:


# we import additional libraries
from sklearn.model_selection import train_test_split
print('importing is done')


# In[245]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[246]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[247]:


# Loading the Models


# In[248]:


# We import additional libraries

from sklearn.svm import SVC
print('importing is done')


# In[249]:


svc = SVC()


# In[250]:


# Training the Models

svc.fit(x_train,y_train)


# In[251]:


# Predicting the Result Using the Trained Models

y_pred = svc.predict(x_test)


# In[252]:


y_pred


# In[253]:


y_test


# In[254]:


# Calculating the Accuracy of the Trained Models


# In[255]:


# we import additional library for accuracy
from sklearn.metrics import accuracy_score, classification_report
print('importing is done')


# In[256]:


# Now we check for accuracy of test data vs training data
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[257]:


np.round(accuracy,2)*100


# In[258]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[259]:


# Results:
# The above result shows us that our model prediction is 95% accurate with 80/20 ratio.
# we can try to change the random state for finding options of better accuracy


# In[260]:


# If we keep split as 70/30
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=70)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
np.round(accuracy,2)*100


# In[261]:


# Accuracy with 80/20 split is by far the best one.


# In[262]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[263]:


x_test[4]


# In[264]:


x_test[4].shape


# In[265]:


x_test[4].reshape(1,30).shape


# In[266]:


svc.predict(x_test[4].reshape(1,30))


# In[267]:


y_test[4]


# In[268]:


# conclusion: Our model is well trained to execute 95 % accurate prediction each time.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




