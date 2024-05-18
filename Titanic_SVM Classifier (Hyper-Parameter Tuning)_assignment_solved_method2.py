#!/usr/bin/env python
# coding: utf-8

# In[346]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[347]:


# load the dataset

df1=pd.read_csv('gender_submission.csv')
df2=pd.read_csv('train.csv')
df3=pd.read_csv('test.csv')
print('loading is done')


# In[348]:


df1


# In[349]:


# Summarizing the dataset.
# Dataset "df1" shows only the passenger id along with their survival/non survival entries.
# Survival as "1" and dead is "0"


# In[350]:


df2


# In[351]:


# Summarizing the dataset.
# The above dataset 'df2' shows information of the legendary titanic ship passengers along with their respective details.
# Rich passengers have been charged highest fare and poor ones are with the least fare for the tickets class they bought that time.
# Also, the dataset shows their individual names along with gender and relevant ticket they owned during their travel.
# Finally, the dataset shows the people who have survived/succumbed during the crash of this legendary ship.


# In[352]:


df3


# In[353]:


# Summarizing the dataset.
# Dataset "df3" again shows all the passenger details except survival data in it.


# In[354]:


# Pre-processing the dataset


# In[355]:


# Before we proceed further we will have to merge the 2 datasets: "df1" & "df3"

# Merge the two datasets based on 'PassengerId'
df4= pd.merge(df1, df3, on='PassengerId', how='inner')


# In[356]:


df4


# In[357]:


# Merge the two datasets 'df2' and 'df4', again to form one single dataset to allow us to proceed further.
df = pd.concat([df2, df4])


# In[358]:


df


# In[359]:


df.info()


# In[360]:


df.describe()


# In[361]:


len(df)


# In[362]:


df.dtypes


# In[363]:


# Here we have 5 columns with data as object type. we will check if there is any null values and then proceed.


# In[364]:


# Now we check for null values

df.isnull().sum()


# In[365]:


# we have found 263 entries in Age column, 2 entries in Embarked column and 1014 entries in Cabin column as null values.


# In[366]:


# we drop both the columns with null values.

df.dropna(axis=1, inplace=True)


# In[367]:


df


# In[368]:


# now we will work upon changing the columns from object data types to integer
# we can see column Name, sex and ticket is in object type so we will use label encoding on it.


# In[369]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[370]:


label = LabelEncoder()
label


# In[371]:


# Column to be encoded
columns_to_encode = ['Name','Sex','Ticket']


# In[372]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[373]:


#This is the final dataset with all integer values

df


# In[374]:


#This is the final dataset with all integer values


# In[375]:


df.info()


# In[376]:


df.isnull().sum().max


# In[377]:


# From the above all codes we were able to transform df dataset into integer types which does not have any null values.
# The above dataset we will use as our final form to process further.


# In[378]:


# Data Visualisation of the above dataset


# In[379]:


# A----BAR PLOTS

# Counting the number of survivors
survival_counts = df['Survived'].value_counts()

# Plotting the bar chart
plt.bar(survival_counts.index, survival_counts.values)

# Adding labels and title
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Number of Survivors')

# Adding annotations
for i, count in enumerate(survival_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

# Showing the plot
plt.show()


# In[380]:


# So the total count happens to be as follow
total_passengers_onboard = (815+494)
print("Total_passengers_onboard:", total_passengers_onboard)
total_death = 494
print("Total_Death:", total_death)
total_survived = 815
print("Total_Survived:", total_survived)


# In[381]:


# From the above chart we can conclude that no.of deaths is 494 and no.of survived people is 815.


# In[382]:


# Total no.of passengers onboard were 1309 ( as per this dataset)


# In[383]:


# Calculate the survival rate for train_data
survival_rate = df['Survived'].mean()
survival_rate = survival_rate * 100
death_rate = 100 - survival_rate
print("Survival Rate:", survival_rate)
print("Death Rate:", death_rate)


# In[384]:


# Here we can see that the rate of survival was 62 % whereas death rate was approx 38%.


# In[385]:


# Filtering the data to select rows where 'Sex' is '0' for female & '1' for male

# Calculate the percentage of women who survived
women_data = df[df['Sex'] == 0]
percentage_survived = (women_data['Survived'].sum() / len(women_data)) * 100
print("Percentage of women who survived:", percentage_survived)


# Calculate the percentage of male who survived
male_data = df[df['Sex'] == 1]
percentage_survived = (male_data['Survived'].sum() / len(male_data)) * 100
print("Percentage of male who survived:", percentage_survived)


# In[386]:


# From the above, we see that % of women who survived surpasses the % of survived men (i.e approx; 83 % vs 13 %)


# In[387]:


# B---- Calculate correlation matrix for all the features(test_data)

corr = df.corr()
plt.figure(figsize=(20,12))
plt.title('Heatmap showing Correlation between all the features', fontsize=20)
sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f")


# In[388]:


# The above heat map shows data with more survival rate than death rate.


# In[389]:


# FROM THE ABOVE WE CAN SEE THAT THE BAR PLOTS GIVES US BETTER RESULT.


# In[390]:


# Segregating the Dataset into Input(x) and Output(y)


# In[391]:


x = df.drop(columns=['Survived']).values
y = df['Survived'].values


# In[392]:


x


# In[393]:


y


# In[394]:


x.shape, y.shape


# In[395]:


# Splitting the Dataset into Training and Testing Data


# In[396]:


# we import additional libraries
from sklearn.model_selection import train_test_split
print('importing is done')


# In[397]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=17)


# In[398]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[399]:


# Loading the Models


# In[400]:


# We import additional libraries

from sklearn.svm import SVC
print('importing is done')


# In[401]:


svc = SVC()


# In[402]:


# Training the Models

svc.fit(x_train,y_train)


# In[403]:


# Predicting the Result Using the Trained Models

y_pred = svc.predict(x_test)


# In[404]:


y_pred


# In[405]:


y_test


# In[406]:


# Calculating the Accuracy of the Trained Models


# In[407]:


# we import additional library for accuracy

from sklearn.metrics import accuracy_score, classification_report
print('importing is done')


# In[408]:


# Now we check for accuracy of test data vs training data

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[409]:


np.round(accuracy,2)*100


# In[410]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[411]:


# Results:
# The above result shows us that our model prediction is 64 % accurate with 80/20 ratio.
# we can try to change the random state for finding options of better accuracy


# In[412]:


# If we keep split as 70/30
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=6)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
np.round(accuracy,2)*100


# In[413]:


# Accuracy with 70/30 split is by far the best one.


# In[414]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[415]:


x_test[4]


# In[416]:


x_test[4].shape


# In[417]:


x_test[4].reshape(1,7).shape


# In[418]:


svc.predict(x_test[4].reshape(1,7))


# In[419]:


y_test[4]


# In[420]:


# conclusion: Even though our model is at 66 % accuracy but it does provide an accurate prediction.


# In[421]:


# Now we try using Grad Boost Classifier on this same dataset


# In[422]:


# We import additional libraries

from sklearn.ensemble import GradientBoostingClassifier
print('importing is done')


# In[423]:


# Now we create a model for our dataset

gbc = GradientBoostingClassifier()
gbc


# In[424]:


# we fit the training data in our model

gbc.fit(x_train,y_train)
gbc


# In[425]:


# Now testing new Model prediction by providing Test Data set

y_pred = gbc.predict(x_test)


# In[426]:


y_pred


# In[427]:


y_test


# In[428]:


# Calculating the Accuracy of the Trained Models

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[429]:


np.round(accuracy,2)*100


# In[430]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[431]:


# With Gradient Boosting Classifier we get 89 % accuracy.


# In[432]:


# now we try with XG Boost Classifier


# In[433]:


# We import additional files 

from xgboost import XGBClassifier
print ("importing is done")


# In[434]:


# Now we create a model for our dataset

xgbc = XGBClassifier()
xgbc


# In[435]:


# we fit the training data in our model

xgbc.fit(x_train,y_train)
xgbc


# In[436]:


# Now testing new Model prediction by providing Test Data set

y_pred = xgbc.predict(x_test)
y_pred


# In[437]:


y_test


# In[438]:


# Calculating the Accuracy of the Trained Models

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))
np.round(accuracy,2)*100


# In[439]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[440]:


# With XGradient Boosting Classifier we get 87 % accuracy.


# In[441]:


# Conclusion : For the above dataset, Gradboost classifier gives the best accuracy for our model.


# In[ ]:





# In[ ]:




