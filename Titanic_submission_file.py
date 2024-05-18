#!/usr/bin/env python
# coding: utf-8

# In[86]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[87]:


# load the dataset

df1=pd.read_csv('gender_submission.csv')
df2=pd.read_csv('train.csv')
df3=pd.read_csv('test.csv')
print('loading is done')


# In[88]:


df1


# In[89]:


# Summarizing the dataset.
# Dataset "df1" shows only the passenger id along with their survival/non survival entries.
# Survival as "1" and Non-Survival is "0"


# In[90]:


df3


# In[91]:


# Summarizing the dataset.
# The above dataset 'df3' shows information of the legendary titanic ship passengers along with their respective details.
# Rich passengers have been charged highest fare and poor ones are with the least fare for the tickets class they bought that time.
# Also, the dataset shows their individual names along with gender and relevant ticket they owned during their travel.
# Finally, the dataset shows the people who have survived/succumbed during the crash of this legendary ship.


# In[ ]:


# DETAILS OF THE DATASET (in short) as per features:

# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex(male/female)
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


# In[92]:


# Pre-processing the dataset


# In[93]:


# Before we proceed further we will have to merge the 2 datasets: "df1" & "df3"

# Merge the two datasets based on 'PassengerId'
df= pd.merge(df1, df3, on='PassengerId', how='inner')


# In[94]:


df


# In[95]:


df.info()


# In[96]:


df.describe()


# In[97]:


len(df)


# In[98]:


df.dtypes


# In[99]:


# Here we have 5 columns with data as object type. we will check if there is any null values and then proceed.


# In[100]:


# Now we check for null values

df.isnull().sum()


# In[101]:


# we have found 86 entries in Age column and 327 entries in Cabin column as null values.


# In[102]:


# we drop both the columns with null values.

df.dropna(axis=1, inplace=True)


# In[103]:


df


# In[104]:


# From 12 columns we are down to 9 columns


# In[105]:


df.dtypes


# In[106]:


# now we will work upon changing the columns from object data types to integer
# we can see column Name, sex ticket and Embarked is in object type so we will use label encoding on it.


# In[107]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[108]:


label = LabelEncoder()
label


# In[109]:


# Column to be encoded
columns_to_encode = ['Name','Sex','Ticket','Embarked']


# In[110]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[111]:


#This is the final dataset with all integer values

df


# In[112]:


#This is the final dataset with all integer values


# In[113]:


df.info()


# In[114]:


df.isnull().sum().max


# In[115]:


# From the above all codes we were able to transform df dataset into integer types which does not have any null values.
# The above dataset we will use as our final form to process further.


# In[116]:


# Data Visualisation of the above dataset


# In[117]:


# BAR PLOT

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


# In[118]:


# So the total count happens to be as follow
total_passengers_onboard = (815+494)
print("Total_passengers_onboard:", total_passengers_onboard)
total_death = 494
print("Total_Death:", total_death)
total_survived = 815
print("Total_Survived:", total_survived)


# In[119]:


# From the above chart we can conclude that no.of deaths is 494 and no.of survived people is 815.
# Total no.of passengers onboard were 1309 ( as per this dataset)


# In[120]:


# Calculate the survival rate for train_data
survival_rate = df['Survived'].mean()
survival_rate = survival_rate * 100
death_rate = 100 - survival_rate
print("Survival Rate:", survival_rate)
print("Death Rate:", death_rate)


# In[121]:


# Here we can see that the rate of survival was 64 % whereas death rate was approx 36%.


# In[122]:


# Seggregating the dataset into x and y

x = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
y = df['Survived']


# In[123]:


x


# In[124]:


y


# In[125]:


x.shape, y.shape


# In[126]:


# Splitting the Dataset into Training and Testing Data


# In[127]:


# we import additional libraries

from sklearn.model_selection import train_test_split
print('importing is done')


# In[128]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=17)


# In[129]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[130]:


# Loading the Models


# In[131]:


# We import additional libraries

from sklearn.linear_model import LogisticRegression
print('Importing is done')


# In[132]:


lr = LogisticRegression()
lr


# In[133]:


# Training the Models

lr.fit(x_train,y_train)


# In[134]:


# Predicting the Result Using the Trained Models

y_pred = lr.predict(x_test)
y_pred


# In[135]:


y_test


# In[136]:


# Calculating the Accuracy of the Trained Models


# In[137]:


# we import additional library for accuracy

from sklearn.metrics import accuracy_score
print('importing is done')


# In[138]:


# Now we check for accuracy of test data vs training data

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[139]:


np.round(accuracy,2)*100


# In[140]:


# Predict survival for females and males

# Assuming Pclass=3, Sex=0 (female), SibSp=0, Parch=0, Embarked=1
female_survived = lr.predict_proba([[3, 0, 0, 0, 1]])[0][1] 
print("Predicted probability of female survival:", female_survived)


# Assuming Pclass=3, Sex=1 (male), SibSp=0, Parch=0, Embarked=1
male_survived = lr.predict_proba([[3, 1, 0, 0, 1]])[0][1]  
print("Predicted probability of male survival:", male_survived)


# In[141]:


# From the above we can conclude the following:
# 93 % of the female survived from the total population onboard the legendary Titanic Ship.
# 2 % of the male survived from the total population onboard
# 5 % of (female as well as male) did not survive the disastrous incident
# ( THE ABOVE CONCLUSIONS ARE DRAWN AS PER THE DATA PROVIDED IN THE DATASET )


# In[145]:


# Save DataFrame to a CSV file
df.to_csv('TITANIC_SUBMISSION_FILE.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




