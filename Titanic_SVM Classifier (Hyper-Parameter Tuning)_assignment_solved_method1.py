#!/usr/bin/env python
# coding: utf-8

# In[590]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[591]:


# load the dataset

surv=pd.read_csv('gender_submission.csv')
trn=pd.read_csv('train.csv')
tst=pd.read_csv('test.csv')
print('loading is done')


# In[592]:


surv


# In[593]:


# Summarizing the dataset.
# Dataset "surv" shows only the passenger id along with their survival/non survival entries.
# Survival as "1" and dead is "0"


# In[594]:


trn


# In[595]:


# Summarizing the dataset.
# The above dataset shows information of the legendary titanic ship passengers along with their respective details.
# Rich passengers have been charged highest fare and poor ones are with the least fare for the tickets class they bought that time.
# Also, the dataset shows their individual names along with gender and relevant ticket they owned during their travel.
# Finally, the dataset shows the people who have survived/succumbed during the crash of this legendary ship.


# In[596]:


tst


# In[597]:


# Summarizing the dataset.
# Dataset "tst" again shows all the passenger details except survival data in it.


# In[598]:


# Pre-processing the dataset


# In[599]:


# 1st we deal with train dataset

trn.info()


# In[600]:


trn.describe()


# In[601]:


len(trn)


# In[602]:


trn.dtypes


# In[603]:


# From the above dataset we can see that 5 columns as "object" type which we have to work upon.
trn.isnull().sum()


# In[604]:


trn.isnull().sum().max


# In[605]:


# From the above we can see that there are 177 entries in Age column and 687 entries in Cabin column as null values.


# In[606]:


# we drop both the columns with null values.

trn.dropna(axis=1, inplace=True)


# In[607]:


trn=trn.iloc[:,:-1]
trn


# In[608]:


x = trn
x


# In[609]:


# now we will work upon changing the columns from object data types to integer
# we can see column Name, sex and ticket is in object type so we will use label encoding on it.


# In[610]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[611]:


label = LabelEncoder()
label


# In[612]:


# Column to be encoded
columns_to_encode = ['Name','Sex','Ticket']


# In[613]:


# Apply label encoding to each column to be encoded

x[columns_to_encode]=x[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[614]:


#This is the final dataset with all integer values

x


# In[615]:


train_data = x


# In[616]:


train_data.info()


# In[617]:


train_data.isnull().sum().max


# In[618]:


# From the above all codes we were able to transform the "x" dataset into integer types which does not have any null values.
# The above we rename as train_data, which we will use as our final training dataset


# In[619]:


# 2nd we deal with test dataset


# In[620]:


tst.info()


# In[621]:


tst.describe()


# In[622]:


len(tst)


# In[623]:


tst.dtypes


# In[624]:


# Here also we have 5 columns with data as object type. we will check if there is any null values and then proceed.

tst.isnull().sum()


# In[625]:


# we have found 86 entries in Age column and 327 entries in Cabn column as null values.


# In[626]:


# we drop both the columns with null values.

tst.dropna(axis=1, inplace=True)


# In[627]:


tst


# In[628]:


# now we will work upon changing the columns from object data types to integer
# we can see column Name, sex and ticket is in object type so we will use label encoding on it.


# In[629]:


# Column to be encoded
columns_to_encode = ['Name','Sex','Ticket']


# In[630]:


# Apply label encoding to each column to be encoded

tst[columns_to_encode]=x[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[631]:


#This is the final dataset with all integer values

tst


# In[632]:


tst1= tst.iloc[:,:-1]
tst1


# In[633]:


tst1.dtypes


# In[634]:


tst1.isnull().sum().max


# In[635]:


# From the above all codes we were able to transform the "tst" dataset into integer types which does not have any null values.
# The above we rename as tst1, which we will process further to be used as our final testing dataset


# In[636]:


# 3rd we deal with "surv" dataset

surv


# In[637]:


surv.info()


# In[638]:


surv.describe()


# In[639]:


len(surv)


# In[640]:


surv.isnull().sum().max


# In[641]:


# we can see that "surv" dataset is all in integer data types with no null values


# In[642]:


# Before we proceed further we will have to merge the 2 dataset: "y" & "surv"

# Merge the two datasets based on 'PassengerId'
test_data = pd.merge(surv, tst1, on='PassengerId', how='inner')


# In[643]:


test_data


# In[644]:


test_data.info()


# In[645]:


test_data.isnull().sum().max


# In[646]:


test_data


# In[647]:


# The above is our cleaned final test data which we will use for further analysis.


# In[648]:


# so we have our test_data and train_data to be used for further processing.

train_data


# In[649]:


test_data


# In[650]:


# Data Visualisation of the above dataset


# In[651]:


# A--- SCATTER PLOTS

# Plot scatterplots for each feature against the target column

target_column = 'Survived'
features = df.drop(columns=[target_column]).columns
num_cols = 3  # Number of columns for subplots
num_rows = (len(features) - 1) // num_cols + 1
plt.figure(figsize=(10, 5))

for i, feature in enumerate(features, start=1):
    plt.subplot(num_rows, num_cols, i)
    plt.scatter(df[feature], df[target_column], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel(target_column)

plt.tight_layout()
plt.show()


# In[652]:


# B ---BAR PLOTS

# 1) Bar plot for train_data

# Count the number of survivors
survival_counts = train_data['Survived'].value_counts()

# Plot the bar chart
plt.bar(survival_counts.index, survival_counts.values)

# Add labels and title
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Number of Survivors')

# Add annotations
for i, count in enumerate(survival_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

# Show the plot
plt.show()


# In[653]:


# So the total count happens to be as follow
total_passengers_onboard = (549+266+342+152)
print("Total_passengers_onboard:", total_passengers_onboard)
total_death = 549 + 266
print("Total_Death:", total_death)
total_survived = 342 + 152
print("Total_Survived:", total_survived)


# In[654]:


# Calculate the survival rate for train_data
survival_rate = train_data['Survived'].mean()
survival_rate = survival_rate * 100
death_rate = 100 - survival_rate
print("Survival Rate:", survival_rate)
print("Death Rate:", death_rate)


# In[655]:


# Train data survival percentage:

# Filtering the data to select rows where 'Sex' is '0' for female & '1' for male

# Calculate the percentage of women who survived
women_data = train_data[train_data['Sex'] == 0]
percentage_survived = (women_data['Survived'].sum() / len(women_data)) * 100
print("Percentage of women who survived:", percentage_survived)


# Calculate the percentage of male who survived
male_data = train_data[train_data['Sex'] == 1]
percentage_survived = (male_data['Survived'].sum() / len(male_data)) * 100
print("Percentage of male who survived:", percentage_survived)


# In[656]:


# From the above train_data, we can conclude that only 342 people onboard survived whereas 549 died in the legendary titanic crash.
# The survival rate was 38.38 from wwhich 74.20 % were women and 18.89 % were men.


# In[657]:


# 2) Bar plot for test_data

# Count the number of survivors
survival_counts = test_data['Survived'].value_counts()

# Plot the bar chart
plt.bar(survival_counts.index, survival_counts.values)

# Add labels and title
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Number of Survivors')

# Add annotations
for i, count in enumerate(survival_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

# Show the plot
plt.show()


# In[658]:


# So the total count happens to be as follow
total_passengers_onboard = (549+266+342+152)
print("Total_passengers_onboard:", total_passengers_onboard)
total_death = 549 + 266
print("Total_Death:", total_death)
total_survived = 342 + 152
print("Total_Survived:", total_survived)


# In[659]:


# Calculate the survival rate for train_data
survival_rate = test_data['Survived'].mean()
survival_rate = survival_rate * 100
death_rate = 100 - survival_rate
print("Survival Rate:", survival_rate)
print("Death Rate:", death_rate)


# In[660]:


# Test data survival percentage:

# Filtering the data to select rows where 'Sex' is '0' for female & '1' for male

# Calculate the percentage of women who survived
women_data = test_data[test_data['Sex'] == 0]
percentage_survived = (women_data['Survived'].sum() / len(women_data)) * 100
print("Percentage of women who survived:", percentage_survived)


# Calculate the percentage of male who survived
male_data = test_data[test_data['Sex'] == 1]
percentage_survived = (male_data['Survived'].sum() / len(male_data)) * 100
print("Percentage of male who survived:", percentage_survived)


# In[661]:


# From the above train_data, we can conclude that 152 people onboard survived whereas 266 died in the legendary titanic crash.
# The survival rate was 36.36 from wwhich 36.25 % were women and 36.43 % were men.


# In[662]:


# From the above we can see that death_rate and surival_rate.


# In[663]:


# Calculate correlation matrix for all the features(test_data)

corr = test_data.corr()
plt.figure(figsize=(20,12))
plt.title('Heatmap showing Correlation between all the features', fontsize=20)
sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f")


# In[664]:


# The above heat map shows data with more death rate than survival rate.


# In[665]:


# Calculate correlation matrix for all the features(train_data)

corr = train_data.corr()
plt.figure(figsize=(20,12))
plt.title('Heatmap showing Correlation between all the features', fontsize=20)
sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f")


# In[666]:


# The above heat map shows data with more survival rate than death rate.


# In[667]:


# FROM THE ABOVE WE CAN SEE THAT THE BAR PLOTS GIVES US BETTER RESULT THAN OTHER CHART TYPES.


# In[668]:


# We do not need to segregate the Dataset into Input(x) and Output(y) as it is already in train and test mode.
# Splitting the Dataset into Training and Testing Data not needed as it is already done
# 1) Train dataset

x_train = train_data.drop(columns=['Survived']).values
y_train = train_data['Survived'].values


# In[669]:


x_train.shape, y_train.shape


# In[670]:


# 2) Test dataset

x_test = test_data.drop(columns=['Survived']).values
y_test = test_data['Survived'].values


# In[671]:


x_test.shape, y_test.shape


# In[672]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[673]:


# Loading the Models


# In[674]:


# We import additional libraries

from sklearn.svm import SVC
print('importing is done')


# In[675]:


svc = SVC()


# In[676]:


# Training the Models

svc.fit(x_train , y_train)


# In[677]:


# Predicting the Result Using the Trained Models

y_pred = svc.predict(x_test)


# In[678]:


y_pred


# In[679]:


y_test


# In[680]:


# Calculating the Accuracy of the Trained Models


# In[681]:


# we import additional library for accuracy
from sklearn.metrics import accuracy_score, classification_report
print('importing is done')


# In[682]:


# Now we check for accuracy of test data vs training data
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[683]:


np.round(accuracy,2)*100


# In[684]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[685]:


# If we keep split as 70/30
test_size=0.3
random_state=17
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
np.round(accuracy,2)*100


# In[686]:


# we import additional libraries

from imblearn.over_sampling import SMOTE
print('importing is done')


# In[687]:


# Create an instance of SMOTE with random_state=42

smote = SMOTE(random_state=42)
print ('Done')


# In[688]:


# Apply SMOTE to generate synthetic samples

x_resampled, y_resampled = smote.fit_resample(x_train, y_train)


# In[689]:


svc.fit(x_resampled, y_resampled)
y_pred = svc.predict(x_test)
y_pred


# In[690]:


y_test


# In[691]:


y_test


# In[692]:


accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[693]:


report = classification_report(y_test, y_pred)
print(report)


# In[694]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[695]:


x_test.shape


# In[696]:


x_test[4].shape


# In[697]:


x_test[4].reshape(1,7)


# In[698]:


x_test[4].reshape(1,7).shape


# In[699]:


svc.predict(x_test[4].reshape(1,7))


# In[700]:


y_test[4]


# In[701]:


# Here the prediction result fail to execute accurately comcluding that our model failed.


# In[702]:


# Now we try using Grad Boost Classifier on this same dataset


# In[703]:


# We import additional libraries

from sklearn.ensemble import GradientBoostingClassifier
print('importing is done')


# In[704]:


# Now we create a model for our dataset

gbc = GradientBoostingClassifier()
gbc


# In[705]:


# we fit the training data in our model

gbc.fit(x_train,y_train)
gbc


# In[706]:


# Now testing new Model prediction by providing Test Data set

y_pred = gbc.predict(x_test)


# In[707]:


y_pred


# In[708]:


y_test


# In[709]:


# Calculating the Accuracy of the Trained Models

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[710]:


np.round(accuracy,2)*100


# In[711]:


report = classification_report(y_test, y_pred)
print(report)


# In[712]:


# Grad Boost Classifier gives us more accuracy as compared to Support Vector Classifier.

