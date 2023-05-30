#!/usr/bin/env python
# coding: utf-8

# # Data Collection

# In[2]:


#Data Collection
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes_data = pd.read_csv('C:\JUPYTER\diabetes.csv')


# # Data Analysis

# In[19]:


print("Pima Indians Diabetes data set dimensions : {}".format(diabetes_data.shape))


# In[36]:


diabetes_data.columns


# In[18]:


diabetes_data.head(10)


# In[20]:


diabetes_data.groupby('Outcome').size()


# In[28]:


sns.countplot(x="Outcome",data = diabetes_data)


# In[48]:


plt.figure(figsize = (10,10))
sns.countplot(x="Outcome",hue = "Pregnancies", data = diabetes_data)


# In[32]:


diabetes_data['Pregnancies'].plot.hist(bins=20, figsize = (10,5))


# In[33]:


diabetes_data['Glucose'].plot.hist(bins=20, figsize = (10,5))


# In[34]:


diabetes_data['BloodPressure'].plot.hist(bins=20, figsize = (10,5))


# In[35]:


diabetes_data['SkinThickness'].plot.hist(bins=20, figsize = (10,5))


# In[37]:


diabetes_data['Insulin'].plot.hist(bins=20, figsize = (10,5))


# In[38]:


diabetes_data['BMI'].plot.hist(bins=20, figsize = (10,5))


# In[39]:


diabetes_data['DiabetesPedigreeFunction'].plot.hist(bins=20, figsize = (10,5))


# In[40]:


diabetes_data['Age'].plot.hist(bins=20, figsize = (10,5))


# In[43]:


relation = diabetes_data.corr() #plotting a heatmap to analyze the correlation between different factors and outcome 
sns.heatmap(relation, xticklabels=relation.columns,yticklabels=relation.columns, cmap='viridis')


# 
# 
# # Data Cleaning

# In[41]:


#Finding and Removing null values
#Finding null values
diabetes_data.isnull()


# In[46]:


diabetes_data.isnull().sum() #checking to see if there are no null values present in dataset


# In[4]:


#Finding and Removing False/Invalid Data or Unexpected Outliers 
#Finding Invalid Data
print("Invalid values for Glucose : ", diabetes_data[diabetes_data.Glucose == 0].shape[0]) #Glucose level cannot be 0
print(diabetes_data[diabetes_data.Glucose == 0].groupby('Outcome')['Age'].count())
print("Invalid values for Blood Pressure : ", diabetes_data[diabetes_data.BloodPressure == 0].shape[0]) # Blood Pressure cannot be 0
print(diabetes_data[diabetes_data.BloodPressure == 0].groupby('Outcome')['Age'].count())
print("Invalid values for Skin Thickness : ", diabetes_data[diabetes_data.SkinThickness ==0].shape[0])# Skin thickness cannot be 0
print(diabetes_data[diabetes_data.SkinThickness == 0].groupby('Outcome')['Age'].count())
print("Invalid values for Insulin : ", diabetes_data[diabetes_data.Insulin == 0].shape[0])# Insulin cannot be 0
print(diabetes_data[diabetes_data.Insulin == 0].groupby('Outcome')['Age'].count())
print("Invalid values for BMI : ", diabetes_data[diabetes_data.BMI == 0].shape[0]) #BMI cannot be 0
print(diabetes_data[diabetes_data.BMI == 0].groupby('Outcome')['Age'].count())


# In[227]:


#Removing data which is invalid for Glucose, Blood Pressure and BMI. 
#Invalid data for Skin Thickness and Insulin is kept as it is more in number and contains a lot of other information.
diabetes_data_clean = diabetes_data[(diabetes_data.Glucose != 0) & (diabetes_data.BloodPressure != 0) & (diabetes_data.BMI != 0)]


# In[319]:


print("Cleaned  data set dimensions : {}".format(diabetes_data_clean.shape))


# # Model Selection

# In[338]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[339]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[340]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('GNB', GaussianNB()))
models.append(('SVC', SVC()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))


# In[341]:


# Splitting the data into training and testing data
dfTrain = diabetes_data_clean[:620]
dfTest = diabetes_data_clean[620:720]
dfCheck = diabetes_data_clean[720:]
y_train = np.asarray(dfTrain['Outcome'])
X_train = np.asarray(dfTrain.drop('Outcome',1))
y_test = np.asarray(dfTest['Outcome'])
X_test = np.asarray(dfTest.drop('Outcome',1))
means = np.mean(X_train, axis=0)
stds = np.std(X_train, axis=0)
X_train = (X_train - means)/stds
X_test = (X_test - means)/stds


# In[354]:


#train/test split method
names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Accuracy': scores})
print(tr_split)


# In[355]:


plot = sns.barplot(x = 'Name', y = 'Accuracy', data = tr_split)
plot.set(xlabel='Name', ylabel='Accuracy')
for p in plot.patches:
    height = p.get_height()
    plot.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# In[356]:


#K-fold cross-validation method 
from sklearn.model_selection import KFold
names = []
scores = []
for name, model in models:
    
    kfold = KFold(n_splits=10) 
    score = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Accuracy': scores})
print(kf_cross_val)


# In[345]:


plot = sns.barplot(x = 'Name', y = 'Accuracy', data =kf_cross_val )
plot.set(xlabel='Name', ylabel='Accuracy')
for p in plot.patches:
    height = p.get_height()
    plot.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# # Model Training,Testing and Predictions

# In[357]:


#Choosing RF as our classifier
from sklearn.ensemble import RandomForestClassifier
diabetesCheck = RandomForestClassifier()
diabetesCheck.fit(X_train, y_train)
predictions = diabetesCheck.predict(X_test)


# In[358]:


from sklearn import metrics
print("Classification Report:")
classification_report(y_test, predictions)


# In[359]:


print("Confusion matrix:")
confusion_matrix(y_test,predictions)


# In[360]:


print("Accuracy of Model:",metrics.accuracy_score(y_test, predictions))


# In[361]:


#printing the remaining four records
print(dfCheck.head(4))


# In[362]:


#using the remaining 4 records to predict if they have diabetes
sampleData = dfCheck[:4]
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
predictionProbability = diabetesCheck.predict_proba(sampleDataFeatures)
prediction = diabetesCheck.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)


# # End
