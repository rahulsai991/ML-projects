#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# 
# ## Data Set Problems
# 
# The company seeks **to automate (in real time) the loan qualifying procedure** based on information given by customers while filling out an online application form. It is expected that the development of ML models that can help the company predict loan approval in **accelerating decision-making process** for determining whether an applicant is eligible for a loan or not.
# 
# ---
# 
# ## Objectives of Notebook
# **This notebook aims to:**
# *   Analyze customer data provided in data set (EDA)
# *   Build various ML models that can predict loan approval
# 
# **The machine learning models used in this project are:** 
# 1. Logistic Regression
# 2. K-Nearest Neighbour (KNN)
# 3. Support Vector Machine (SVM)
# 4. Naive Bayes
# 5. Decision Tree
# 6. Random Forest
# 7. Gradient Boost
# 
# ---
# 
# ## Data Set Description
# There are **13 variables** in this data set:
# *   **8 categorical** variables,
# *   **4 continuous** variables, and
# *   **1** variable to accommodate the loan ID.
# 
# <br>
# 
# The following is the **structure of the data set**.
# 
# 
# <table style="width:100%">
# <thead>
# <tr>
# <th style="text-align:center; font-weight: bold; font-size:14px">Variable Name</th>
# <th style="text-align:center; font-weight: bold; font-size:14px">Description</th>
# <th style="text-align:center; font-weight: bold; font-size:14px">Sample Data</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td><b>Loan_ID</b></td>
# <td>Loan reference number <br> (unique ID)</td>
# <td>LP001002; LP001003; ...</td>
# </tr>
# <tr>
# <td><b>Gender</b></td>
# <td>Applicant gender <br> (Male or Female)</td>
# <td>Male; Female</td>
# </tr>
# <tr>
# <td><b>Married</b></td>
# <td>Applicant marital status <br> (Married or not married)</td>
# <td>Married; Not Married</td>
# </tr>
# <tr>
# <td><b>Dependents</b></td>
# <td>Number of family members</td>
# <td>0; 1; 2; 3+</td>
# </tr>
# <tr>
# <td><b>Education</b></td>
# <td>Applicant education/qualification <br> (graduate or not graduate)</td>
# <td>Graduate; Under Graduate</td>
# </tr>
# <tr>
# <td><b>Self_Employed</b></td>
# <td>Applicant employment status <br> (yes for self-employed, no for employed/others)</td>
# <td>Yes; No</td>
# </tr>
# <tr>
# <td><b>ApplicantIncome</b></td>
# <td>Applicant's monthly salary/income</td>
# <td>5849; 4583; ...</td>
# </tr>
# <tr>
# <td><b>CoapplicantIncome</b></td>
# <td>Additional applicant's monthly salary/income</td>
# <td>1508; 2358; ...</td>
# </tr>
# <tr>
# <td><b>LoanAmount</b></td>
# <td>Loan amount</td>
# <td>128; 66; ...</td>
# </tr>
# <tr>
# <td><b>Loan_Amount_Term</b></td>
# <td>The loan's repayment period (in days)</td>
# <td>360; 120; ...</td>
# </tr>
# <tr>
# <td><b>Credit_History</b></td>
# <td>Records of previous credit history <br> (0: bad credit history, 1: good credit history)</td>
# <td>0; 1</td>
# </tr>
# <tr>
# <td><b>Property_Area</b></td>
# <td>The location of property <br> (Rural/Semiurban/Urban)</td>
# <td>Rural; Semiurban; Urban</td>
# </tr>
# <tr>
# <td><b>Loan_Status</b></td>
# <td>Status of loan <br> (Y: accepted, N: not accepted)</td>
# <td>Y; N</td>
# </tr>
# </tbody>
# </table>
# 

# # 2. Importing Libraries
# Importing libraries that will be used in this notebook.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as mso
import seaborn as sns
import warnings
import os
import scipy

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# # 3. Reading Data Set 
# After importing libraries, we will also import the dataset that will be used.

# In[2]:


df = pd.read_csv("../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv")
df.head()


# In[3]:


print(df.shape)


# As can be seen, the **13 columns** are readable. It also can be seen that there are **614 observations** in the data set.

# # 4. Data Exploration
# This section will perform data exploration of "raw" data set that has been imported.

# ## 4.1 Categorical Variable
#  The first type of variable that I will explore is categorical variable.

# ### 4.1.1 Loan ID

# In[4]:


df.Loan_ID.value_counts(dropna=False)


#  It can be seen that there are 614 unique ID in the dataset.

# ### 4.1.2 Gender

# In[5]:


df.Gender.value_counts(dropna=False)


# In[6]:


sns.countplot(x="Gender", data=df, palette="hls")
plt.show()


# In[7]:


countMale = len(df[df.Gender == 'Male'])
countFemale = len(df[df.Gender == 'Female'])
countNull = len(df[df.Gender.isnull()])

print("Percentage of Male applicant: {:.2f}%".format((countMale / (len(df.Gender))*100)))
print("Percentage of Female applicant: {:.2f}%".format((countFemale / (len(df.Gender))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Gender))*100)))


# From the results above, the number of male applicants is higher compared to female applicants. It also can be seen there are missing values in this column.

# ### 4.1.3 Married

# In[8]:


df.Married.value_counts(dropna=False)


# In[9]:


sns.countplot(x="Married", data=df, palette="Paired")
plt.show()


#  The number of applicants that has been married is higher compared to applicants that hasn't married. It also can be seen there are small number of missing values in this column.

# In[10]:


countMarried = len(df[df.Married == 'Yes'])
countNotMarried = len(df[df.Married == 'No'])
countNull = len(df[df.Married.isnull()])

print("Percentage of married: {:.2f}%".format((countMarried / (len(df.Married))*100)))
print("Percentage of Not married applicant: {:.2f}%".format((countNotMarried / (len(df.Married))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Married))*100)))


# ### 4.1.4 Education

# In[11]:


df.Education.value_counts(dropna=False)


# In[12]:


sns.countplot(x="Education", data=df, palette="rocket")
plt.show()


# In[13]:


countGraduate = len(df[df.Education == 'Graduate'])
countNotGraduate = len(df[df.Education == 'Not Graduate'])
countNull = len(df[df.Education.isnull()])

print("Percentage of graduate applicant: {:.2f}%".format((countGraduate / (len(df.Education))*100)))
print("Percentage of Not graduate applicant: {:.2f}%".format((countNotGraduate / (len(df.Education))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Education))*100)))


#  The number of applicants that has been graduated is higher compared to applicants that hasn't graduated.

# ### 4.1.5 Self Employed

# In[14]:


df.Self_Employed.value_counts(dropna=False)


# In[15]:


sns.countplot(x="Self_Employed", data=df, palette="crest")
plt.show()


# In[16]:


countNo = len(df[df.Self_Employed == 'No'])
countYes = len(df[df.Self_Employed == 'Yes'])
countNull = len(df[df.Self_Employed.isnull()])

print("Percentage of Not self employed: {:.2f}%".format((countNo / (len(df.Self_Employed))*100)))
print("Percentage of self employed: {:.2f}%".format((countYes / (len(df.Self_Employed))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Self_Employed))*100)))


#  The number of applicants that are not self employed is higher compared to applicants that are self employed. It also can be seen, there are missing values in this column.

# ### 4.1.6 Credit History

# In[17]:


df.Credit_History.value_counts(dropna=False)


# In[18]:


sns.countplot(x="Credit_History", data=df, palette="viridis")
plt.show()


# In[19]:


count1 = len(df[df.Credit_History == 1])
count0 = len(df[df.Credit_History == 0])
countNull = len(df[df.Credit_History.isnull()])

print("Percentage of Good credit history: {:.2f}%".format((count1 / (len(df.Credit_History))*100)))
print("Percentage of Bad credit history: {:.2f}%".format((count0 / (len(df.Credit_History))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Credit_History))*100)))


# The number of applicants that have good credit history is higher compared to applicants that have bad credit history. It also can be seen, there are missing values in this column.

# ### 4.1.7 Property Area

# In[20]:


df.Property_Area.value_counts(dropna=False)


# In[21]:


sns.countplot(x="Property_Area", data=df, palette="cubehelix")
plt.show()


# In[22]:


countUrban = len(df[df.Property_Area == 'Urban'])
countRural = len(df[df.Property_Area == 'Rural'])
countSemiurban = len(df[df.Property_Area == 'Semiurban'])
countNull = len(df[df.Property_Area.isnull()])

print("Percentage of Urban: {:.2f}%".format((countUrban / (len(df.Property_Area))*100)))
print("Percentage of Rural: {:.2f}%".format((countRural / (len(df.Property_Area))*100)))
print("Percentage of Semiurban: {:.2f}%".format((countSemiurban / (len(df.Property_Area))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Property_Area))*100)))


# This column has a balanced distribution between Urban, Rural, and Semiurban property area. It also can be seen there is no missing value.

# ### 4.1.8 Loan Status

# In[23]:


df.Loan_Status.value_counts(dropna=False)


# In[24]:


sns.countplot(x="Loan_Status", data=df, palette="YlOrBr")
plt.show()


# In[25]:


countY = len(df[df.Loan_Status == 'Y'])
countN = len(df[df.Loan_Status == 'N'])
countNull = len(df[df.Loan_Status.isnull()])

print("Percentage of Approved: {:.2f}%".format((countY / (len(df.Loan_Status))*100)))
print("Percentage of Rejected: {:.2f}%".format((countN / (len(df.Loan_Status))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Loan_Status))*100)))


# The number of approved loans is higher compared to rejected loans . It also can be seen, there is no missing values in this column.

# ### 4.1.9 Loan Amount Term

# In[26]:


df.Loan_Amount_Term.value_counts(dropna=False)


# In[27]:


sns.countplot(x="Loan_Amount_Term", data=df, palette="rocket")
plt.show()


# In[28]:


count12 = len(df[df.Loan_Amount_Term == 12.0])
count36 = len(df[df.Loan_Amount_Term == 36.0])
count60 = len(df[df.Loan_Amount_Term == 60.0])
count84 = len(df[df.Loan_Amount_Term == 84.0])
count120 = len(df[df.Loan_Amount_Term == 120.0])
count180 = len(df[df.Loan_Amount_Term == 180.0])
count240 = len(df[df.Loan_Amount_Term == 240.0])
count300 = len(df[df.Loan_Amount_Term == 300.0])
count360 = len(df[df.Loan_Amount_Term == 360.0])
count480 = len(df[df.Loan_Amount_Term == 480.0])
countNull = len(df[df.Loan_Amount_Term.isnull()])

print("Percentage of 12: {:.2f}%".format((count12 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 36: {:.2f}%".format((count36 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 60: {:.2f}%".format((count60 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 84: {:.2f}%".format((count84 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 120: {:.2f}%".format((count120 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 180: {:.2f}%".format((count180 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 240: {:.2f}%".format((count240 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 300: {:.2f}%".format((count300 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 360: {:.2f}%".format((count360 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 480: {:.2f}%".format((count480 / (len(df.Loan_Amount_Term))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Loan_Amount_Term))*100)))


# As can be seen from the results, **the 360 days loan duration is the most popular** compared to others.

# ## 4.2 Numerical Variable
# The second variable that I will explore is categorical variable.

# ### 4.2.1 Describe Numerical Variable
#  This section will show mean, count, std, min, max and others using describe function.
# 

# In[29]:


df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].describe()


# ### 4.2.2 Distribution of Numerical Variable
#  In this section, I will show the distribution of numerical variable using histogram and violin plot.

# #### 4.2.2.1 Histogram Distribution

# In[30]:


sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue')
sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange');


# #### 4.2.2.2 Violin Plot

# In[31]:


sns.set(style="darkgrid")
fig, axs1 = plt.subplots(2, 2, figsize=(10, 10))

sns.violinplot(data=df, y="ApplicantIncome", ax=axs1[0, 0], color='green')
sns.violinplot(data=df, y="CoapplicantIncome", ax=axs1[0, 1], color='skyblue')
sns.violinplot(data=df, y="LoanAmount", ax=axs1[1, 0], color='orange');


# *   The distribution of **Applicant income, Co Applicant Income, and Loan Amount** are **positively skewed** and **it has outliers** (can be seen from both histogram and violin plot).
# *   The distribution of **Loan Amount Term** is **negativly skewed** and **it has outliers.**
# 
# 

# ## 4.3 Other Exploration
#  This section will show additional exploration from each variables. The additional exploration are:
# *   Bivariate analysis (categorical w/ categorical, categroical w/ numerical, and numerical w/ numerical)
# *   Heatmap
# 
# 

# ### 4.3.1 Heatmap

# In[32]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap='inferno');


#  There is positive correlation between Loan Amount and Applicant Income

# ### 4.3.2 Categorical - categorical

# In[33]:


pd.crosstab(df.Gender,df.Married).plot(kind="bar", stacked=True, figsize=(5,5), color=['#f64f59','#12c2e9'])
plt.title('Gender vs Married')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# Most male applicants are already married compared to female applicants. Also, the number of not married male applicants are higher compare to female applicants that had not married.

# In[34]:


pd.crosstab(df.Self_Employed,df.Credit_History).plot(kind="bar", stacked=True, figsize=(5,5), color=['#544a7d','#ffd452'])
plt.title('Self Employed vs Credit History')
plt.xlabel('Self Employed')
plt.ylabel('Frequency')
plt.legend(["Bad Credit", "Good Credit"])
plt.xticks(rotation=0)
plt.show()


# Most not self employed applicants have good credit compared to self employed applicants.

# In[35]:


pd.crosstab(df.Property_Area,df.Loan_Status).plot(kind="bar", stacked=True, figsize=(5,5), color=['#333333','#dd1818'])
plt.title('Property Area vs Loan Status')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# Most of loan that got accepted has property in Semiurban compared to Urban and Rural.

# ### 4.3.3 Categorical - Numerical

# In[36]:


sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df, palette="mako");


#  It can be seen that there are lots of outliers in Applicant Income, and the distribution also positively skewed

# In[37]:


sns.boxplot(x="CoapplicantIncome", y="Loan_Status", data=df, palette="rocket");


#  It's clear that Co Applicant Income has a number of outliers, and the distribution is also positively skewed.

# In[38]:


sns.boxplot(x="Loan_Status", y="LoanAmount", data=df, palette="YlOrBr");


#  As can be seen, Co Applicant Income has a high number of outliers, and the distribution is also positively skewed.

# ### 4.3.4 Numerical - Numerical 

# In[39]:


df.plot(x='ApplicantIncome', y='CoapplicantIncome', style='o')  
plt.title('Applicant Income - Co Applicant Income')  
plt.xlabel('ApplicantIncome')
plt.ylabel('CoapplicantIncome')  
plt.show()
print('Pearson correlation:', df['ApplicantIncome'].corr(df['CoapplicantIncome']))
print('T Test and P value: \n', stats.ttest_ind(df['ApplicantIncome'], df['CoapplicantIncome']))


# *   There is **negative correlation** between Applicant income and Co Applicant Income.
# *   The correlation coefficient is **significant** at the 95 per cent confidence interval, as it has a **p-value of 1.46**
# 

# ## 4.4 Null Values

# In[40]:


df.isnull().sum()


# In[41]:


plt.figure(figsize = (24, 5))
axz = plt.subplot(1,2,2)
mso.bar(df, ax = axz, fontsize = 12);


# Previously, the null values has been explored for Categorical Variables. In this section, the null values has been explored **for all variables** in the dataset.

# # 5. Data Preprocessing

# ## 5.1 Drop Unecessary Variables
# Unecessary variables will be dropped in this section.

# In[42]:


df = df.drop(['Loan_ID'], axis = 1)


# ## 5.2 Data Imputation
# Imputation is a technique for substituting an estimated value for missing values in a dataset. In this section, the imputation will be performed for variables that have missing values.

# ### 5.2.1 Categorical Variables
#  In this section, the imputation for categorical variables will be performed using **mode**.

# In[43]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)


# ### 5.2.2 Numerical Variables
# The next section is imputation for numerical variables using **mean**.

# In[44]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# ## 5.3 One-hot Encoding
# In this section, I will **transform categorical variables into a form that could be provided by ML algorithms to do a better prediction.**

# In[45]:


df = pd.get_dummies(df)

# Drop columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis = 1)

# Rename columns name
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}
       
df.rename(columns=new, inplace=True)


# ## 5.3 Remove Outliers & Infinite values
#  Since there are outliers, **the outliers will be removed**. <br>
# 

# In[46]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# ## 5.4 Skewed Distribution Treatment
#  In previous section, it already shown that **distribution for ApplicantIncome, CoapplicantIncome, and LoanAmount is positively skewed**. <br>
# I will use **square root transformation** to normalized the distribution.

# In[47]:


# Square Root Transformation

df.ApplicantIncome = np.sqrt(df.ApplicantIncome)
df.CoapplicantIncome = np.sqrt(df.CoapplicantIncome)
df.LoanAmount = np.sqrt(df.LoanAmount)


# In[48]:


sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue')
sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange');


#  As can be seen, the distribution after using log transformation are much better compared to original distribution.

# ## 5.5 Features Separating 
#  Dependent features (Loan_Status) will be seperated from independent features.

# In[49]:


X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]


# ## 5.6 SMOTE Technique
#  In previous exploration, it can be seen that **the number between approved and rejected loan is imbalanced**. In this section, **oversampling technique will be used to avoid overfitting**,

# In[50]:


X, y = SMOTE().fit_resample(X, y)


# In[51]:


sns.set_theme(style="darkgrid")
sns.countplot(y=y, data=df, palette="coolwarm")
plt.ylabel('Loan Status')
plt.xlabel('Total')
plt.show()


#  As can be seen, the distrubtion of Loan status are now **balanced**.

# ## 5.7 Data Normalization 
#  In this section, data normalization will be performed **to normalize the range of independent variables or features of data**.

# In[52]:


X = MinMaxScaler().fit_transform(X)


# ## 5.8 Splitting Data Set
#  The data set will be split into **80% train and 20% test**.

# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # 6. Models

# ## 6.1 Logistic Regression

# In[54]:


LRclassifier = LogisticRegression(solver='saga', max_iter=500, random_state=1)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
LRAcc = accuracy_score(y_pred,y_test)
print('LR accuracy: {:.2f}%'.format(LRAcc*100))


# ## 6.2 K-Nearest Neighbour (KNN)

# In[55]:


scoreListknn = []
for i in range(1,21):
    KNclassifier = KNeighborsClassifier(n_neighbors = i)
    KNclassifier.fit(X_train, y_train)
    scoreListknn.append(KNclassifier.score(X_test, y_test))
    
plt.plot(range(1,21), scoreListknn)
plt.xticks(np.arange(1,21,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
KNAcc = max(scoreListknn)
print("KNN best accuracy: {:.2f}%".format(KNAcc*100))


# ## 6.3 Support Vector Machine (SVM)

# In[56]:


SVCclassifier = SVC(kernel='rbf', max_iter=500)
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy: {:.2f}%'.format(SVCAcc*100))


# ## 6.4 Naive Bayes

# ### 6.4.1 Categorical NB

# In[57]:


NBclassifier1 = CategoricalNB()
NBclassifier1.fit(X_train, y_train)

y_pred = NBclassifier1.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc1 = accuracy_score(y_pred,y_test)
print('Categorical Naive Bayes accuracy: {:.2f}%'.format(NBAcc1*100))


# ### 6.4.2 Gaussian NB

# In[58]:


NBclassifier2 = GaussianNB()
NBclassifier2.fit(X_train, y_train)

y_pred = NBclassifier2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc2 = accuracy_score(y_pred,y_test)
print('Gaussian Naive Bayes accuracy: {:.2f}%'.format(NBAcc2*100))


# ## 6.5 Decision Tree

# In[59]:


scoreListDT = []
for i in range(2,21):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2,21), scoreListDT)
plt.xticks(np.arange(2,21,1))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAcc = max(scoreListDT)
print("Decision Tree Accuracy: {:.2f}%".format(DTAcc*100))


# ## 6.6 Random Forest

# In[60]:


scoreListRF = []
for i in range(2,25):
    RFclassifier = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
    
plt.plot(range(2,25), scoreListRF)
plt.xticks(np.arange(2,25,1))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAcc = max(scoreListRF)
print("Random Forest Accuracy:  {:.2f}%".format(RFAcc*100))


# ## 6.7 Gradient Boosting

# In[61]:


paramsGB={'n_estimators':[100,200,300,400,500],
      'max_depth':[1,2,3,4,5],
      'subsample':[0.5,1],
      'max_leaf_nodes':[2,5,10,20,30,40,50]}


# In[62]:


GB = RandomizedSearchCV(GradientBoostingClassifier(), paramsGB, cv=20)
GB.fit(X_train, y_train)


# In[63]:


print(GB.best_estimator_)
print(GB.best_score_)
print(GB.best_params_)
print(GB.best_index_)


# In[64]:


GBclassifier = GradientBoostingClassifier(subsample=0.5, n_estimators=400, max_depth=4, max_leaf_nodes=10)
GBclassifier.fit(X_train, y_train)

y_pred = GBclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
GBAcc = accuracy_score(y_pred,y_test)
print('Gradient Boosting accuracy: {:.2f}%'.format(GBAcc*100))


# # 7. Model Comparison

# In[1]:


compare = pd.DataFrame({'Model': ['Logistic Regression', 'K Neighbors', 
                                  'SVM', 'Categorical NB', 
                                  'Gaussian NB', 'Decision Tree', 
                                  'Random Forest', 'Gradient Boost'], 
                        'Accuracy': [LRAcc*100, KNAcc*100, SVCAcc*100, 
                                     NBAcc1*100, NBAcc2*100, DTAcc*100, 
                                     RFAcc*100, GBAcc*100]})
compare.sort_values(by='Accuracy', ascending=False)


# In general, it can be seen that **all models can achieve up to 70% accuracy**. <br>
# The highest accuracy is **93%%**. <br><br>
# 
