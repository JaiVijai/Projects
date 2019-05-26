import pandas as pd
import numpy as np
import os

### set work directory###
os.chdir('')

### import the data ###

train=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
test=pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

##3 check for missing value##

train_missing=train.isnull().sum()
test1_missing = test.isnull().sum()

### Imput the missing value in Train ###
## in Gender ##3

train['Gender'].value_counts()
new_gender=np.where(train['Gender'].isnull(),'Male',train['Gender'])
train['Gender']=new_gender
train['Gender'].isnull().sum()

### in Married ###
train['Married'].value_counts()
train['Married']=np.where(train['Married'].isnull(),'Yes',train['Married'])
train['Married'].isnull().sum()

### in Dependents ###
train['Dependents'].isnull().sum()
train['Dependents'].value_counts()
train['Dependents']=train['Dependents'].replace('3+',3)
train['Dependents']=train['Dependents'].replace('0',0)

train['Dependents']=np.where(train['Dependents'].isnull(),0,train['Dependents'])
train['Dependents'].isnull().sum()
train['Dependents']=pd.to_numeric(train['Dependents'])
train['Dependents'].dtypes ##3 to check the data types

##### Self_Employed ####
train['Self_Employed'].value_counts()
train['Self_Employed']=np.where(train['Self_Employed'].isnull(),'No',train['Self_Employed'])
train['Self_Employed'].isnull().sum()

### in Loan_Amount ####
train['LoanAmount'].value_counts()
train['LoanAmount']=np.where(train['LoanAmount'].isnull(),np.nanmedian(train['LoanAmount']),
     train['LoanAmount'])
train['LoanAmount'].isnull().sum()

### Loan Amount Term ###
train['Loan_Amount_Term']=np.where(train['Loan_Amount_Term'].isnull(),'360',train['Loan_Amount_Term'])
train['Loan_Amount_Term'].isnull().sum()

### in Credit_History ###
train['Credit_History'].value_counts()
train['Credit_History']=np.where(train['Credit_History'].isnull(),'1',train['Credit_History'])
train['Credit_History'].isnull().sum()

#### Encoding the variable #### change the variable to binary
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

train['Gender']=LE.fit_transform(train['Gender'])
train['Gender'].value_counts()

train['Married']=LE.fit_transform(train['Married'])
train['Married'].value_counts()

train['Education']=LE.fit_transform(train['Education'])
train['Education'].value_counts()

train['Self_Employed']=LE.fit_transform(train['Self_Employed'])
train['Self_Employed'].value_counts()

train['Property_Area']=LE.fit_transform(train['Property_Area'])
train['Property_Area'].value_counts()

train['Loan_Status']=LE.fit_transform(train['Loan_Status'])
train['Loan_Status'].value_counts()

#### Model ###

Y=train['Loan_Status']

X=train.iloc[:,1:11]

from sklearn.linear_model import LogisticRegression

Log_Reg = LogisticRegression()

Log_Reg.fit(X,Y)

Preds_Log_Reg=Log_Reg.predict(X)

from sklearn.metrics import confusion_matrix

cm_log =confusion_matrix(Y,Preds_Log_Reg)

print(cm_log)

##### Random Forest #####

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=500)

RF.fit(X,Y)

Preds_RF=RF.predict(X)

cm_rf=confusion_matrix(Y,Preds_RF)

print(cm_rf)

#### Navie Bayes ####

from sklearn.naive_bayes import GaussianNB

NB=GaussianNB()
NB.fit(X,Y)
Preds_NB=NB.predict(X) ### Prediction ###

cm_nb=confusion_matrix(Y,Preds_NB)
print(cm_nb)


##############3 Test ############

test.isnull().sum()

### Imput the missing value in Test ###
## in Gender ##3

test['Gender'].value_counts()
new_gender=np.where(test['Gender'].isnull(),'Male',test['Gender'])
test['Gender']=new_gender
test['Gender'].isnull().sum()
test.isnull().sum()

### in Dependents ###
test['Dependents'].isnull().sum()
test['Dependents'].value_counts()
test['Dependents']=test['Dependents'].replace('3+',3)
test['Dependents']=test['Dependents'].replace('0',0)

test['Dependents']=np.where(test['Dependents'].isnull(),0,test['Dependents'])
test['Dependents'].isnull().sum()
test['Dependents']=pd.to_numeric(test['Dependents'])
test['Dependents'].dtypes ##3 to check the data types

##### Self_Employed ####
test['Self_Employed'].value_counts()
test['Self_Employed']=np.where(test['Self_Employed'].isnull(),'No',test['Self_Employed'])
test['Self_Employed'].isnull().sum()

### in Loan_Amount ####
test['LoanAmount'].value_counts()
test['LoanAmount']=np.where(test['LoanAmount'].isnull(),np.nanmedian(test['LoanAmount']),
     test['LoanAmount'])
test['LoanAmount'].isnull().sum()

### Loan Amount Term ###
test['Loan_Amount_Term']=np.where(test['Loan_Amount_Term'].isnull(),'360',test['Loan_Amount_Term'])
test['Loan_Amount_Term'].isnull().sum()

### in Credit_History ###
test['Credit_History'].value_counts()
test['Credit_History']=np.where(test['Credit_History'].isnull(),'1',test['Credit_History'])
test['Credit_History'].isnull().sum()

#### Encoding the variable #### change the variable to binary
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

test['Gender']=LE.fit_transform(test['Gender'])
test['Gender'].value_counts()

test['Married']=LE.fit_transform(test['Married'])
test['Married'].value_counts()

test['Education']=LE.fit_transform(test['Education'])
test['Education'].value_counts()

test['Self_Employed']=LE.fit_transform(test['Self_Employed'])
test['Self_Employed'].value_counts()

test['Property_Area']=LE.fit_transform(test['Property_Area'])
test['Property_Area'].value_counts()

test['Loan_Status']=LE.fit_transform(test['Loan_Status'])
test['Loan_Status'].value_counts() #### Not in Test Data ###

test['LoanAmount'].isnull().sum()

###### Test Model #####

test=pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

Test_data=test.iloc[:,0:14]

cm_test=RF.predict(Test_data) ##### error####

Test_data.to_csv('output.csv')




