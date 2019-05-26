import numpy as np
import pandas as pd
import os

test=pd.read_csv('testt.csv')
train=pd.read_csv('train.csv')


### Impute the missing value ###
trainmiss=train.isnull().sum()
testtmiss=test.isnull().sum()


### employee length ###

train['emp_length'].value_counts()
train['emp_length']=train['emp_length'].replace('10+ years','10 years')
train['emp_length']=np.where(train['emp_length'].isnull(),'10 years',train['emp_length'])

test['emp_length'].value_counts()
test['emp_length']=test['emp_length'].replace('10+ years','10 years')
test['emp_length']=np.where(test['emp_length'].isnull(),'10 years',test['emp_length'])

### emp title ###
train['emp_title'].value_counts()

### title ###
train['title'].value_counts()
train['title']=np.where(train['title'].isnull(),'Debt consolidation',train['title'])

test['title'].value_counts()
test['title']=np.where(test['title'].isnull(),'Debt consolidation',test['title'])

## mths_since_last_delinq ##
train['mths_since_last_delinq'].value_counts()
train['mths_since_last_delinq']=np.where(train['mths_since_last_delinq'].isnull(),
     np.nanmedian(train['mths_since_last_delinq']),train['mths_since_last_delinq'])

test['mths_since_last_delinq'].value_counts()
test['mths_since_last_delinq']=np.where(test['mths_since_last_delinq'].isnull(),
    np.nanmedian(test['mths_since_last_delinq']),test['mths_since_last_delinq'])

### mths_since_last_record #####
train['mths_since_last_record'].value_counts()
train['mths_since_last_record']=np.where(train['mths_since_last_record'].isnull(),
     np.nanmedian(train['mths_since_last_record']),train['mths_since_last_record'])

test['mths_since_last_record'].value_counts()
test['mths_since_last_record']=np.where(test['mths_since_last_record'].isnull(),
    np.nanmedian(test['mths_since_last_record']),test['mths_since_last_record'])

### revol_util ####
train['revol_util'].value_counts()
train['revol_util']=np.where(train['revol_util'].isnull(),'0.00',train['revol_util'])

test['revol_util'].value_counts()
test['revol_util']=np.where(test['revol_util'].isnull(),'0.0',test['revol_util'])

##### last_pymnt_d ####
train['last_pymnt_d'].value_counts()
train['last_pymnt_d']=np.where(train['last_pymnt_d'].isnull(),'16-Jan',train['last_pymnt_d'])

test['last_pymnt_d'].value_counts()
test['last_pymnt_d']=np.where(test['last_pymnt_d'].isnull(),'Jan-16',test['last_pymnt_d'])

### next_pymnt_d ###
train['next_pymnt_d'].value_counts()
train['next_pymnt_d']=np.where(train['next_pymnt_d'].isnull(),'16-Feb',train['next_pymnt_d'])

test['next_pymnt_d'].value_counts()
test['next_pymnt_d']=np.where(test['next_pymnt_d'].isnull(),'Feb-16',test['next_pymnt_d'])

##### last_credit_pull_d ###
train['last_credit_pull_d'].value_counts()
train['last_credit_pull_d']=np.where(train['last_credit_pull_d'].isnull(),'16-Jan',train['last_credit_pull_d'])

test['last_credit_pull_d'].value_counts()
test['last_credit_pull_d']=np.where(test['last_credit_pull_d'].isnull(),'Jan-16',test['last_credit_pull_d'])

### collections_12_mths_ex_med ###
train['collections_12_mths_ex_med'].value_counts()
train['collections_12_mths_ex_med']=np.where(train['collections_12_mths_ex_med'],'0.0',
     train['collections_12_mths_ex_med'])

##### mths_since_last_major_derog ####
train['mths_since_last_major_derog'].value_counts() ### not imputed ###
test['mths_since_last_major_derog'].value_counts() #### need to check ###

##### tot_coll_amt #### 
train['tot_coll_amt'].value_counts()
train['tot_coll_amt']=np.where(train['tot_coll_amt'].isnull(),'0.0',train['tot_coll_amt'])

#### tot_cur_bal ####
train['tot_cur_bal'].value_counts()
train['tot_cur_bal']=np.where(train['tot_cur_bal'].isnull(),np.nanmedian(train['tot_cur_bal']),
     train['tot_cur_bal'])

#### total_rev_hi_lim ####
train['total_rev_hi_lim'].value_counts()
train['total_rev_hi_lim']=np.where(train['total_rev_hi_lim'].isnull(),np.nanmedian(train['total_rev_hi_lim']),
     train['total_rev_hi_lim'])


#### Imput the in consistance value ###
train['verification_status']=train['verification_status'].replace('Verified','Source Verified')
train['verification_status'].value_counts()

test['verification_status'].value_counts()
test['verification_status']=test['verification_status'].replace('Verified','Source Verified')

### Encoding the data #####
trainmissing=train.isnull().sum()

from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

train['term']=LE.fit_transform(train['term'])
train['term'].value_counts()

test['term']=LE.fit_transform(test['term'])
test['term'].value_counts()

train['grade']=LE.fit_transform(train['grade'])
train['grade'].value_counts()

test['grade']=LE.fit_transform(test['grade'])
test['grade'].value_counts()

train['sub_grade']=LE.fit_transform(train['sub_grade'])
train['sub_grade'].value_counts()

test['sub_grade']=LE.fit_transform(test['sub_grade'])
test['sub_grade'].value_counts()

train['emp_length']=LE.fit_transform(train['emp_length'])
train['emp_length'].value_counts()

test['emp_length']=LE.fit_transform(test['emp_length'])
test['emp_length'].value_counts()

train['home_ownership']=LE.fit_transform(train['home_ownership'])
train['home_ownership'].value_counts()

test['home_ownership']=LE.fit_transform(test['home_ownership'])
test['home_ownership'].value_counts()

train['verification_status']=LE.fit_transform(train['verification_status'])
train['verification_status'].value_counts()

test['verification_status']=LE.fit_transform(test['verification_status'])
test['verification_status'].value_counts()

train['pymnt_plan']=LE.fit_transform(train['pymnt_plan'])
train['pymnt_plan'].value_counts()

test['pymnt_plan']=LE.fit_transform(test['pymnt_plan'])
test['pymnt_plan'].value_counts()

train['purpose']=LE.fit_transform(train['purpose'])
train['purpose'].value_counts()

test['purpose']=LE.fit_transform(test['purpose'])
test['purpose'].value_counts()

train['zip_code'].dtypes

train['initial_list_status'].value_counts()
train['initial_list_status']=LE.fit_transform(train['initial_list_status'])
train['initial_list_status'].value_counts()

test['initial_list_status'].value_counts()
test['initial_list_status']=LE.fit_transform(test['initial_list_status'])
test['initial_list_status'].value_counts()

train['application_type']=LE.fit_transform(train['application_type'])
train['application_type'].value_counts()

test['application_type']=LE.fit_transform(test['application_type'])
test['application_type'].value_counts()

##### model ####
from sklearn.linear_model import LogisticRegression

LG=LogisticRegression()

Y=train['default_ind']
trainn=train.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,14,16,18,22,23,25,26,27,28,29,30,31,32,
                     33,34,35,36,37,38,39,40,41,42,44,47,49,50,54,55,56,68]]
X=trainn.iloc[:,0:42]

LG.fit(X,Y)

PredsLG=LG.predict(X)

from sklearn.metrics import confusion_matrix

cm_LG=confusion_matrix(Y,PredsLG)

(552781+44268)/(552781+41+1888+44268)  #### 99677 ###

#### Random Forest ###
from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier()

RF.fit(X,Y)

predsRF=RF.predict(X)

from sklearn.metrics import confusion_matrix

cmRF=confusion_matrix(Y,predsRF)

(552822+45814)/(552822+342+45814) #### 99942 ####

### Decision Tree ###

from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier()

DT.fit(X,Y)

predsDT=DT.predict(X)
from sklearn.metrics import confusion_matrix

cmDT=confusion_matrix(Y,predsDT) ##### 100 ##### 

#### navie Bayes #####
from sklearn.naive_bayes import GaussianNB 

NB=GaussianNB()

NB.fit(X,Y)

PredsNB=NB.predict(X)

cmNB=confusion_matrix(Y,PredsNB)
(551775+25015)/(551775+1047+21141+25015) ### 96295 ####

### SVM ###
from sklearn.svm import SVC

SVM=SVC()

SVM.fit(X,Y)

##### NN ##

from sklearn.neural_network import MLPClassifier

MLP=MLPClassifier(hidden_layer_sizes=(50,60,50),solver='adam',verbose=True,max_iter=300)



### Test Data ####

YY=test['default_ind']
testt=test.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,14,16,18,22,23,25,26,27,28,29,30,31,32,
                     33,34,35,36,37,38,39,40,41,42,44,47,49,50,54,55,56,68]]

XX=testt.iloc[:,0:42]

predsRF_test=RF.predict(XX)

cmRF_test=confusion_matrix(YY,predsRF_test)

(116166+308)/(116166+140514+3+308) ##### 4532 ####

#### Test Logistic Regression ###

predsLG_test=LG.predict(XX)

cmLG_test=confusion_matrix(YY,predsLG_test)
(256642+248)/(256642+38+63+248)    ##### 999606 ####

#### Decision Tree in Test ####

DT.fit(XX,YY)
perdsDT_test=DT.predict(XX)

cmDT_test=confusion_matrix(YY,perdsDT_test)

### Naive Bayes in Test ###

NB.fit(XX,YY)

predsNB_test=NB.predict(XX)

cmNB_test=confusion_matrix(YY,predsNB_test) 
(256574+14)/(256574+106+297+14) ##### 9984 ###


