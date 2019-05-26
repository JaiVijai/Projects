import pandas as pd
import numpy as np
import os

### impoet the data ####

train=pd.read_csv('train_LZdllcl.csv')
test=pd.read_csv('test_2umaH9m.csv')

### missing value ###
trainmiss=train.isnull().sum()

### missing education ##
train['education'].value_counts()
train['education']=np.where(train['education'].isnull(),"Bachelor's",train['education'])

## previous_year_rating ###
train['previous_year_rating'].value_counts()
train['previous_year_rating']=np.where(train['previous_year_rating'].isnull(),'3.0',
     train['previous_year_rating'])
train['previous_year_rating'].dtypes



### Encoding the train ###
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

train['department']=LE.fit_transform(train['department'])
train['department'].value_counts()

train['region']=LE.fit_transform(train['region'])
train['region'].value_counts()

train['education']=LE.fit_transform(train['education'])
train['education'].value_counts()

train['gender']=LE.fit_transform(train['gender'])
train['gender'].value_counts()

train['recruitment_channel']=LE.fit_transform(train['recruitment_channel'])

### set X & Y ###
Y=train['is_promoted']
X=train.iloc[:,1:13]

### split the data #### in train
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=40)

#### model ####
from sklearn.neural_network import MLPClassifier

MLP=MLPClassifier(hidden_layer_sizes = (50,50,50),solver='adam',verbose=True)

MLP.fit(Xtrain,Ytrain)

Preds_train=MLP.predict(Xtrain)
preds_test=MLP.predict(Xtest)

### Confusion matrix ####
from sklearn.metrics import confusion_matrix

trainMLP=confusion_matrix(Ytrain,Preds_train)
testMLP=confusion_matrix(Ytest,preds_test)


### another model ###
from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier()

RF.fit(Xtrain,Ytrain)

preds_trainRF=RF.predict(Xtrain)
preds_testRF=RF.predict(Xtest)

### Confussion Matrix ###
trainRF=confusion_matrix(Ytrain,preds_trainRF)
testRF=confusion_matrix(Ytest,preds_testRF)

### Apply in Test ####
testmiss=test.isnull().sum()
test['education'].value_counts()
test['education']=np.where(test['education'].isnull(),"Bachelor's",test['education'])

## previous_year_rating ###
test['previous_year_rating'].value_counts()
test['previous_year_rating']=np.where(test['previous_year_rating'].isnull(),'3.0',
     test['previous_year_rating'])

#### Encoding ####
test['department']=LE.fit_transform(test['department'])
train['department'].value_counts()

test['region']=LE.fit_transform(test['region'])
train['region'].value_counts()

test['education']=LE.fit_transform(test['education'])
train['education'].value_counts()

test['gender']=LE.fit_transform(test['gender'])
train['gender'].value_counts()

test['recruitment_channel']=LE.fit_transform(test['recruitment_channel'])

##### Model fit ###
testR=test.iloc[:,1:]

testRF=RF.predict(testR)


### Another model ###

from xgboost import XGBClassifier
XGB=XGBClassifier()
XGB.fit(Xtrain,Ytrain)


























