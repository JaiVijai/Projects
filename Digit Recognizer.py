import pandas as pd
import numpy as np
import os

## set work directory
os.chdir()

## read the data ##
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.isnull().sum()
testmissing=test.isnull().sum()

##### set X and Y
Y=train['label']
X=train.iloc[:,1:]

### split the date in train ###

from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
### random state is give the same record when ever we run ####

#### Scaling the data ####
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
SC.fit(Xtrain)
X_train=SC.transform(Xtrain)
X_test=SC.transform(Xtest)
######we do scaling or not todo scaling



from sklearn.neural_network import MLPClassifier

MLP=MLPClassifier(hidden_layer_sizes=(50,60,50),solver='adam',verbose=True,max_iter=300)

MLP.fit(Xtrain,Ytrain)

Y_preds_trainMLP=MLP.predict(Xtrain) ### Xtrain prediction ### 
Y_preds_testMLP=MLP.predict(Xtest) ##### Xtest prediction ####

from sklearn.metrics import confusion_matrix

trainMLP=confusion_matrix(Ytrain,Y_preds_trainMLP)
testMLP=confusion_matrix(Ytest,Y_preds_testMLP)

### Apply in Normal train data ####

from sklearn.neural_network import MLPClassifier

MLP=MLPClassifier(hidden_layer_sizes=(50,60,50),solver='adam',verbose=True,max_iter=300)

MLP.fit(X,Y)

preds_MLP=MLP.predict(X)

out_MLP=confusion_matrix(Y,preds_MLP)


#### Apply that in Normal test Data ####

test_data=test.iloc[:,0:]

test_outMLP=MLP.predict(test_data)











