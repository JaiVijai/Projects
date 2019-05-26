#Import the data (Train and Test)
train<-read.csv("C:\\Users\\Vijai\\Documents\\RRRR\\Loan Prediction\\train_u6lujuX_CVtuZ9i.csv",
                header=TRUE,sep=",",na.strings=c("")) 
View(train)
test<-read.csv("C:\\Users\\Vijai\\Documents\\RRRR\\Loan Prediction\\test_Y3wMUE5_7gLdaTN.csv",
               header=TRUE,sep=",",na.strings=c(""))
View(test)

summary(train)
str(train)
library(Hmisc)
library(e1071)
skewness(train$Loan_Amount_Term)
skewness(train$LoanAmount)
describe(train)

### missing value in Loan amount ##
str(train)
sum(is.na(train$LoanAmount))
train$LoanAmount<-ifelse(is.na(train$LoanAmount),median(train$LoanAmount,na.rm=TRUE),train$LoanAmount)
sum(is.na(train$LoanAmount))
sum(is.na(train$Loan_Amount_Term))
skewness(train$LoanAmount)

### Gender ####
sum(is.na(train$Gender))
train$Gender<-as.character(train$Gender)
train$Gender<-ifelse(is.na(train$Gender),'Male',train$Gender)
sum(is.na(train$Gender))
train$Gender<-as.factor(train$Gender)

### Married ##
sum(is.na(train$Married))
train$Married<-as.character(train$Married)
train$Married<-ifelse(is.na(train$Married),'Yes',train$Married)
train$Married<-as.factor(train$Married)

## Loan amount term ##
sum(is.na(train$Loan_Amount_Term))
train$Loan_Amount_Term<-ifelse(is.na(train$Loan_Amount_Term),median(train$Loan_Amount_Term,na.rm=TRUE),train$Loan_Amount_Term)
sum(is.na(train$Loan_Amount_Term))

## Credit ##
sum(is.na(train$Credit_History))
train$Credit_History<-ifelse(is.na(train$Credit_History),'1',train$Credit_History)
sum(is.na(train$Credit_History))

## Self employment ##Education

sum(is.na(train$Self_Employed))
train$Self_Employed<-as.character(train$Self_Employed)    
train$Self_Employed<-ifelse(is.na(train$Self_Employed),'No',train$Self_Employed)
sum(is.na(train$Self_Employed))
train$Self_Employed<-as.factor(train$Self_Employed)    

## Dependents ##
str(train)
train$Dependents<-ifelse(is.na(train$Dependents),'0',train$Dependents)
sum(is.na(train$Dependents))

### Model ##

model<-glm(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
             ApplicantIncome+CoapplicantIncome+Loan_Amount_Term+LoanAmount+
             Credit_History+Property_Area,family='binomial',data=train)
summary(model)
train$preds <- predict(model, train,
                       type = 'response')
View(train)
train$outcome <- ifelse(train$preds>=0.5,'Y','N')
View(train)
table(train$Loan_Status,train$outcome)

(84+415)/(84+108+7+415)
## Random Forest ##

library('randomForest')
model_rf<-randomForest(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                         ApplicantIncome+CoapplicantIncome+Loan_Amount_Term+LoanAmount+
                         Credit_History+Property_Area,ntree=500,data=train)
model_rf
(86+404)/(86+106+18+404)

### Log function for  Applicantincome ###
skewness(train$ApplicantIncome)
skewness(train$CoapplicantIncome)

## log for Coapplicantincome ###
train$CoapplicantIncome<-as.integer((train$CoapplicantIncome))
train$ln_CoapplicantIncome<-log(train$CoapplicantIncome)
str(train)

train$ln_CoapplicantIncome<-ifelse(train$ln_CoapplicantIncome==-Inf,0,train$ln_CoapplicantIncome)
Name(train)

model_rf2<-randomForest(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                         ApplicantIncome+ln_CoapplicantIncome+Loan_Amount_Term+LoanAmount+
                         Credit_History+Property_Area,ntree=500,data=train)
model_rf2
(86+403)/(86+106+19+403)
## log for Applicantincome ##

train$ln_ApplicantIncome<-log(train$ApplicantIncome)

train$ln_ApplicantIncome<-ifelse(train$ln_ApplicantIncome==-Inf,o,train$ln_ApplicantIncome)
modle_rf3<-randomForest(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                          ln_ApplicantIncome+ln_CoapplicantIncome+Loan_Amount_Term+LoanAmount+
                          Credit_History+Property_Area,ntree=500,data=train)
modle_rf3
(85+402)/(85+107+20+402) 


### SVM model ###
library('e1071')
model_svm<-svm(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+Loan_Amount_Term+LoanAmount+Credit_History+Property_Area,data=train)
model_svm
train$svm<-predict(model_svm,train)
View(train)
table(train$Loan_Status,train$svm)
(85+416)/(85+107+6+416)
