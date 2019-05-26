#Import  data (Test and Test) Titanic #
train<-read.csv("C:\\Users\\Vijai\\Documents\\RRRR\\kaggle-titanic-master\\input\\train.csv",
                header=TRUE,sep=",",na.strings=c("")) 
test<-read.csv("C:\\Users\\Vijai\\Documents\\RRRR\\kaggle-titanic-master\\input\\test.csv",
               header=TRUE,sep=",",na.strings=c("")) 


View(train)
str(train)
summary(train)
boxplot(train$Age)
library(Hmisc)
describe(train)
instal.packages('e1071')
library(e1071)
boxplot(train$Fare)
skewness(train$Fare)

#convert to factor#
train$Survived<-as.factor(train$Survived)
train$Pclass<-as.factor(train$Pclass)
train$Pclass<-as.integer(train$Pclass)
train$Embarked<-as.factor(train$Embarked)
test$Embarked<-as.factor((test$Embarked))
str(train)

#Missing value in AGE #
train$Age<-ifelse(is.na(train$Age),median(train$Age,na.rm=TRUE),train$Age)
View(train)
sum(is.na(train$Age))

#Missing Value in Embarked #
sum(is.na(train$Embarked))
table(train$Embarked)
train$Embarked<-as.character(train$Embarked)
train$Embarked<-ifelse(is.na(train$Embarked),"S",train$Embarked)
sum(is.na(train$Embarked))
str(train)

cor(train$Age,train$Fare)

# Model - Logistic Regression #

names(train)
model<-glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, 
           family='binomial',data = train)
model
summary(model)

train$preds<-predict(model,train,type = 'response')
View(train)


train$Outcome<-ifelse(train$preds>=0.5,1,0)

table(train$Survived,train$Outcome)

(477+240)/(477+72+102+240) ## Model give 80% accurce ##

###### --------------#########
# Another Mode - fare value Normalising #

train$ln_fare<-log(train$Fare)


train$ln_fare<-ifelse(train$ln_fare==-Inf,0,train$ln_fare)
View(train)
model2<-glm(Survived~Pclass+Sex+Age+SibSp+Parch+ln_fare+Embarked, 
            family='binomial',data = train)
View(train)
train$preds_model2<-predict(model2,train,type = 'response')
train$Outcome2<-ifelse(train$preds_model2>=0.5,1,0)
table(train$Survived,train$Outcome2)
(466+244)/(466+83+98+244)

#######    Test     #######

View(test)
str(test)
summary(test)
sum(is.na(test$Age))
sum(is.na(test$Embarked))
test$Age<-ifelse(is.na(test$Age),median(test$Age,na.rm=TRUE),test$Age)
sum(is.na(test$Fare))
test$Fare<-ifelse(is.na(test$Fare),median(test$Fare,na.rm = TRUE),test$Fare)

test$preds <- predict(model, test, type = 'response')
test$outcome <- ifelse(test$preds>=0.5,1,0)



write.csv(test,'finalpreds.csv')
getwd()

### Decision Tree ###

install.packages('party')
library('party')
png(file='decision_tree.png')
names(train)
str(train)
model_tree<-ctree(Survived~Pclass+Age+Sex+SibSp+Parch+Fare+Embarked,data=train)
model_tree

## Plot Test ##
plot(model_tree)
dev.off()
Summary(model_tree)
model_tree

#### Confusion Matrix - Decision tree ###
train$preds_modeltree<-predict(model_tree,train)
table(train$Survived,train$preds_modeltree)
(492+246)/(492+57+96+246)

###Random Forest ## Char is not taken ###

install.packages('randomForest')
library('randomForest')
str(train)
train$Embarked<-as.factor(train$Embarked)
model_rf<-randomForest(as.factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+
                         Embarked,ntree=500,data=train)
model_rf

(502+242)/(502+47+100+242)

#### SVM ###
library('e1071')
model_svm<-svm(Survived~Pclass+Age+Sex+SibSp+Parch+Fare+Embarked,cost=2,data=train)
model_svm
train$svm_preds<-predict(model_svm,train)### through ERROR ###

### Noe we remove the other variable are not used model##3
names(train)
train1<-subset(train,select=-c(PassengerId,Name,Ticket,Cabin))
train1$preds_svm<-predict(model_svm,train1)
table(train1$Survived,train1$preds_svm)
(494+249)/(494+55+93+249)

#### SVM in Test ###
test1$Embarked<-as.character(test1$Embarked)
test1<-subset(test,select=-c(PassengerId,Name,Ticket,Cabin))
test1$svm<-predict(model_svm,test1)
names(test1)
View(test1)
str(train1)
str(test1)
write.csv(test1,'final.csv')
