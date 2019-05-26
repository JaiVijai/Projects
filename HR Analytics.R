### Import data ###
train<-read.csv('train_LZdllcl.csv',na.string=c(""))
test<-read.csv('test_2umaH9m.csv',na.strings = c(""))

sum(is.na(train$education))
describe(test)
str(train)
table(train$previous_year_rating)
## change the variable type##
train$KPIs_met..80.<-as.factor(train$KPIs_met..80.)
test$KPIs_met..80.<-as.factor(test$KPIs_met..80.)
train$is_promoted<-as.factor(train$is_promoted)
train$awards_won.<-as.factor(train$awards_won.)
test$awards_won.<-as.factor(test$awards_won.)
train$previous_year_rating<-as.factor(train$previous_year_rating)
test$previous_year_rating<-as.factor(test$previous_year_rating)

### treating the missing valu using Amelio ###

names(train)
library('Amelia')
new<-amelia(train)

### treating the missing value ##
sum(is.na(train))
sum(is.na(train$previous_year_rating))
train$previous_year_rating<-ifelse(is.na(train$previous_year_rating),'3',train$previous_year_rating)
table(train$previous_year_rating)
str(train)

sum(is.na(train$education))
train$education<-as.character(train$education)
table(train$education)

train$education<-ifelse(is.na(train$education),"Bachelor's",train$education)
View(train)

#### Exploratory Data analysis ###
table(train$department)
table(train$region)

#condition ###
names(train$department)
mean(train$age[train$department=='HR'])
mean(train$age[train$department=='Analytics'])
mean(train$age[train$department=='Finance'])

##most frequency region ###
library('plyr')
count('train','train$department')

### GLM model ###

names(train)

model_glm<-glm(is_promoted~education+gender+recruitment_channel+no_of_trainings+age+previous_year_rating+length_of_service+KPIs_met..80.+awards_won.+avg_training_score,family='binomial',data=train)
model_glm
train$pred<-predict(model_glm,train)
View(train)
train$out<-ifelse(train$pred>=0.5,'1','0')
table(train$is_promoted,train$out)
(50039+197)/(50039+101+4471+197)


### Decision treee ### in decision tree Char variable not supported

library('party')
str(train)
train$previous_year_rating<-as.factor(train$previous_year_rating)
train$education<-as.factor(train$education)
model_tree<-ctree(is_promoted~education+gender+recruitment_channel+no_of_trainings+age+previous_year_rating+length_of_service+KPIs_met..80.+awards_won.+avg_training_score,data=train)
model_tree
train$pred_tree<-predict(model_tree,train)
table(train$is_promoted,train$pred_tree)
(49870+872)/(49870+270+3796+872)

##### Random Forest ####
library('randomForest')
model_rff<-randomForest(is_promoted~education+gender+recruitment_channel+no_of_trainings+age+previous_year_rating+length_of_service+KPIs_met..80.+awards_won.+avg_training_score,ntree=500,data=train)

model_rff
(49973+730)/(49973+167+3938+730)

### SVM ###
library('e1071')
model_svm<-svm(is_promoted~education+gender+recruitment_channel+no_of_trainings+age+previous_year_rating+length_of_service+KPIs_met..80.+awards_won.+avg_training_score,data=train)
names(train)
train1<-subset(train,select=-c(employee_id,department,region,pred,out,pred_tree,out_tree))
train1$pred_svm<-predict(model_svm,train1)
table(train1$is_promoted,train1$pred_svm)
(50029+351)/(50029+111+4317+351)

#### Caret Packages ####
library('caret')
X<-c("department","region","education","gender","recruitment_channel","no_of_trainings","age","Previous_year_rating","length_of_service","KPIs_met..80.","awards_won.","avg_training_score")
Y<-c("is_promoted")
train2<-subset(train,select=-c(employee_id,pred,out,pred_tree,out_tree))
model_rf<-train2(train2[,X],train2[,Y],method='rf')



##### Test ####3
summary(test)
library('Hmisc')
describe(test)
sum(is.na(test))
str(test)
str(train)
sum(is.na(test$previous_year_rating))
sum(is.na(test$department))
sum(is.na(test$education))
table(test$previous_year_rating)

### missing value in Previous year ###
test$previous_year_rating<-ifelse(is.na(test$previous_year_rating),'3',test$previous_year_rating)

### missing value in education ###3
test$education<-as.character(test$education)
test$education<-ifelse(is.na(test$education),"Bachelor's",test$education)
test$education<-as.factor(test$education)
str(test)
test$previous_year_rating<-as.factor(test$previous_year_rating)
test$pred_tree<-predict(model_tree,test)
View(test)
write.csv(test,'finals.csv')


## Naive Bayes ##3
library('e1071')
model_nb<-naiveBayes(is_promoted~education+gender+recruitment_channel+no_of_trainings+age+previous_year_rating+length_of_service+KPIs_met..80.+awards_won.+avg_training_score,data=train)
train$pred_nb<-predict(model_nb,train)
table(train$is_promoted,train$pred_nb)
(49176+718)/(49176+964+3950+718)

### in model i included Department and Region in Naive Bayes ###
model_nb1<-naiveBayes(is_promoted~department+region+education+gender+recruitment_channel+no_of_trainings+age+previous_year_rating+length_of_service+KPIs_met..80.+awards_won.+avg_training_score,data=train)
train$pred_nb1<-predict(model_nb1,train)
table(train$is_promoted,train$pred_nb1)
(48895+818)/(48895+1245+3850+818) 

### combine all model ###

train$final<-ifelse(train$out=="1" & train$pred_tree=="1","1",
                    ifelse(train$out=="1" & train$pred_nb=="1","1",
                           ifelse(train$pred_tree=="1" & train$pred_nb=="1","1","0")))
table(train$is_promoted,train$final)
View(train)
(50026+465)/(50026+114+4203+465)
