##This program is the compute the Y2 Binary Variable for the logistic regression assignment
##Batch 13 : Harsha, Vinod & James

rm(list=ls(all=TRUE))

install.packages("vegan")
install.packages("Amelia")
install.packages("mice")
install.packages("caret")
install.packages("MASS")
install.packages("glmnet")
install.packages("DMwR")
install.packages("Metrics")
library(Metrics)
library(MASS)
library(car)
library(caret)
library(mice)
library(Amelia)
library(vegan)
library(glmnet)
library(DMwR)
library(ROCR)

# Setting the working Directory 
setwd("E:/DATA SCIENCE/CUTE_02/CUTe-Code Final")
getwd()

#Reading the data from the data files
train <-read.csv("data13.csv")
test  <-read.csv("test13.csv")

dim(train)
dim(test)

sum(is.na(train))

#Removing target variable from train data
train_data_without_target         <- train[,-c(1,110,111)]
dim(train_data_without_target)
str(train_data_without_target)

test_data_without_target         <- test[,-c(1)]
dim(test_data_without_target)
str(test_data_without_target)

#Binding both test and train data
new_data<-rbind(train_data_without_target,test_data_without_target)
dim(new_data)

#Missing value imputation
miss<-sum(is.na(new_data))
miss
library(DMwR)

#Imputing missing value with Central Imputation(Mean)
imp_data<-centralImputation(new_data)
miss<-sum(is.na(imp_data))
miss


#Imputing missing value with knn Imputation
#imp_data_knn<-knnImputation(new_data)

#Standardization of data
library(vegan)
std_data<-decostand(imp_data,"standardize")
std_train_data<-std_data[1:1769,]
std_test_data<-decostand(test,"standardize")


get_back_train = new_data[1:1769,]
dim(get_back_train)

y2<-subset(train,select=c(y2))
std_train_data_with_y2<-cbind(get_back_train,y2)
set.seed(100) #to take same sample data 

#Dividing train data into train and validation  to predict the accuracy

train_rows<-sample(1:nrow(std_train_data_with_y2),0.7*nrow(std_train_data_with_y2))

train.cl<-std_train_data_with_y2[train_rows,]
val.cl<-std_train_data_with_y2[-train_rows,]


#Builiding logistic  model

stock_mod<-glm(y2~.,data = train.cl, family = "binomial")
library(MASS)
stock_mod

#Step AIC of the model
step_mod<-stepAIC(stock_mod)
step_mod

#Prediction of y2 values
pred<-predict(step_mod,newdata=val.cl, type = "response")
tab<-table(pred>0.5,val.cl$y2)
accu<-sum(diag(tab))/sum(tab)
accu

#This is where we have pulled out the predictions
########################################
test_pred<-predict(step_mod,newdata=test, type = "response")
test_pred[test_pred>0.55]<-1
test_pred[test_pred<0.55]<-0
write.csv(test_pred, "102y2_predictions.csv")

####################################


#Building a confusion Matrix
library(caret)
confusionMatrix(pred,val.cl$y2)
library(ROCR)
pred1<-prediction(pred,val.cl$y2)
eval<-performance(pred1,"acc")

#Roc curve 
roc<-performance(pred1,"tpr","fpr")
plot(roc,colorize=T,main="ROC Curve",ylab="sensitivity",xlab="1-specificity",print.cutoffs.at=seq(0,1,0.05))

auc<-performance(pred1,"auc")   #Area Under Curve
auc
max<-which.max(slot(eval,"y.values")[[1]])
acc<-slot(eval,"y.values")[[1]][max]
acc
cut<-slot(eval,"x.values")[[1]][max]
cut

#with step AIC

pred1<-predict(step_mod,newdata=val.cl, type = "response")
tab1<-table(pred1>0.5,val.cl$y2)
acc1<-sum(diag(tab1))/sum(tab1)
acc1

#accuracy with   Step AIC is better 




###############################################################################
#prediction of y1
###############################################################################


##This program is the compute the Y2 Binary Variable for the logistic regression assignment
##Batch 13 : Harsha, Vinod & James

rm(list=ls(all=TRUE))

# install.packages("vegan")
# install.packages("Amelia")
# install.packages("mice")
# install.packages("caret")
# install.packages("MASS")
# install.packages("glmnet")
# install.packages("DMwR")
#install.packages("Metrics")
library(Metrics)
library(MASS)
library(car)
library(caret)
library(mice)
library(Amelia)
library(vegan)
library(glmnet)
library(DMwR)
library(ROCR)

# Setting the working Directory 
setwd("Set working directory")
getwd()

#Reading the data from the data files
train <-read.csv("data13.csv")
test  <-read.csv("test13.csv")
dim(train)
dim(test)


#Removing target variable from train data
train_data_without_target <- train[,-c(1,110,111)]
dim(train_data_without_target)
str(train_data_without_target)

test_data_without_target         <- test[,-c(1)]
dim(test_data_without_target)
str(test_data_without_target)
dim(test_data_without_target)

#Binding both test and train data
new_data<-rbind(train_data_without_target,test_data_without_target)
dim(new_data)

#new_data=train_data_without_target

#Missing value imputation
miss<-sum(is.na(new_data))

miss
library(DMwR)

#Imputing missing value with Central Imputation(Mean)
imp_data<-centralImputation(new_data)
miss<-sum(is.na(imp_data))
miss

#Standardization of data
library(vegan)
std_data<-decostand(imp_data,"standardize")
dim(std_data)

#Now we get back our train data
std_train_data<-std_data[1:1769,]
dim(std_train_data)
#Now we get back our test data
std_test_data <-std_data[1770:1799,]
dim(std_test_data)

set.seed(100)

y1=train$y1
std_train_data= cbind(std_train_data,y1)
dim(std_train_data)

train_rows<-sample(1:nrow(std_train_data),0.75*nrow(std_train_data))
train.cl<-std_train_data[train_rows,]
val.cl<-std_train_data[-train_rows,]

dim(train.cl)
dim(val.cl)

lm_1 = lm(y1~.,train.cl)  
summary(lm_1)  #Multiple R-squared:  0.1449,	Adjusted R-squared:  0.0648 

#Therefore we are going in for stepAIC

lm_pred =predict(lm_1,train.cl)

plot(lm_1)

            
err = regr.eval(train.cl$y1,lm_1$fitted.values )
err
## The resultant values are
## mae          mse         rmse         mape 
## 0.0120856863 0.0002657448 0.0163016795 3.0312138484 

y1_stepout = stepAIC(lm_1, direction="both")

#Step:  AIC=-10098.39
#y1 ~ d_0 + f_0 + f_1 + f_2 + f_3 + f_5 + f_8 + f_9 + f_10 + f_12 + 
#  f_13 + f_14 + f_16 + f_19 + f_21 + f_22 + f_23 + f_24 + f_25 + 
#  f_26 + f_29 + f_31 + f_32 + f_34 + f_35 + f_38 + f_43 + f_44 + 
#  f_46 + f_48 + f_49 + f_50 + f_51 + f_53 + f_54 + f_56 + f_57 + 
#  f_58 + f_60 + f_61 + f_63 + t_2 + t_7 + t_13 + t_14 + t_19 + 
#  t_20 + t_25 + t_28 + t_29 + t_30 + t_31 + t_38 + t_39 + t_40 + 
#  t_41 + t_42 + d_3
y1_pred = predict(y1_stepout, val.cl)
summary(y1_pred)

errors = regr.eval(val.cl$y1,y1_pred)
errors

hist(y1_pred)
plot(y1_pred)

#we got the values as given below and we do see that the model has improved and 
#the errors have decreased
#
#> errors
#mae          mse         rmse         mape 
#0.0131128933 0.0003271254 0.0180866089 3.0069529639 
#

#now writing the fitted values for test to test_prediction_y1.csv

y1_test_pred= predict(y1_stepout,std_test_data)
write.csv(y1_test_pred,"7thtest_prediction_y1.csv")

