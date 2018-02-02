#----------------------------------------------------------------------------------------------#
#                               Topic:"Telecom Customer Churn"                                 #
#----------------------------------------------------------------------------------------------#
# 
#Objective: To Predict the customers who are potentially going to churn 
#----------------------------------------------------------------------------------------------#
# Primary pupose is build a predictive model by doing manipulations on data to improve Accuracy 
# and recall by applying Machine Learning Classification models
#----------------------------------------------------------------------------------------------#
#
# AGENDA:
# ------
# 1. Source data
# 2. Data Charatcerization
# 3. Data Preprocessing
# 4. Data Exploring
# 5. Model Building
# 6. Evaluating model results
# 7. Tuning hyperparameters
# 8. Summary
#-----------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#
# 
#removing the previous existed global environment variables
rm(list=ls(all=TRUE))
#installing required libraries

#installing required libraries
library(mlr) #This library all required machine learning libraries
library(e1071) #Invoke naiveBayes method,SVM and other models
library(MLmetrics) #It will help to calulate all machine learning metrics
library(DMwR) #This package includes functions "DataMiningwithR" Central Imputation 
library(ggplot2) # Create Elegant Data Visulalisations Using the Grammar of Graphics
library(rpart)  # "Recursive Partitioning and Regression Trees"
library(caret)  # "Classification and Regression Training" library that contains xgboost models 
library(randomForest) #RandomForest implements Breiman's random forest algorithm for classification and regression
library(rpart.plot) #Plot an rpart model, automatically tailoring the plot for the model's response type. 
library(nnet) #Logistic regression 
library(glmnet) #These functions provide the base mechanisms for defining new functions in the R language
library(ROCR) #To plot ROC CURVE 
library(dummies) #contains functions,methods to creat,manipulate dummy variables
library(dummy) #Automatic Dummy Variable Creation with Support for Predictive Contexts
library(dplyr) #dplyr provides a flexible grammar of data manipulation.
library(Hmisc)#use the Hmisc package, you can take advantage of some labeling features.
library(lattice) #The lattice package is based on the Grid graphics engine, provides its own interface for querying
library(grid) #grid adds an nx by ny rectangular grid to an existing plot
library(MASS)
library(reshape2)
library(GoodmanKruskal) #to check the corelation between Categorical variables
#install.packages("ggthemes")
library(ggthemes)
library(gridExtra)
library(xgboost)
#install.packages("arules")
library(arules)
#install.packages("arulesViz")
library(arulesViz)
library (data.table)
library(plyr) 
library(MASS) 
library(reshape2)
library(corrplot)
library (stringr)
library(caret)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(MASS)

#Setting working directory 
setwd("G:/DATA SCIENCE/PHD/data_description (1)/20180120_PHD_Batch31_Classification_Patterns_ForStudents/TrainData")

#Step:1
#Reading given Traindata file into dataframe format 
actual_data1 <- read.csv("Train.csv",sep=",",header = T)
actual_data2 <- read.csv("Train_AccountInfo.csv",sep=",",header = T)
actual_data3 <- read.csv("Train_Demographics.csv",sep=",",header = T)
actual_data4 <- read.csv("Train_ServicesOptedFor.csv",sep=",",header = T)
dim(actual_data5)


#using library "reshape" with fuction decast to convert an molten "Train_ServicesOptedFor.csv" data into data frame.
actual_data5<- dcast(actual_data4, CustomerID ~ TypeOfService) 

#Used merge fuction to combine two data files into one data frame by taking primary keys in both the tables
merged_actual1<-merge(actual_data1,actual_data2,by.x = "CustomerID",by.y = "CustomerID")
merged_actual2<-merge(actual_data5,actual_data3,by.x = "CustomerID",by.y = "HouseholdID")
merged_actual3<-merge(merged_actual1,merged_actual2,by.x = "CustomerID",by.y = "CustomerID")

merged_train_data<-merged_actual3

#Reading given Test data files into dataframe format 
actual_test_data1 <- read.csv("Test.csv",sep=",",header = T)
actual_test_data2 <- read.csv("Test_AccountInfo.csv",sep=",",header = T)
actual_test_data3 <- read.csv("Test_Demographics.csv",sep=",",header = T)
actual_test_data4 <- read.csv("Test_ServicesOptedFor.csv",sep=",",header = T)

dim(actual_test_data3)


actual_test_data5<-dcast(actual_test_data4, CustomerID ~ TypeOfService)

merged_test_actual1<-merge(actual_test_data1,actual_test_data2,by.x = "CustomerID",by.y = "CustomerID")
merged_test_actual2<-merge(actual_test_data5,actual_test_data3,by.x = "CustomerID",by.y = "HouseholdID")
merged_test_actual3<-merge(merged_test_actual1,merged_test_actual2,by.x = "CustomerID",by.y = "CustomerID")

merged_test_data<-merged_test_actual3

#checking for dimentions and structure of merged data
dim(merged_test_data)
dim(merged_train_data)

str(merged_train_data)
str(merged_test_data)


#Step:2
#we found that dataset is having missing values which are repesented as ?,NA,MISSINGVAL,blank space
#so we replacing "?" "NA" "MISSINGVAL" " " with NA for missing values in traindata
merged_train_data[merged_train_data=="?"]<- NA 
merged_train_data[merged_train_data=="NA"]<- NA  
merged_train_data[merged_train_data==""]<-NA
merged_train_data[merged_train_data=="MISSINGVAL"]<-NA
merged_train_data[merged_train_data==" "]<-NA

#replacing "?" "NA" "MISSINGVAL" " " with NA for missing values in testdata
merged_test_data[merged_test_data=="?"]<- NA 
merged_test_data[merged_test_data=="NA"]<- NA  
merged_test_data[merged_test_data==""]<-NA
merged_test_data[merged_test_data=="MISSINGVAL"]<-NA
merged_test_data[merged_test_data==" "]<-NA

#Step:3
#recoding in Train data mapping 1 and 0 with "YES" and "NO"
merged_train_data$HasPhoneService <- as.factor(mapvalues(merged_train_data$HasPhoneService,from=c("0","1"),to=c("No", "Yes")))
merged_train_data$Retired <- as.factor(mapvalues(merged_train_data$Retired,from=c("0","1"),to=c("No", "Yes")))
merged_train_data$HasPartner <- as.factor(mapvalues(merged_train_data$HasPartner,from=c("1","2"),to=c("Yes", "No")))
merged_train_data$HasDependents <- as.factor(mapvalues(merged_train_data$HasDependents,from=c("1","2"),to=c("Yes", "No")))

#recoding in Test data mapping 1 and 0 with "YES" and "NO"
merged_test_data$HasPhoneService <- as.factor(mapvalues(merged_test_data$HasPhoneService,from=c("0","1"),to=c("No", "Yes")))
merged_test_data$Retired <- as.factor(mapvalues(merged_test_data$Retired,from=c("0","1"),to=c("No", "Yes")))
merged_test_data$HasPartner <- as.factor(mapvalues(merged_test_data$HasPartner,from=c("1","2"),to=c("Yes", "No")))
merged_test_data$HasDependents <- as.factor(mapvalues(merged_test_data$HasDependents,from=c("1","2"),to=c("Yes", "No")))


#Step:4
#Converting data types of Train data 
merged_train_data$DeviceProtection<-as.factor(merged_train_data$DeviceProtection)
merged_train_data$InternetServiceCategory<-as.factor(merged_train_data$InternetServiceCategory)
merged_train_data$MultipleLines<-as.factor(merged_train_data$MultipleLines)
merged_train_data$OnlineBackup<-as.factor(merged_train_data$OnlineBackup)
merged_train_data$OnlineSecurity<-as.factor(merged_train_data$OnlineSecurity)
merged_train_data$StreamingMovies<-as.factor(merged_train_data$StreamingMovies)
merged_train_data$StreamingTelevision<-as.factor(merged_train_data$StreamingTelevision)
merged_train_data$TechnicalSupport<-as.factor(merged_train_data$TechnicalSupport)
merged_train_data$TotalCharges<-as.numeric(merged_train_data$TotalCharges)

#Converting data types of Test data
merged_test_data$DeviceProtection<-as.factor(merged_test_data$DeviceProtection)
merged_test_data$InternetServiceCategory<-as.factor(merged_test_data$InternetServiceCategory)
merged_test_data$MultipleLines<-as.factor(merged_test_data$MultipleLines)
merged_test_data$OnlineBackup<-as.factor(merged_test_data$OnlineBackup)
merged_test_data$OnlineSecurity<-as.factor(merged_test_data$OnlineSecurity)
merged_test_data$StreamingMovies<-as.factor(merged_test_data$StreamingMovies)
merged_test_data$StreamingTelevision<-as.factor(merged_test_data$StreamingTelevision)
merged_test_data$TechnicalSupport<-as.factor(merged_test_data$TechnicalSupport)
merged_test_data$TotalCharges<-as.numeric(merged_test_data$TotalCharges)

# #droping levels train data
# merged_train_data$DeviceProtection[merged_train_data$DeviceProtection=="No internet service"]<-"No"
# merged_train_data$TechnicalSupport[merged_train_data$TechnicalSupport=="No internet service"]<-"No"
# merged_train_data$StreamingTelevision[merged_train_data$StreamingTelevision=="No internet service"]<-"No"
# merged_train_data$StreamingMovies[merged_train_data$StreamingMovies=="No internet service"]<-"No"
# merged_train_data$OnlineSecurity[merged_train_data$OnlineSecurity=="No internet service"]<-"No"
# merged_train_data$OnlineBackup[merged_train_data$OnlineBackup=="No internet service"]<-"No"
# merged_train_data$MultipleLines[merged_train_data$MultipleLines=="No phone service"]<-"No"

# #droping levels test data
# merged_test_data$DeviceProtection[merged_test_data$DeviceProtection=="No internet service"]<-"No"
# merged_test_data$TechnicalSupport[merged_test_data$TechnicalSupport=="No internet service"]<-"No"
# merged_test_data$StreamingTelevision[merged_test_data$StreamingTelevision=="No internet service"]<-"No"
# merged_test_data$StreamingMovies[merged_test_data$StreamingMovies=="No internet service"]<-"No"
# merged_test_data$OnlineSecurity[merged_test_data$OnlineSecurity=="No internet service"]<-"No"
# merged_test_data$OnlineBackup[merged_test_data$OnlineBackup=="No internet service"]<-"No"
# merged_test_data$MultipleLines[merged_test_data$MultipleLines=="No phone service"]<-"No"


#Step 7
str(merged_train_data)
#coverting date format and getting no of months of the customer tenure from date's
library(lubridate)
merged_train_data$DOC<-dmy(merged_train_data$DOC)
merged_train_data$DOE<-dmy(merged_train_data$DOE)
number_of_months<-(year(merged_train_data$DOC)-year(merged_train_data$DOE))*12+month(merged_train_data$DOC)-month(merged_train_data$DOE)
merged_train_data <-cbind(merged_train_data,number_of_months)

min(merged_train_data$number_of_months)
max(merged_train_data$number_of_months)

colnames(merged_train_data)

#coverting date format and getting no of months of the customer tenure from date's in test
merged_test_data$DOC<-dmy(merged_test_data$DOC)
merged_test_data$DOE<-dmy(merged_test_data$DOE)
number_of_months_test<-(year(merged_test_data$DOC)-year(merged_test_data$DOE))*12+month(merged_test_data$DOC)-month(merged_test_data$DOE)
merged_test_data <-cbind(merged_test_data,number_of_months_test)
max(merged_test_data$number_of_months_test)
min(merged_test_data$number_of_months_test)
colnames(merged_test_data)[25]<-c("number_of_months")


#Creating new column for months
group_tenure <- function(number_of_months)  {
  if (number_of_months >= 0 & number_of_months <= 6)  {
    return('0-6 Month')   }
  if (number_of_months >= 6 & number_of_months <= 12)  {
    return('0-12 Month')   }
  else if(number_of_months > 12 & number_of_months <= 24){
    return('12-24 Month')   }
  else if (number_of_months > 24 & number_of_months <= 48){
    return('24-48 Month')     }
  else if (number_of_months > 48 & number_of_months <=60){
    return('48-60 Month')     }
  else if (number_of_months > 60){
    return('> 60 Month')     }
}

merged_train_data$tenure_group <- sapply(merged_train_data$number_of_months,group_tenure)
merged_train_data$tenure_group <- as.factor(merged_train_data$tenure_group)

merged_test_data$tenure_group <- sapply(merged_test_data$number_of_months,group_tenure)
merged_test_data$tenure_group <- as.factor(merged_test_data$tenure_group)

#Step:8
#checking for missing values
library(DMwR)
sum(is.na(merged_train_data))
sum(is.na(merged_test_data))

sapply(merged_train_data, function(x) sum(is.na(x)))
sapply(merged_test_data, function(x) sum(is.na(x)))

percentge_missing_val<-sum(is.na(merged_train_data))/nrow(merged_train_data)*100
percentge_missing_val
per_miss_val<-(colSums(is.na(merged_train_data))/nrow(merged_train_data))*100
per_miss_val

#Handling missing values in train data
merged_train_data$TotalCharges[is.na(merged_train_data$TotalCharges)] <- 0
merged_train_data_imp<-centralImputation(merged_train_data)

sum(is.na(merged_train_data_imp))

merged_train_data<-(merged_train_data_imp)
sum(is.na(merged_train_data))

#merged_train_data_Gender<-centralImputation(merged_train_data$Gender)
#merged_train_data_Education<-centralImputation(merged_train_data$Education)
#merged_train_data_ContractType<-centralImputation(merged_train_data$ContractType)
library(DMwR)
#Handling missing values in test data
merged_test_data$TotalCharges[is.na(merged_test_data$TotalCharges)] <- 0
merged_test_data_imp<-centralImputation(merged_test_data)
sum(is.na(merged_test_data_imp))

merged_test_data<-merged_test_data_imp
sum(is.na(merged_test_data))
# merged_test_data$Gender<-centralImputation(merged_test_data$Gender)
# merged_test_data$Education<-centralImputation(merged_test_data$Education)
# merged_test_data$ContractType<-centralImputation(merged_test_data$ContractType)
# merged_test_data$Country<-centralImputation(merged_test_data$Country)

dim(merged_train_data)
dim(merged_test_data)

#Step: 9
#Drop unnecessary variables  
droplevels(merged_train_data)->merged_train_data
droplevels(merged_test_data)->merged_test_data

#merged_train_data <- merged_train_data[complete.cases(merged_train_data), ]
#merged_test_data <- merged_test_data[complete.cases(merged_test_data), ]

#Step: 10
#Creating Total charges binning 
TotalCharges_range <- function(TotalCharges)  {
    if (TotalCharges > 0 & TotalCharges <= 100)  {
      return('90-100 range')   }
    else if(TotalCharges > 100 & TotalCharges <= 200){
      return('100-200 range')   }
    else if(TotalCharges> 200 & TotalCharges <= 300){
      return('200-300 range')   }
    else if(TotalCharges > 300 & TotalCharges <= 500){
      return('300-500 range')   }
    else if(TotalCharges > 500 & TotalCharges <= 2000){
      return('500-2000 range')   }
    else if(TotalCharges > 2000 & TotalCharges <= 3000){
      return('2000-3000 range')   }
    else if(TotalCharges > 3000 & TotalCharges <= 4000){
      return('3000-4000 range')   }
    else if (TotalCharges > 4000) {
      return('> 4000 above')    }
  }
  
merged_train_data$totalcharge_range <- sapply(merged_train_data$TotalCharges,TotalCharges_range)
merged_test_data$totalcharge_range <- sapply(merged_test_data$TotalCharges,TotalCharges_range)


#Creating BASE Charges binning 
baseCharge_range <- function(BaseCharges)  {
  if (BaseCharges > 90 & BaseCharges <= 100)  {
    return('90-100 range')   }
  else if(BaseCharges > 100 & BaseCharges <= 200){
    return('100-200 range')   }
  else if(BaseCharges > 200 & BaseCharges <= 300){
    return('200-300 range')   }
  else if(BaseCharges > 300 & BaseCharges <= 400){
    return('300-400 range')   }
  else if (BaseCharges > 400) {
    return('> 400 above')    }
}

merged_train_data$baseCharge_range <- sapply(merged_train_data$BaseCharges,baseCharge_range)
merged_test_data$baseCharge_range <- sapply(merged_test_data$BaseCharges,baseCharge_range)

merged_train_data$baseCharge_range<-as.factor(merged_train_data$baseCharge_range)
merged_test_data$baseCharge_range<-as.factor(merged_test_data$baseCharge_range)


head(merged_train_data)
head(merged_test_data)

str(merged_train_data)
str(merged_test_data)

names(merged_train_data)
names(merged_test_data)


#Step: 11
#//////////////////////////
#// Data Visualisation////
#/////////////////////////

#Box plot of continous variables 
boxplot(merged_train_data$BaseCharges, main= "Box plot of Base Charges", ylab= "Range", xlab="Base charges")
boxplot(merged_train_data$TotalCharges, main="Box plot of Total charges", ylab="Range", xlab="Total Charges")
boxplot(merged_train_data$BaseCharges,merged_train_data$TotalCharges, main="Box plot of numerical vairables", xlab="Base charges, Total Charges")

#Target variabel Churn plot
table(merged_train_data$Churn)
ggplot(data = merged_train_data,mapping=aes(x=(Churn),fill=Churn),colors())+geom_bar()

#Base Charges plot
hist(merged_train_data$BaseCharges,col = blues9, main =" Base Charge frequency", xlab="Base charges range")
hist(merged_train_data$TotalCharges,col = blues9, main= "Total charges frequecy")
plot(merged_train_data$HasPhoneService,merged_train_data$Churn)
plot(merged_train_data$HasDependents,merged_train_data$Churn)
plot(merged_train_data$InternetServiceCategory,merged_train_data$Churn)
plot(merged_train_data$HasPartner,merged_train_data$Churn)
plot(merged_train_data$ContractType,merged_train_data$Churn)


ggplot(data = merged_train_data,mapping=aes(x=(Churn),fill=Churn),colors())+geom_bar()+ggtitle("No of customers Churn or not") + xlab("Churn or not") 

ggplot(data = merged_train_data,mapping=aes(x=(ElectronicBilling),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(Gender),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(Education),fill=Churn),colors())+geom_bar()+ggtitle("Based on education customers Churn or not")

ggplot(data = merged_train_data,mapping=aes(x=(PaymentMethod),fill=PaymentMethod),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(ContractType),fill=ContractType),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(HasDependents),fill=HasDependents),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(HasPartner),fill=HasPartner),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(Retired),fill=Retired),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(TechnicalSupport),fill=TechnicalSupport),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(StreamingMovies),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(StreamingTelevision),fill=StreamingTelevision),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(DeviceProtection),fill=DeviceProtection),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(HasPhoneService),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(InternetServiceCategory),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(MultipleLines),fill=MultipleLines),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(OnlineBackup),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(OnlineSecurity),fill=OnlineSecurity),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(tenure_group),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(tenure_group),fill=baseCharge_range),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(Education),fill=Churn),colors())+geom_bar()

ggplot(data = merged_train_data,mapping=aes(x=(ElectronicBilling),fill=Churn),colors())+geom_bar()



library(sqldf)

churn_totalcharges<-sqldf("Select count(Churn) from merged_train_data where TotalCharges < 100 ")
churn_totalcharges

sqldf("Select count(Churn) from merged_train_data where education ='Highschool or below' and ContractType='Month-to-month' and Churn='Yes' ")

sqldf("Select count(Churn) from merged_train_data where education in ('Highschool or below','Graduation') and ContractType='Month-to-month' and Churn='Yes' ")

sqldf("Select count(Churn) from merged_train_data where Churn='Yes' union Select count(Churn) from merged_train_data where Churn='No'")



dim(merged_train_data)
#Step 12:
#Removing unnecessory variable
merged_train_data<- subset(merged_train_data, select = -c(CustomerID,Country,State,DOE,DOC,BaseCharges,totalcharge_range,number_of_months))
merged_test_data <- subset(merged_test_data, select = -c(CustomerID,Country,State,DOE,DOC,BaseCharges,totalcharge_range,number_of_months))

#writing preprocessed data to local system
#write.csv(merged_train_data,"train_data_26012018.csv")
#write.csv(merged_test_data,"test_data_26012018.csv")

colnames(merged_train_data)
str(merged_train_data)
 
#Step: 13
#divide into data in to training,test 80,20 proportions
set.seed(1234)
proportion_data <- sample(seq(1,2),size = nrow(merged_train_data),replace = TRUE, prob = c(.7, .3))
train_imp_data <- merged_train_data[proportion_data == 1,]
val_imp_data <- merged_train_data[proportion_data == 2,]
test_imp_data <- merged_test_data

names(train_imp_data)
dim(train_imp_data)
dim(val_imp_data)
dim(test_imp_data)

#colnames(train_imp_data)[1]<-c("BaseCharges")
#colnames(val_imp_data)[1]<-c("BaseCharges")
#colnames(test_imp_data)[1]<-c("BaseCharges")

#----------------------------Logistic regression model--------------------------------------------
logistic_model1<-glm(Churn~.,data=train_imp_data,family=binomial)
summary(logistic_model1)

confint(logistic_model1)
anova(logistic_model1, test="Chisq")

log_predict_train1 <- predict(logistic_model1,train_imp_data,type = 'response')
log_train_pred1  <- ifelse(log_predict_train1 > 0.3, "Yes", "No")
confusionMatrix(log_train_pred1,train_imp_data$Churn,positive = 'Yes')

log_predict_val1 <- predict(logistic_model1,val_imp_data,type = 'response')
log_val_pred1  <- ifelse(log_predict_val1 > 0.3, "Yes", "No")
confusionMatrix(log_val_pred1,val_imp_data$Churn,positive = 'Yes')

log_predict_test1 <- predict(logistic_model1,test_imp_data,type = "response")
log_test_pred1  <- ifelse(log_predict_test1 > 0.3, "Yes", "No")
table(log_test_pred1)


names(train_imp_data)
#model 2 Logistic
logistic_model2<-glm(Churn~ TotalCharges+ContractType+Education+tenure_group,data=train_imp_data,family=binomial)

log_predict_train2 <- predict(logistic_model2,train_imp_data,type = 'response')
log_train_pred2  <- ifelse(log_predict_train2 > 0.3, "Yes", "No")
confusionMatrix(log_train_pred2,train_imp_data$Churn,positive = 'Yes')

log_predict_val2 <- predict(logistic_model2,val_imp_data,type = 'response')
log_val_pred2  <- ifelse(log_predict_val2 > 0.3, "Yes", "No")
confusionMatrix(log_val_pred2,val_imp_data$Churn,positive = 'Yes')

log_predict_test2 <- predict(logistic_model2,test_imp_data,type = "response")
log_test_pred2  <- ifelse(log_predict_test2 > 0.3, "Yes", "No")
table(log_test_pred2)


Accuracy(log_train_pred1,train_imp_data$Churn)
Recall(log_train_pred1,train_imp_data$Churn)
Precision(log_train_pred1,train_imp_data$Churn)

Accuracy(log_val_pred1,val_imp_data$Churn)
Recall(log_val_pred1,val_imp_data$Churn)
Precision(log_val_pred1,val_imp_data$Churn)


Accuracy(log_train_pred2,train_imp_data$Churn)
Recall(log_train_pred2,train_imp_data$Churn)
Precision(log_train_pred2,train_imp_data$Churn)

Accuracy(log_val_pred2,val_imp_data$Churn)
Recall(log_val_pred2,val_imp_data$Churn)
Precision(log_val_pred2,val_imp_data$Churn)



# tab<-table(log_train_pred1,train_imp_data$Churn)
# tab
# 
# #correct classification percentage 
# sum(diag(tab))/sum(tab) *100
# 
# #miss classification percentage
# (1-sum(diag(tab))/sum(tab))*100
# 
# table(train_imp_data$Churn)
# 
# 2798/(2798+939)
# 
# table(merged_train_data$Churn)
# 
# (3974/5298)#+1324)
# 
# #model preformance evaluation
# library(ROCR)
# 
#head(log_predict_val1)
#head(val_imp_data)

hist(log_predict_val1,col = blues9)

CustomerID<-as.data.frame(merged_test_actual3$CustomerID)
Churn <- as.data.frame(log_test_pred1)

predictions <- cbind(CustomerID,Churn)
colnames(predictions)<- c("CustomerID","Churn")
names(predictions)

write.csv(predictions,"predictions_26_2.csv",row.names = F)

#-----------------------------------------------------------------------
step_Model<-stepAIC(logistic_model1)
predict_step<-predict(step_Model,val_imp_data)
pred_step_porb<- ifelse(predict_step > 0.3, "Yes", "No")
confusionMatrix(pred_step_porb,val_imp_data$Churn,positive = "Yes")

#-----------------------------------------------------------------------------------------------------
rpart_model2<-rpart(Churn~.,train_imp_data)
printcp(rpart_model2)
plotcp(rpart_model2)
rpart.plot(rpart_model2,type=1,main="Rpart Plot")

rpart.plot(rpart_model2,type=2,main="Rpart right split plot")
summary(rpart_model2)

pd_train_rpart2<-predict(rpart_model2,train_imp_data,"class")
pd_val_rpart2<-predict(rpart_model2,val_imp_data,"class")
pd_test_rpart2<-predict(rpart_model2,test_imp_data,"class")

confusionMatrix(pd_train_rpart2,train_imp_data$Churn,positive = 'Yes')
confusionMatrix(pd_val_rpart2,val_imp_data$Churn,positive = 'Yes')

Accuracy(pd_train_rpart2,train_imp_data$Churn)
Recall(pd_train_rpart2,train_imp_data$Churn)
Precision(pd_train_rpart2,train_imp_data$Churn)

Accuracy(pd_val_rpart2,val_imp_data$Churn)
Recall(pd_val_rpart2,val_imp_data$Churn)
Precision(pd_val_rpart2,val_imp_data$Churn)

write.csv(pd_test_rpart2,"rpart_test_pd_26012018.csv")

#-----------------------------------------------------------------------------------------------------
library(e1071)
naive_model<-naiveBayes(Churn~.,train_imp_data)

naive_prob_train<-predict(naive_model,train_imp_data,type="raw") 
naive_train_pred<- ifelse(naive_prob_train > 0.3, "Yes", "No")

naive_prob_val<-predict(naive_model,val_imp_data,type="raw") 
naive_val_pred<- ifelse(naive_prob > 0.3, "Yes", "No")

naive_prob_test<-predict(naive_model,test_imp_data,type="raw") 
naive_test_pred<- ifelse(naive_prob_test > 0.3, "Yes", "No")

confusionMatrix(naive_train_pred[,2],train_imp_data$Churn,positive = "Yes")
confusionMatrix(naive_val_pred[,2],val_imp_data$Churn,positive = "Yes")

Accuracy(naive_train_pred[,2],train_imp_data$Churn)
Recall(naive_train_pred[,2],train_imp_data$Churn)
Precision(naive_train_pred[,2],train_imp_data$Churn)

Accuracy(naive_val_pred[,2],val_imp_data$Churn)
Recall(naive_val_pred[,2],val_imp_data$Churn)
Precision(naive_val_pred[,2],val_imp_data$Churn)


#------------------------------------------------------------------------------------
#random forest
randomforest_model<-randomForest(Churn~.,train_imp_data,ntree=100,mtry=5)
summary(randomforest_model)
var_imortance<-varImp(randomforest_model)
varImpPlot(randomforest_model)

random_train_pred<-predict(randomforest_model,train_imp_data)
random_pred<-predict(randomforest_model,val_imp_data)


#confusionMatrix(random_pred,train_imp_data$Churn,positive = "Yes")
confusionMatrix(random_pred,val_imp_data$Churn,positive = "Yes")

Accuracy(random_train_pred,train_imp_data$Churn)
Recall(random_train_pred,train_imp_data$Churn)
Precision(random_train_pred,train_imp_data$Churn)

Accuracy(random_pred,val_imp_data$Churn)
Recall(random_pred,val_imp_data$Churn)
Precision(random_pred,val_imp_data$Churn)


#-----------------------------------------------------------------------------------

#SVM
svmfit1<-svm(train_imp_data$Churn~.,train_imp_data,kernal="linear")

pd_train_svmfit1<-predict(svmfit1,train_imp_data)
table(pd_train_svmfit1)

pd_val_svmfit1<-predict(svmfit1,val_imp_data)
table(pd_val_svmfit1)

confusionMatrix(pd_train_svmfit1,train_imp_data$Churn,positive = "Yes")
confusionMatrix(pd_val_svmfit1,val_imp_data$Churn,positive = "Yes")


Accuracy(pd_train_svmfit1,train_imp_data$Churn)
Recall(pd_train_svmfit1,train_imp_data$Churn)
Precision(pd_train_svmfit1,train_imp_data$Churn)

Accuracy(pd_val_svmfit1,val_imp_data$Churn)
Recall(pd_val_svmfit1,val_imp_data$Churn)
Precision(pd_val_svmfit1,val_imp_data$Churn)


#--------------------Decesion tree--->----C50 Model----------------------------------

colnames(train_imp_data)
labels_train<-names(train_imp_data[,-1])
label_target<-names(train_imp_data[1])
#
treeModel <- C5.0(x= train_imp_data[,labels_train], y = train_imp_data$Churn,rules = T)
treeModel
summary(treeModel)

#
treeModel2 <- C5.0(x= train_imp_data[,labels2], y = train_imp_data$Churn,rules = T)
treeModel2
summary(treeModel2)
#

ruleModel <- C5.0(Churn ~ ., data = train_imp_data, rules = TRUE)
ruleModel
summary(ruleModel)

rule_mod <- C5.0(x = train_imp_data[,labels], y = train_imp_data$Churn, rules = TRUE)
rule_mod
summary(rule_mod)

pred_rule<-predict(rule_mod,test_imp_data[,labels])

table(pred_rule)

pred_tree_Modle2_rule<-predict(treeModel2,test_imp_data[,labels2])
table(pred_tree_Modle2_rule)

pred_tree_Modle_rule<-predict(treeModel,test_imp_data[,labels])
table(pred_tree_Modle_rule)


#--------------------------
#install.packages("MlBayesOpt")
#library(MlBayesOpt)


#rules
 

#results


#Summary








