---
output:
  pdf_document: default
  html_document: default
---
# Practical Machine Learning - Course Project
*Jayasree kulothungan*

# Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

# Importing libraries

```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```

# Reading the data

```{r}
train <- read.csv("pml-training.csv")
test<- read.csv("pml-testing.csv")
dim(train)
dim(test)
```
# Cleaning data

Remove all the data with missing values
``` {r}
sum(complete.cases(train))
sum(complete.cases(test))
trainData<- train[, colSums(is.na(train)) == 0]
testData <- test[, colSums(is.na(test)) == 0]
dim(trainData)
dim(testData)

```
Remove variables with less impact to the outcome
```{r}
trainData <- trainData[, -c(1:7)]
testData <- testData[, -c(1:7)]
dim(trainData)
dim(testData)
```
removing variables with near zero variance 

```{r}
NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]
dim(trainData)
dim(testData)
```

# Prepare the data for prediction

split the cleaned training set into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in future steps.

```{r}
set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
trainData <- trainData[inTrain, ]
testData1 <- trainData[-inTrain, ]
dim(trainData)
dim(testData1)
```
# Correlation Matrix Visualization  

The following correlation plot uses the following parameters (source:CRAN Package ‘corrplot’) “FPC”: the first principal component order. “AOE”: the angular order tl.cex Numeric, for the size of text label (variable names) tl.col The color of text label.

```{r}
cor_mat <- cor(trainData[, -53])
corrplot(cor_mat, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8)

```

# Data Modelling
For this project we will use two different algorithms
- classification trees
- random forests

## Random Forest
fit a predictive model for activity recognition using **Random Forest** algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use **5-fold cross validation** when applying the algorithm.  

```{r}
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF1$finalModel
```
estimate the performance of the model on the validation data set

```{r}

predictRF1 <- predict(modRF1, newdata=testData1)
cmrf <- confusionMatrix(predictRF1,as.factor(testData1$classe))
cmrf

```

```{r}
plot(modRF1)
```

```{r}
plot(cmrf$table, col = cmrf$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))
```


## Classification Tree Visualization

We first obtail the model, and then we use the fancyRpartPlot() function to plot the classification tree as a dendogram.
```{r}
set.seed(12345)
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
rpart.plot(decisionTreeMod1)
```
 
 validate the model “decisionTreeModel” on the testData to find out how well it performs by looking at the accuracy variable
 
```{r}
predictTreeMod1 <- predict(decisionTreeMod1, testData1, type = "class")
cmtree <- confusionMatrix(predictTreeMod1,as.factor(testData1$classe))
cmtree
```

plot matrix results

```{r}
plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Decision Tree - Accuracy =", round(cmtree$overall['Accuracy'], 4)))
```

# Result

Random Forest method is comparitively more accurate than Classification Tree