
# Practical Machine Learning
## Course Project
#### January 2015

The goal of this project is to use the data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants to quantify how well the participant performs barbell lifts. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The data for this analysis came from "Qualitative Activity Recognition of Weight Lifting Exercises" by Velloso et al. The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


### Download csv training and testing data

```r
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
```

### Read in the CSV data

```r
pml_train <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
pml_test <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
dim(pml_train)
```

```
## [1] 19622   160
```

```r
dim(pml_test)
```

```
## [1]  20 160
```

### Remove any columns that contain only missing values. 100 columns are removed from both sets.

```r
pml_train <- pml_train[,colSums(is.na(pml_train)) == 0]
pml_test <- pml_test[,colSums(is.na(pml_test)) == 0]
dim(pml_train)
```

```
## [1] 19622    60
```

```r
dim(pml_test)
```

```
## [1] 20 60
```

### Keep only the columns that contain the words belt, arm, dumbbell, and forearm in the name. This becomes all columns except for the first 7 columns.

```r
names(pml_train)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

```r
pml_train <- pml_train[,-c(1:7)]
pml_test <- pml_test[,-c(1:7)]
dim(pml_train)
```

```
## [1] 19622    53
```

```r
dim(pml_test)
```

```
## [1] 20 53
```

### Partition the training data into a training set (80%) and a validation set (20%)

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(123)
trainingset <- createDataPartition(pml_train$classe, p = 0.8, list = FALSE)
training <- pml_train[trainingset, ]
validation <- pml_train[-trainingset, ]
dim(training)
```

```
## [1] 15699    53
```

```r
dim(validation)
```

```
## [1] 3923   53
```


## Random Forest Model

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rf <- randomForest(classe ~ ., data = training, importance = TRUE, ntrees = 5)
```

### Check the accuracy of the training set when using the Random Forest model

```r
predict_training <- predict(rf, training)
print(confusionMatrix(predict_training, training$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

### Now, check the accuracy of the Random Forest model against the validation set

```r
predict_validation <- predict(rf, validation)
print(confusionMatrix(predict_validation, validation$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    3    0    0    0
##          B    1  754    3    0    0
##          C    0    2  680    3    0
##          D    0    0    1  640    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9964        
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9955        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9934   0.9942   0.9953   0.9986
## Specificity            0.9989   0.9987   0.9985   0.9994   1.0000
## Pos Pred Value         0.9973   0.9947   0.9927   0.9969   1.0000
## Neg Pred Value         0.9996   0.9984   0.9988   0.9991   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1922   0.1733   0.1631   0.1835
## Detection Prevalence   0.2850   0.1932   0.1746   0.1637   0.1835
## Balanced Accuracy      0.9990   0.9961   0.9963   0.9974   0.9993
```



### Let's use the model on the testing data set to determine the outcome of classe

```r
predict_test <- predict(rf, pml_test)
predict_test
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
