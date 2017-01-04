setwd("G:/Data Science/Test")
library(caret)
train.1 <- read.csv("G:/Data Science/Test/sample_data.csv")
train.2 <- read.csv("G:/Data Science/Test/sample_output.csv")

train.raw <- merge(train.1, train.2, by = "Application_ID")

test.raw <- read.csv("G:/Data Science/Test/test_data.csv")
test.raw$Loan_Status <- 'Y'
View(test.raw)
total.raw <- rbind(train.raw, test.raw)

total.raw$Total_Income <- total.raw$ApplicantIncome + total.raw$CoapplicantIncome

total.raw$Married[is.na(total.raw$Married)] <- "Yes"

impute.mean <- function(x) {
  z <- mean(x, na.rm = TRUE)
  if(is.numeric(x) & any(is.na(x))){
    x[is.na(x)] <- z
  } else {
    x
  }
  return(x)
}

impute.med <- function(x) {
  z <- median(x, na.rm = TRUE)
  if(is.numeric(x) & any(is.na(x))){
    x[is.na(x)] <- z
  } else {
    x
  }
  return(x)
}

total.raw$LoanAmount <- impute.mean(total.raw$LoanAmount)

total.raw$Loan_Amount_Term <- impute.med(total.raw$Loan_Amount_Term)

total.raw$Credit_History <- impute.med(total.raw$Credit_History)
train.final <- total.raw[1:100,]
test.final <- total.raw[101:614, -13]

train.keeps <- c("Loan_Status", "Gender", "Married", "Dependents", "Total_Income", 
                 "LoanAmount", "Loan_Amount_Term", "Credit_History")

set.seed(23)
training.rows <- createDataPartition(train.final$Loan_Status, 
                                     p = 0.8, list = FALSE)
train.batch <- train.final[training.rows, ]
test.batch <- train.final[-training.rows, ]

## Define control function to handle optional arguments for train function
## Models to be assessed based on largest absolute area under ROC curve
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

View(train.batch)
set.seed(35)

glm.tune <- caret::train(Loan_Status ~ Gender + Married + Dependents + Total_Income + LoanAmount + Loan_Amount_Term + Credit_History, data = train.batch, method = "glm", metric = "ROC", trControl = cv.ctrl)
				   
## note the dot preceding each variable
ada.grid <- expand.grid(.iter = c(50, 100),
                        .maxdepth = c(4, 8),
                        .nu = c(0.1, 1))

# set.seed(35)
# ada.tune <- caret::train(Loan_Status ~ Gender + Married + Dependents + Total_Income + LoanAmount + Loan_Amount_Term + Credit_History, 
#                   data = train.batch,
#                   method = "ada",
#                   metric = "ROC",
#                   tuneGrid = ada.grid,
#                   trControl = cv.ctrl)

rf.grid <- data.frame(.mtry = c(2, 3))
set.seed(35)
rf.tune <- caret::train(Loan_Status ~ Gender + Married + Dependents + Total_Income + LoanAmount + Loan_Amount_Term + Credit_History, 
                 data = train.batch,
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = rf.grid,
                 trControl = cv.ctrl)

set.seed(35)
svm.tune <- caret::train(Loan_Status ~ Gender + Married + Dependents + Total_Income + LoanAmount + Loan_Amount_Term + Credit_History, 
                  data = train.batch,
                  method = "svmRadial",
                  tuneLength = 9,
                  preProcess = c("center", "scale"),
                  metric = "ROC",
                  trControl = cv.ctrl)


glm.pred <- predict(glm.tune, test.batch)
confusionMatrix(glm.pred, test.batch$Loan_Status)

rf.pred <- predict(rf.tune, test.batch)
confusionMatrix(rf.pred, test.batch$Loan_Status)

svm.pred <- predict(svm.tune, test.batch)
confusionMatrix(svm.pred, test.batch$Loan_Status)

#Cast Your Votes				 
				  
# data prepped for casting predictions
test.keeps <- train.keeps[-1]
pred.these <- test.final[test.keeps]	  
# use the random forect model to generate predictions
Loan_Status <- predict(rf.tune, newdata = pred.these)

predictions <- as.data.frame(Loan_Status)

predictions$Application_ID <- test.final$Application_ID
View(test.final)
write.csv(predictions[,c("Application_ID", "Loan_Status")], 
          file="test_output.csv", row.names=FALSE, quote=FALSE)

