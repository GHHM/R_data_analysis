library(caret) # total packages for data analysis
library(skimr) # Descriptive statistics
library(RANN) # Predicting missing values

#Load data from csv
data_raw = read.csv("https://www.dropbox.com/s/4wpkhme7476zdt3/dataset.csv?dl=1")

#Split training set and test set
trainRowNumbers = createDataPartition(data_raw$credit_amount, p = 0.8, list = FALSE)
trainData = data_raw[trainRowNumbers, ]
testData = data_raw[-trainRowNumbers, ]

x = trainData[, 1:21]
y = trainData$class

#Preprocessing_Imputation
anyNA(trainData) #FALSE
preProcess_missingdata_model = preProcess(trainData, method='knnImpute')
trainData_impute = predict(preProcess_missingdata_model,newdata=trainData)
anyNA(trainData_impute) # no more NA!

#one-hot encoding
dummies_model = dummyVars(class ~., data=trainData_impute)
trainData_mat = predict(dummies_model, newdata=trainData_impute)
trainData_dummy = data.frame(trainData_mat)

preProcess_range_model = preProcess(trainData_dummy, method='range')
trainData_pre = predict(preProcess_range_model,newdata = trainData_dummy)

apply(trainData_pre[,1:61],2,FUN = function(x){c('min'=min(x),'max'=max(x))})
trainData_pre$class = y 
rm(trainData_dummy, trainData_impute, trainData_mat, trainRowNumbers)

#make testData set
testData2 = predict(preProcess_missingdata_model, testData) #testData 의 빈 부분을 채움
testData3 = predict(dummies_model,testData2) #boolean값을 1,0 으로 바꿔줌
testData3 = data.frame(testData3)#매트릭스로는 R을 돌릴수 없다. 프레임화
testData4 = predict(preProcess_range_model, testData3)

# Hyperparameter tuning _ tree를 몇 개를 만들지 튜닝한다.
fitControl <-trainControl(
  method = 'cv',  #Cross validation (overfit을 막아주지만..)
  number = 5,  #number of folds
  savePredictions = 'final', #가장 성능 좋은 녀석을 저장해라
  classProbs = T,
  summaryFunction = twoClassSummary #output이 두 개이기 때문에 twoClass. 그 이상은 
)
#mars2
model_mars = train(class ~., data = trainData_pre, method = 'earth', metric = 'ROC',
                    tuneLength = 5, trControl = fitControl)
#confusionMatrix
predicted_mars = predict(model_mars, testData4)
confusionMatrix(reference = testData$class, data = predicted_mars, mode='everything')

# Ensemble
library(caretEnsemble)

trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'nnet')
models <- caretList(class ~ ., data=trainData_pre, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)
bwplot(results, scales = list(x=list(relation = "free"), y=list(relation="free")))

model_ensemble = caretEnsemble(models)
predicted_ensemble = predict(model_ensemble, testData4)
confusionMatrix(reference = testData$class, data = predicted_ensemble, mode = 'everything')


