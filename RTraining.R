library(caret)
library(earth)

data_raw = iris

trainRowNumbers = createDataPartition(data_raw$Species, p = 0.8, list = FALSE)
trainData = data_raw[trainRowNumbers, ]
testData = data_raw[-trainRowNumbers, ]

model = train(Species ~. , data = trainData, method = 'rf')
predictedData = predict(model, testData)

sum(predictedData == testData$Species) / 30 * 100

class_pack = c("caret",
               "skimr",
               "RANN",
               "randomForest",
               "fastAdaboost",
               "gbm",
               "xgboost",
               "caretEnsemble",
               "C50",
               "earth")
install.packages(class_pack)


library(caret) # total packages for data analysis
library(skimr) # Descriptive statistics
library(RANN) # Predicting missing values

orange = read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv')

#Step 3 : create the training and test data
trainRowNumbers = createDataPartition(orange$Purchase, p = 0.8, list = FALSE)
trainData = orange[trainRowNumbers, ]
testData = orange[-trainRowNumbers, ]

x = trainData[, 2:18]
y = trainData$Purchase

# show Descriptive statistics
skimed = skim_to_wide(trainData)
skimmed[,c(10:16)]

#Step 4 : Imputing missing values using preProcess
anyNA(trainData)
preProcess_missingdata_model = preProcess(trainData, method='knnImpute')
trainData_impute = predict(preProcess_missingdata_model,newdata=trainData)
anyNA(trainData_impute) # no more NA!

dummies_model = dummyVars(Purchase ~., data=trainData_impute)
trainData_mat = predict(dummies_model, newdata=trainData_impute)
trainData_dummy = data.frame(trainData_mat)
preProcess_range_model = preProcess(trainData_dummy, method='range')
trainData_pre = predict(preProcess_range_model,newdata = trainData_dummy)

#모든 값을 0과 1사이 값으로 맵핑
apply(trainData_pre[,1:10],2,FUN = function(x){c('min'=min(x),'max'=max(x))})
trainData_pre$Purchase = y 
rm(trainData_dummy, trainData_impute, trainData_mat, trainRowNumbers) 
rm(list= ls())

# Random forest를 만드는 가장 큰 이유는.. variable imporance 를 찍었을 때 해석이 가능하기 때문! 
model_rf = train(Purchase ~., data=trainData_pre,method='rf')
model_rf
plot(model_rf)
varimp_rf = varImp(model_rf)
plot(varimp_rf)

testData2 = predict(preProcess_missingdata_model, testData) #testData 의 빈 부분을 채움
testData3 = predict(dummies_model,testData2) #boolean값을 1,0 으로 바꿔줌
testData3 = data.frame(testData3)#매트릭스로는 R을 돌릴수 없다. 프레임화
testData4 = predict(preProcess_range_model, testData3)

# Hyperparameter tuning
# tree를 몇 개를 만들지 튜닝한다.
fitControl <-trainControl(
  method = 'cv',  #Cross validation (overfit을 막아주지만..)
  number = 5,  #number of folds
  savePredictions = 'final', #가장 성능 좋은 녀석을 저장해라
  classProbs = T,
  summaryFunction = twoClassSummary #output이 두 개이기 때문에 twoClass. 그 이상은 
)

model_mars2 = train(Purchase ~., data = trainData_pre, method = 'earth', metric = 'ROC',
                    tuneLength = 5, trControl = fitControl)
model_rf2 = train(Purchase ~., data=trainData_pre, method = 'rf', metric = 'ROC', tuneLength = 5,
                  trControl = fitControl)
model_nnet2 = train(Purchase ~., data=trainData_pre, method = 'nnet', metric = 'ROC', tuneLength = 5,
                   trControl = fitControl)


predicted_mars2 = predict(model_mars2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted_mars2, mode='everything', positive = 'MM')

predicted_rf2 = predict(model_rf2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted_rf2, mode='everything', positive = 'MM')

predicted_nnet2 = predict(model_nnet2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted_nnet2, mode='everything', positive = 'MM')
# F1 score = recall과 precision의 조합평균
# 주의: 반드시 testset에서의 성능평가여야함


#모델별로 나오는 값들을 평균냄
#Ensemble
library(caretEnsemble)

trainControl <- trainControl(method = "repeatdcv", number=10, repeats=3, savePrecictions=TRUE, classProbs=TRUE)
