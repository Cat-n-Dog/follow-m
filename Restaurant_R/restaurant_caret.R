library('ggplot2')
library('lubridate')
library('dplyr')
library('tree')
library('randomForest')
library('gbm')
library('caret')

train_df <- read.csv(file = 'train.csv')
train_df$Open.Date <- lubridate::mdy(train_df$Open.Date)
train_df <- train_df %>%
  dplyr::mutate(Open.Year = year(Open.Date), Open.Month = month(Open.Date),
                City2 = ifelse(test = City.Group == 'Big Cities', yes = as.character(City), no = 0))
train_df$City2 <- as.factor(train_df$City2)
train_df$Open.Year <- as.factor(train_df$Open.Year)
train_df$Open.Month <- as.factor(train_df$Open.Month)

train_df_wo_3_outliers <- train_df %>% filter(revenue < 1e7)

# This set excludes three outlier records. It selects all features including Type and p's.
train_df_model_1 <- train_df_wo_3_outliers %>%
  select(-(Id:City.Group))

# This set excludes three outlier records. It selects all p's but excludes Type.
train_df_model_2 <- train_df_wo_3_outliers %>%
  select(-(Id:Type))

fitControl <- trainControl(method = "repeatedcv",number = 5,repeats = 5)
caret.fit <- train(revenue ~ ., data = train_df_model_1, subset = train,
                 method = "cubist", trControl = fitControl)

y.test.train_df <- train_df_wo_3_outliers[-train, 'revenue']
yhat_in_sample <- predict(caret.fit, newdata = train_df_model_1[-train,])
sqrt(mean((yhat_in_sample - y.test.train_df)^2))




caret.fit1 <- train(revenue ~ ., data = train_df_model_1,
                   method = "cubist", trControl = fitControl)
caret.fit2 <- train(revenue ~ ., data = train_df_model_2,
                    method = "cubist", trControl = fitControl)


test_df2 <- read.csv('test_fixed_year.csv')
test_df2$Open.Year <- as.factor(test_df2$Open.Year)
test_df2$Open.Month <- as.factor(test_df2$Open.Month)
test_df2 <- test_df2 %>%
  dplyr::mutate(
    City2 = ifelse(test = City.Group == 'Big Cities', yes = as.character(City), no = 0))
test_df2$City2 <- as.factor(test_df2$City2)

test_df_model_1 <- test_df2 %>% filter(Type!='MB')
test_df_model_1$Type <- droplevels(test_df_model_1$Type)

submission_model_1 <- predict(caret.fit1, newdata = select(test_df_model_1, -(Id:City.Group)))

names(submission_model_1) <- test_df_model_1$Id
submission_model_1_df <- data.frame(submission_model_1, row.names = names(submission_model_1))
colnames(submission_model_1_df) <- 'Prediction'

test_df_model_2 <- test_df2 %>% filter(Type=='MB')

submission_model_2 <- predict(caret.fit2, newdata = select(test_df_model_2, -(Id:Type)))

names(submission_model_2) <- test_df_model_2$Id
submission_model_2_df <- data.frame(submission_model_2, row.names = names(submission_model_2))
colnames(submission_model_2_df) <- 'Prediction'

submission_df <- rbind(submission_model_1_df, submission_model_2_df)

write.csv(submission_df, file='submission_cubist.csv', row.names=TRUE, quote=FALSE)
