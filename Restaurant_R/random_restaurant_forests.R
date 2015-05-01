library('ggplot2')
library('lubridate')
library('dplyr')
library('glmnet')
library('tree')
library('randomForest')
library('gbm')


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


# train_df_model_3 <- train_df_wo_3_outliers %>%
#   select(-(Id:Type, ))

y.test.train_df <- train_df_wo_3_outliers[-train, 'revenue']

rf_model_in_sample <- randomForest(revenue ~ ., data=train_df_model_1, importance = TRUE,
                                   mtry = 13, subset = train)
yhat_in_sample <- predict(rf_model_in_sample, newdata = train_df_model_1[-train,])
sqrt(mean((yhat_in_sample - y.test.train_df)^2))

boost_model_in_sample <- gbm(revenue ~ ., data=train_df_model_1[train,],
                             distribution = 'gaussian', n.trees =5000 ,
                             interaction.depth =4, shrinkage=0.001)
yhat_in_sample <- predict(boost_model_in_sample, newdata = train_df_model_1[-train,], n.trees=5000)
sqrt(mean((yhat_in_sample - y.test.train_df)^2))


rf_model_1 <- randomForest(revenue ~ ., data=train_df_model_1, importance = TRUE)
rf_model_2 <- randomForest(revenue ~ ., data=train_df_model_2, importance = TRUE)

boost_model_1 <- gbm(revenue ~ ., data=train_df_model_1,
                     distribution = 'gaussian', n.trees =5000,
                     interaction.depth =4, shrinkage=0.001)
boost_model_2 <- gbm(revenue ~ ., data=train_df_model_2,
                     distribution = 'gaussian', n.trees =5000,
                     interaction.depth =4, shrinkage=0.001)



test_df2 <- read.csv('test_fixed_year.csv')
test_df2$Open.Year <- as.factor(test_df2$Open.Year)
test_df2$Open.Month <- as.factor(test_df2$Open.Month)
test_df2 <- test_df2 %>%
  dplyr::mutate(
    City2 = ifelse(test = City.Group == 'Big Cities', yes = as.character(City), no = 0))
test_df2$City2 <- as.factor(test_df2$City2)

test_df_model_1 <- test_df2 %>% filter(Type!='MB')
test_df_model_1$Type <- droplevels(test_df_model_1$Type)

submission_model_1 <- predict(rf_model_1, newdata = select(test_df_model_1, -(Id:City.Group)))
submission_model_1 <- predict(boost_model_1, newdata = select(test_df_model_1, -(Id:City.Group)), n.trees=5000)

names(submission_model_1) <- test_df_model_1$Id
submission_model_1_df <- data.frame(submission_model_1, row.names = names(submission_model_1))
colnames(submission_model_1_df) <- 'Prediction'

test_df_model_2 <- test_df2 %>% filter(Type=='MB')

submission_model_2 <- predict(rf_model_2, newdata = select(test_df_model_2, -(Id:Type)))
submission_model_2 <- predict(boost_model_2, newdata = select(test_df_model_2, -(Id:Type)), n.trees=5000)

names(submission_model_2) <- test_df_model_2$Id
submission_model_2_df <- data.frame(submission_model_2, row.names = names(submission_model_2))
colnames(submission_model_2_df) <- 'Prediction'

submission_df <- rbind(submission_model_1_df, submission_model_2_df)
write.csv(submission_df, file='submission_rf_2.csv', row.names=TRUE, quote=FALSE)
write.csv(submission_df, file='submission_boost.csv', row.names=TRUE, quote=FALSE)
