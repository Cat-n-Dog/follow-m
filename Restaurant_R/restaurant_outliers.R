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
train_df_3_outliers <- train_df %>% filter(revenue >= 1e7)