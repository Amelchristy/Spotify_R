library(neuralnet)
library(forecast)
library(tidymodels)
library(xgboost)
library(caret)
library(fastDummies)
library(ggplot2)
library(ellipsis)
library(FNN)
library(fastDummies)

set.seed(12345)

normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}


############################################################
##KNN
############################################################

getwd()
setwd('~/BUAN448 PreditiveR')
library(caret)
library(ellipsis)
library(FNN)

songs <- read.csv('spotify_songs.csv')

#step1 - categorical -> dummy
d_genre <- dummyVars('~playlist_genre',songs)
songs_genre <- predict(d_genre,songs)

d_subgenre <- dummyVars('~playlist_subgenre',songs)
songs_subgenre <- predict(d_subgenre,songs)

songs <- cbind(songs,songs_genre,songs_subgenre)
songs$is_popular <- ifelse(songs$track_popularity>70,1,0)

songs <- songs[,-c(1,2,3,4,5,6,7,8,9,10,11,14)]

#step2 - scale the features
norm.values <- preProcess(songs[ ,c(3,10,11)], method='range')
songs[,c(3,10,11)] <- predict(norm.values,songs[,c(3,10,11)])

#split the data
set.seed(12345)
train.index <- sample(c(1:dim(songs)[1]), dim(songs)[1]*0.6)
train.df <- songs[train.index, ]
valid.df <- songs[-train.index, ]

#build the model
accuracy_df <- data.frame(k=1:15, Accuracy = numeric(15))

for(k in 1:16){
  songs.pred <- knn(train=train.df,
                   test = valid.df,
                   cl=train.df$is_popular,
                   k=k)
  accuracy <- mean(songs.pred == valid.df$is_popular)
  accuracy_df$Accuracy[k - 1] <- accuracy
}

print(accuracy_df)
#high accuracy - consider the model is overfitting

#using cross-validation
songs$is_popular <- as.factor(songs$is_popular)
train_control <- trainControl(method = "cv", number = 10)
knn_fit <- train(is_popular ~ ., data = songs, method = "knn", trControl = train_control)
print(knn_fit)

#the final value used for the model was k = 9

#########################################
##LOGISTIC REGRESSION
#########################################

# Read and preprocess the dataset
songs.glm <- read.csv("spotify_songs.csv", header = TRUE, sep = ",")
head(songs.glm)

# Keep the desired features
features_to_keep <- c("track_popularity", "playlist_genre", "playlist_subgenre", 
                      "danceability", "energy", "key", "loudness", "mode", 
                      "speechiness", "acousticness", "instrumentalness", 
                      "liveness", "valence", "tempo", "duration_ms")
songs.glm <- songs.glm[, features_to_keep]

# Normalize certain columns
columns_non_normal <- c("loudness", "tempo", "duration_ms")
songs.glm[columns_non_normal] <- as.data.frame(lapply(songs.glm[columns_non_normal], normalize))

# Create dummy variables and remove the original categorical columns
songs.glm <- dummy_cols(songs.glm, select_columns = c("playlist_genre", "playlist_subgenre"), remove_selected_columns = TRUE)

# Create binary target variable and convert it to a factor
# Modify the 'popular' column using ifelse
songs.glm$popular <- ifelse(songs.glm$track_popularity > 70, 1, 0)

# Remove the 'track_popularity' column
songs.glm$track_popularity <- NULL
head(songs.glm)

# Split into training (60%) and testing set (40%)
set.seed(12345)
train_test_split <- initial_split(songs.glm, prop = 0.6)
train.glm <- training(train_test_split)
test.glm <- testing(train_test_split)

summary(train)

# Option 1: Remove rows with NA
train_no_na <- na.omit(train.glm)
# Fit the initial GLM model with the cleaned data
model1 <- glm(popular ~ danceability + energy + key + loudness + mode +
                speechiness + acousticness + instrumentalness +
                liveness + valence + tempo + duration_ms,
              data = train_no_na, family = "binomial")
summary(model1)

#predictions on the valid set
test.glm$predictions <- ifelse(predict(model1, newdata = test.glm, type = "response") > 0.3, 1, 0)

# Accuracy of the model
accuracy<-mean(test.glm$predictions == test.glm$popular)
accuracy
# 0.8587635
# Confusion matrix
cmatrix_model <- table(test.glm$predictions,test.glm$popular)
cmatrix_model


#MODEL2 use all features
model2 <- glm(popular ~ ., data = train_no_na, family = "binomial")

summary(model2)

#predictions on the valid set
test.glm$predictions <- ifelse(predict(model2, newdata = test.glm, type = "response") > 0.3, 1, 0)

# Accuracy of the model
accuracy2<-mean(test.glm$predictions == test.glm$popular)
accuracy2
#0.8442211
# Confusion matrix
cmatrix_model2 <- table(valid.data$predictions, valid.data$popular)
cmatrix_model2
#correlation Heatmap 
install.packages("corrplot")
library(corrplot)
numerical_features <- spotify_songs[, c("danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms")]
correlations <- cor(numerical_features)
corrplot(correlations, method = "color", color='green')


#popularity by genre
ggplot(spotify_songs, aes(x = playlist_genre, fill = as.factor(popular))) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("0" = "black", "1" = "green")) +
  labs(title = "Popularity Distribution by Genre", x = "Genre", y = "Count", fill = "Popularity")


#coefficient model 1 popular ~ danceability + energy + key + loudness + mode +speechiness + acousticness + instrumentalness +
#liveness + valence + tempo + duration_ms,
library(ggplot2)
model1_coef <- coef(model1)
model1_df <- data.frame(feature = names(model1_coef), coefficient = model1_coef)
ggplot(model1_df, aes(x = reorder(feature, coefficient), y = coefficient)) +
  geom_col(fill="green3") +
  coord_flip() +
  labs(title = "Feature Importance in Model 1", x = "Feature", y = "Coefficient")



#model 2 
model2_coef <- coef(model2)
model2_df <- data.frame(feature = names(model2_coef), coefficient = model2_coef)
ggplot(model2_df, aes(x = reorder(feature, coefficient), y = coefficient)) +
  geom_col(fill="green4") +
  coord_flip() +
  labs(title = "Feature Importance in Model 2", x = "Feature", y = "Coefficient")



######################################
##XG BOOST
#########################################

songs.xg <- read.csv("spotify_songs.csv", header = TRUE, sep = ",")
head(songs.xg)

# Exploratory Data Analysis (optional steps not included here)

# Keep the desired features
features_to_keep <- c("track_popularity", "playlist_genre", "playlist_subgenre", 
                      "danceability", "energy", "key", "loudness", "mode", 
                      "speechiness", "acousticness", "instrumentalness", 
                      "liveness", "valence", "tempo", "duration_ms")
songs.xg <- songs.xg[, features_to_keep]

# Normalize certain columns
columns_non_normal <- c("loudness", "tempo", "duration_ms")
songs.xg[columns_non_normal] <- as.data.frame(lapply(spotify_songs[columns_non_normal], normalize))

# Create dummy variables and remove the original categorical columns
songs.xg <- dummy_cols(songs.xg, select_columns = c("playlist_genre", "playlist_subgenre"), remove_selected_columns = TRUE)

# Create binary target variable and convert it to a factor
songs.xg$popular <- as.factor(songs.xg$track_popularity > 70)
songs.xg$track_popularity <- NULL

# Split into training (75%) and testing set (25%)
set.seed(12345)
train_test_split.xg <- initial_split(songs.xg, prop = 0.75)
train.xg <- training(train_test_split.xg)
test.xg <- testing(train_test_split.xg)

# Prepare data for xgboost
train_x <- as.matrix(train.xg[, -which(names(train.xg) == "popular")])
train_y <- as.numeric(train.xg$popular) - 1

test_x <- as.matrix(test.xg[, -which(names(test.xg) == "popular")])
test_y <- as.numeric(test.xg$popular) - 1

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

# Define parameters and perform cross-validation
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# 10-fold cross-validation
cv_model <- xgb.cv(params = params, data = xgb_train, nrounds = 1000, nfold = 8, verbose = 0, early_stopping_rounds = 10)

# Train the model using optimal number of rounds
optimal_nrounds <- cv_model$best_iteration
xgb_model <- xgboost(params = params, data = xgb_train, nrounds = optimal_nrounds, verbose = 0)

# Predict and evaluate
test_pred_probs <- predict(xgb_model, xgb_test)
test_pred <- ifelse(test_pred_probs > 0.5, 1, 0)

# Confusion matrix and additional evaluation metrics
conf_mat.xg <- confusionMatrix(as.factor(test_pred), as.factor(test_y))

# Print evaluation results
print(conf_mat.xg)

# Feature importance
importance_plot <- xgb.plot.importance(importance_matrix)

# Modify the plot to change bar color to green
library(ggplot2)
library(xgboost)

# Assuming 'importance_matrix' is already created from your xgb model
# importance_matrix <- xgb.importance(feature_names = colnames(train_x), model = xgb_model)

# Convert importance_matrix to a data frame if it's not already
importance_matrix_df <- as.data.frame(importance_matrix)

# Create the feature importance plot manually using ggplot2
importance_plot <- ggplot(importance_matrix_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "green4") +
  coord_flip() +  # Flip the axes to make it horizontal
  theme_minimal() +
  xlab("Features") +
  ylab("Gain")

# Display the plot
print(importance_plot)

##########################################
LINEAR REGRESSION
##########################################

spotify_songs2 <- read.csv("spotify_songs.csv")

# Extracting necessary columns
features <- c("track_popularity", "playlist_genre", "playlist_subgenre", 
              "danceability", "energy", "key", "loudness", "mode", 
              "speechiness", "acousticness", "instrumentalness", 
              "liveness", "valence", "tempo", "duration_ms")
spotify_songs2 <- spotify_songs2[, features]

# normalization function
normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

norm_cols <- c("loudness", "tempo", "duration_ms", "track_popularity")
spotify_songs2[norm_cols] <- lapply(spotify_songs2[norm_cols], normalize)


set.seed(12345)

# Splitting the data into training and validation sets
train_index <- sample(1:nrow(spotify_songs2), 0.6 * nrow(spotify_songs2))
train_data <- spotify_songs2[train_index, ]
valid_data <- spotify_songs2[-train_index, ]

# Building linear regression models 
null_model <- lm(track_popularity ~ 1, data = train_data)
full_model <- lm(track_popularity ~ ., data = train_data)

# Backward 
backward_model <- step(full_model, direction = "backward", scope = list(lower = null_model, upper = full_model))
summary(backward_model)

# Forward 
forward_model <- step(null_model, direction = "forward", scope = list(lower = null_model, upper = full_model))
summary(forward_model)

# Both direction 
both_model <- step(null_model, direction = "both", scope = list(lower = null_model, upper = full_model))
summary(both_model)

# Making predictions on the validation set
backward_pred <- predict(backward_model, valid_data)
forward_pred <- predict(forward_model, valid_data)
both_pred <- predict(both_model, valid_data)

# Calculating RMSE 
rmse_backward <- sqrt(mean((valid_data$track_popularity - backward_pred)^2))
rmse_forward <- sqrt(mean((valid_data$track_popularity - forward_pred)^2))
rmse_both <- sqrt(mean((valid_data$track_popularity - both_pred)^2))
rmse_backward
#0.2296593
rmse_both
#0.2296593
rmse_forward
#0.2296593

# Calculating MAE
mae_backward <- mean(abs(valid_data$track_popularity - backward_pred))
mae_forward <- mean(abs(valid_data$track_popularity - forward_pred))
mae_both <- mean(abs(valid_data$track_popularity - both_pred))
#Results 
mae_backward
#0.1904584
mae_forward
#0.1904584
mae_both
#0.1904584

# Calculating ME
me_backward <- mean(valid_data$track_popularity - backward_pred)
me_forward <- mean(valid_data$track_popularity - forward_pred)
me_both <- mean(valid_data$track_popularity - both_pred)
me_backward
#0.003762642
me_forward
#0.003762642
me_both
#0.003762642

##########################################
#NEURAL NETWORK
##########################################
  songs.nn <- read.csv("spotify_songs.csv")
  
  columns_non_normal.nn <- c("loudness", "tempo", "duration_ms", "track_popularity")
  songs.nn[columns_non_normal.nn] <- as.data.frame(lapply(songs.nn[columns_non_normal.nn], 
                                                          normalize))
  
  #Creating sub data frames by playlist genre category
  songs.df.pop <- subset(songs.nn, songs.nn$playlist_genre=="pop")
  songs.df.rap <- subset(songs.nn, songs.nn$playlist_genre=="rap")
  songs.df.rock <- subset(songs.nn, songs.nn$playlist_genre=="rock")
  songs.df.latin <- subset(songs.nn, songs.nn$playlist_genre=="latin")
  songs.df.rnb <- subset(songs.nn, songs.nn$playlist_genre=="r&b")
  songs.df.edm <- subset(songs.nn, songs.nn$playlist_genre=="edm")
  
  accuracy.nn <- c()
  accuracy.nn.mae <- c()

"
Neural networks were created by each playlist genre, then all the songs as a whole.
Processing time was too long to 

Two hidden levels, one with two nodes and the second with one node. This was 
determined to be the best balance of accuracy and performance time
"
  
  ##POP SONGS
  
  
    train_size_pop <- round(nrow(songs.df.pop)*.60)
    train_index_pop <- sample(nrow(songs.df.pop),train_size_pop)
    
    train.df_pop <- songs.df.pop[train_index_pop, ]
    valid.df_pop <- songs.df.pop[-train_index_pop, ]
    
    popularity_model_pop <- neuralnet(track_popularity~
                                        danceability+energy+loudness+
                                        speechiness+acousticness+instrumentalness+ liveness+valence+
                                        tempo+duration_ms, 
                                      data=train.df_pop,
                                      hidden=c(2,1))
    plot(popularity_model_pop)
    
    valid.prediction_pop <- compute(popularity_model_pop, valid.df_pop)
    valid.df_pop$prediction <- valid.prediction_pop$net.result
    round(cor(valid.df_pop$track_popularity,valid.df_pop$prediction),digits=2)*100
    
    accuracy.nn.mae[1] <-  accuracy(valid.df_pop$track_popularity,valid.df_pop$prediction)[3]
    accuracy.nn[1] <-  accuracy(valid.df_pop$track_popularity,valid.df_pop$prediction)[5]
  
  
  
  
  ##RAP MODEL
  
    train_size_rap <- round(nrow(songs.df.rap)*.60)
    train_index_rap <- sample(nrow(songs.df.rap),train_size_rap)
    
    train.df_rap <- songs.df.rap[train_index_rap, ]
    valid.df_rap <- songs.df.rap[-train_index_rap, ]
    
    popularity_model_rap <- neuralnet(track_popularity~
                                        danceability+energy+loudness+
                                        speechiness+acousticness+instrumentalness+ liveness+valence+
                                        tempo+duration_ms, 
                                      data=train.df_rap,
                                      hidden=c(2,1))
    
    valid.prediction_rap <- compute(popularity_model_rap, valid.df_rap)
    valid.df_rap$prediction <- valid.prediction_rap$net.result
    round(cor(valid.df_rap$track_popularity,valid.df_rap$prediction),digits=2)*100
  
  accuracy(valid.df_rap$track_popularity,valid.df_rap$prediction)
  
  accuracy.nn.mae[2] <-   accuracy(valid.df_rap$track_popularity,valid.df_rap$prediction)[3]
  accuracy.nn[2] <-   accuracy(valid.df_rap$track_popularity,valid.df_rap$prediction)[5]

  
  
  ##EDM MODEL
  
    train_size_edm <- round(nrow(songs.df.edm)*.60)
    train_index_edm <- sample(nrow(songs.df.edm),train_size_edm)
    
    train.df_edm <- songs.df.edm[train_index_edm, ]
    valid.df_edm <- songs.df.edm[-train_index_edm, ]
    
    popularity_model_edm <- neuralnet(track_popularity~
                                        danceability+energy+loudness+
                                        speechiness+acousticness+instrumentalness+ liveness+valence+
                                        tempo+duration_ms, 
                                      data=train.df_edm,
                                      hidden=c(2,1))
    
    valid.prediction_edm <- compute(popularity_model_edm, valid.df_edm)
    valid.df_edm$prediction <- valid.prediction_edm$net.result
    round(cor(valid.df_edm$track_popularity,valid.df_edm$prediction),digits=2)*100
    
    accuracy.nn.mae[3] <- accuracy(valid.df_edm$track_popularity,valid.df_edm$prediction)[3]
    accuracy.nn[3] <- accuracy(valid.df_edm$track_popularity,valid.df_edm$prediction)[5]
    
    
  ##R&B
  
    train_size_rnb <- round(nrow(songs.df.rnb)*.60)
    train_index_rnb <- sample(nrow(songs.df.rnb),train_size_rnb)
    
    train.df_rnb <- songs.df.rnb[train_index_rnb, ]
    valid.df_rnb <- songs.df.rnb[-train_index_rnb, ]
    
    popularity_model_rnb <- neuralnet(track_popularity~
                                        danceability+energy+loudness+
                                        speechiness+acousticness+instrumentalness+ 
                                        liveness+valence+tempo+duration_ms, 
                                      data=train.df_rnb,
                                      hidden=c(2,1))
    
    valid.prediction_rnb <- compute(popularity_model_rnb, valid.df_rnb)
    valid.df_rnb$prediction <- valid.prediction_rnb$net.result
    round(cor(valid.df_rnb$track_popularity,valid.df_rnb$prediction),digits=2)*100
    
    accuracy.nn.mae[4] <- accuracy(valid.df_rnb$track_popularity,valid.df_rnb$prediction)[3]
    accuracy.nn[4] <- accuracy(valid.df_rnb$track_popularity,valid.df_rnb$prediction)[5]
    
  
  
  
  
  #ROCK
  
  
  train_size_rock <- round(nrow(songs.df.rock)*.60)
  train_index_rock <- sample(nrow(songs.df.rock),train_size_rock)
  
  train.df_rock <- songs.df.rock[train_index_rock, ]
  valid.df_rock <- songs.df.rock[-train_index_rock, ]
  
  popularity_model_rock <- neuralnet(track_popularity~
                                       danceability+energy+loudness+
                                       speechiness+acousticness+instrumentalness+ 
                                       liveness+valence+tempo+duration_ms, 
                                     data=train.df_rock,
                                     hidden=c(2,1))
  
  valid.prediction_rock <- compute(popularity_model_rock, valid.df_rock)
  valid.df_rock$prediction <- valid.prediction_rock$net.result
  round(cor(valid.df_rock$track_popularity,valid.df_rock$prediction),digits=2)*100
  
  accuracy.nn.mae[5] <- accuracy(valid.df_rock$track_popularity,valid.df_rock$prediction)[3]
  accuracy.nn[5] <- accuracy(valid.df_rock$track_popularity,valid.df_rock$prediction)[5]
  
  
  ##LATIN
  train_size_latin <- round(nrow(songs.df.latin)*.60)
  train_index_latin <- sample(nrow(songs.df.latin),train_size_latin)
  
  train.df_latin <- songs.df.latin[train_index_latin, ]
  valid.df_latin <- songs.df.latin[-train_index_latin, ]
  
  popularity_model_latin <- neuralnet(track_popularity~
                                        danceability+energy+loudness+
                                        speechiness+acousticness+instrumentalness+ 
                                        liveness+valence+tempo+duration_ms, 
                                      data=train.df_latin,
                                      hidden=c(2,1))
  
  valid.prediction_latin <- compute(popularity_model_latin, valid.df_latin)
  valid.df_latin$prediction <- valid.prediction_latin$net.result
  round(cor(valid.df_latin$track_popularity,valid.df_latin$prediction),digits=2)*100
  
  accuracy.nn.mae[6] <- accuracy(valid.df_latin$track_popularity,valid.df_latin$prediction)[3]
  accuracy.nn[6] <- accuracy(valid.df_latin$track_popularity,valid.df_latin$prediction)[5]
  
  
  
  
  ##ALL SONGS


  train_size_nn <- round(32833*.60)
  train_index_nn <- sample(32833,train_size_nn)
  
  train.df_nn <- songs.nn[train_index_nn, ]
  valid.df_nn <- songs.nn[-train_index_nn, ]
  
  "
  We must eliminate some variables in order to decrease processing times
  
  Instrumentalness captured in speechiness
  Liveness deemed inconsequential
  Loudness captured in energy
  
  "
  
  popularity_model.nn <- neuralnet(track_popularity~
                                     danceability+energy+
                                     speechiness+acousticness+ 
                                     valence+tempo+duration_ms,  
                                   data=train.df_nn,
                                   hidden=c(2,1))
  
  plot(popularity_model.nn)
  
  
  
  valid.prediction <- compute(popularity_model.nn, valid.df_nn)
  valid.df_nn$prediction <- valid.prediction$net.result
  
  accuracy.nn.mae[7] <- accuracy(valid.df_nn$track_popularity,valid.df_nn$prediction)[3]
  accuracy.nn[7] <- accuracy(valid.df_nn$track_popularity,valid.df_nn$prediction)[5]
  
  accuracy.nn
  
  final.accuracy.nn <- data.frame(Playlist_Genre=c("Pop","Rap","EDM","R&B","Rock","Latin","All"),
                        MAE = round(accuracy.nn.mae,digits=2),
                        MAPE = round(accuracy.nn,digits=2)
    
    
  )

##NN Sensitivity Analysis
"
Sensitivity analysis conducted to determine variable importance
Base case is all inputs set to .5
Each variable is increased and decreased by the sensitivity factor to determine
impact on track variability

"
  sensitivity_factor = .25
  
  new_set.nn <- data.frame(danceability=.5, energy=.5,
                           speechiness=.5, acousticness=.5, 
                           valence=.5,tempo=.5,duration_ms=.5)
  
  new_set.nn.danceability <- data.frame(danceability=.5+sensitivity_factor, energy=.5,
                                        speechiness=.5, acousticness=.5, 
                                        valence=.5,tempo=.5,duration_ms=.5)
  
  new_set.nn.energy <- data.frame(danceability=.5, energy=.5+sensitivity_factor,
                                  speechiness=.5, acousticness=.5, 
                                  valence=.5,tempo=.5,duration_ms=.5)
  
  
  new_set.nn.speechiness <- data.frame(danceability=.5, energy=.5,
                                       speechiness=.5+sensitivity_factor, acousticness=.5, 
                                       valence=.5,tempo=.5,duration_ms=.5)
  
  new_set.nn.acousticness <- data.frame(danceability=.5, energy=.5,
                                        speechiness=.5, acousticness=.5+sensitivity_factor, 
                                        valence=.5,tempo=.5,duration_ms=.5)
  
  new_set.nn.valence <- data.frame(danceability=.5, energy=.5,
                                   speechiness=.5, acousticness=.5, 
                                   valence=.5+sensitivity_factor,tempo=.5,duration_ms=.5)
  
  
  new_set.nn.tempo <- data.frame(danceability=.5, energy=.5,
                                 speechiness=.5, acousticness=.5, 
                                 valence=.5,tempo=.5+sensitivity_factor,duration_ms=.5)
  
  new_set.nn.duration <- data.frame(danceability=.5, energy=.5,
                                    speechiness=.5, acousticness=.5, 
                                    valence=.5,tempo=.5,duration_ms=.5+sensitivity_factor)
  
  new_set.nn.danceability.neg <- data.frame(danceability=.5-sensitivity_factor, energy=.5,
                                            speechiness=.5, acousticness=.5, 
                                            valence=.5,tempo=.5,duration_ms=.5)
  
  new_set.nn.energy.neg <- data.frame(danceability=.5, energy=.5-sensitivity_factor,
                                      speechiness=.5, acousticness=.5, 
                                      valence=.5,tempo=.5,duration_ms=.5)
  
  
  new_set.nn.speechiness.neg <- data.frame(danceability=.5, energy=.5,
                                           speechiness=.5-sensitivity_factor, acousticness=.5, 
                                           valence=.5,tempo=.5,duration_ms=.5)
  
  new_set.nn.acousticness.neg <- data.frame(danceability=.5, energy=.5,
                                            speechiness=.5, acousticness=.5-sensitivity_factor, 
                                            valence=.5,tempo=.5,duration_ms=.5)
  
  new_set.nn.valence.neg <- data.frame(danceability=.5, energy=.5,
                                       speechiness=.5, acousticness=.5, 
                                       valence=.5-sensitivity_factor,tempo=.5,duration_ms=.5)
  
  
  new_set.nn.tempo.neg <- data.frame(danceability=.5, energy=.5,
                                     speechiness=.5, acousticness=.5, 
                                     valence=.5,tempo=.5-sensitivity_factor,duration_ms=.5)
  
  new_set.nn.duration.neg <- data.frame(danceability=.5, energy=.5,
                                        speechiness=.5, acousticness=.5, 
                                        valence=.5,tempo=.5,duration_ms=.5-sensitivity_factor)
  
  prediction_nn.base <- compute(popularity_model.nn, new_set.nn)
  prediction_nn.danceability <- compute(popularity_model.nn, new_set.nn.danceability)
  prediction_nn.energy <- compute(popularity_model.nn, new_set.nn.energy)
  prediction_nn.speechiness <- compute(popularity_model.nn, new_set.nn.speechiness)
  prediction_nn.acousticness <- compute(popularity_model.nn, new_set.nn.acousticness)
  prediction_nn.valence <- compute(popularity_model.nn, new_set.nn.valence)
  prediction_nn.tempo <- compute(popularity_model.nn, new_set.nn.tempo)
  prediction_nn.duration <- compute(popularity_model.nn, new_set.nn.duration)
  
  prediction_nn.danceability.neg <- compute(popularity_model.nn, new_set.nn.danceability.neg)
  prediction_nn.energy.neg <- compute(popularity_model.nn, new_set.nn.energy.neg)
  prediction_nn.speechiness.neg <- compute(popularity_model.nn, new_set.nn.speechiness.neg)
  prediction_nn.acousticness.neg <- compute(popularity_model.nn, new_set.nn.acousticness.neg)
  prediction_nn.valence.neg <- compute(popularity_model.nn, new_set.nn.valence.neg)
  prediction_nn.tempo.neg <- compute(popularity_model.nn, new_set.nn.tempo.neg)
  prediction_nn.duration.neg <- compute(popularity_model.nn, new_set.nn.duration.neg)
  
  sensitivity.nn <- data.frame(Input=c("Base", "Danceability","Energy","Speechiness", 
                                       "Acousticness", "Valence", "Tempo", "Duration"),
                               Populairty_Pos = c(prediction_nn.base$net.result,prediction_nn.danceability$net.result,
                                                  prediction_nn.energy$net.result, prediction_nn.speechiness$net.result,
                                                  prediction_nn.acousticness$net.result, prediction_nn.valence$net.result,
                                                  prediction_nn.tempo$net.result,prediction_nn.duration$net.result),
                               Populairty_Neg=c(prediction_nn.base$net.result,prediction_nn.danceability.neg$net.result,
                                                prediction_nn.energy.neg$net.result, prediction_nn.speechiness.neg$net.result,
                                                prediction_nn.acousticness.neg$net.result, prediction_nn.valence.neg$net.result,
                                                prediction_nn.tempo.neg$net.result,prediction_nn.duration.neg$net.result))
  Base_Popularity.nn <- sensitivity.nn$Populairty_Pos[1]
  
  sensitivity.nn.change <- data.frame(Input=c("Base", "Danceability","Energy","Speechiness", 
                                              "Acousticness", "Valence", "Tempo", "Duration"))
  sensitivity.nn.change$Impact_Neg <- round(sensitivity.nn$Populairty_Neg-Base_Popularity.nn,digits=3)
  sensitivity.nn.change$Impact_Pos <- round(sensitivity.nn$Populairty_Pos-Base_Popularity.nn,digits=3)
  View(sensitivity.nn.change)
