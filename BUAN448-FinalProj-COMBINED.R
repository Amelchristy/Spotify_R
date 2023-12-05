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

songs.knn <- read.csv('spotify_songs.csv')

head(songs.knn)
View(songs.knn)

#step1 - categorical -> dummy
d_genre <- dummyVars('~playlist_genre',songs.knn)
songs_genre <- predict(d_genre,songs.knn)

d_subgenre <- dummyVars('~playlist_subgenre',songs.knn)
songs_subgenre <- predict(d_subgenre,songs.knn)

songs.knn <- cbind(songs.knn,songs_genre,songs_subgenre)
songs.knn$is_popular <- ifelse(songs.knn$track_popularity>70,1,0)

songs.knn <- songs.knn[,-c(1,2,3,4,5,6,7,8,9,10,11,14)]

#step2 - scale the features
norm.values <- preProcess(songs.knn[ ,c(3,10,11)], method=c("center", "scale"))
songs.knn[,c(3,10,11)] <- predict(norm.values,songs.knn[,c(3,10,11)])

#split the data
set.seed(12345)
train.index.knn <- sample(c(1:dim(songs.knn)[1]), dim(songs.knn)[1]*0.6)
train.df.knn <- songs.knn[train.index.knn, ]
valid.df.knn <- songs.knn[-train.index.knn, ]

#build the model
accuracy_df.knn <- data.frame(k=1:15, Accuracy = numeric(15))

for(k in 1:16){
  songs.pred.knn <- knn(train=train.df.knn,
                        test = valid.df.knn,
                        cl=train.df.knn$is_popular,
                        k=k)
  accuracy <- mean(songs.pred.knn == valid.df.knn$is_popular)
  accuracy_df.knn$Accuracy[k - 1] <- accuracy
}

print(accuracy_df.knn)


#what's the best k?  K=2
best <- which.max(accuracy_df.knn$Accuracy)
print(best)

#analysis
prediction.knn <- knn(train=train.df.knn,
                      test = valid.df.knn,
                      cl=train.df.knn$is_popular,
                      k=2)
conf_matrix.knn <- table(prediction.knn,valid.df.knn$is_popular)

confusionMatrix(conf_matrix.knn)



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
songs.glm$popular <- ifelse(songs.glm$track_popularity > 80, 1, 0)

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
spotify_glm_no_na <- glm(popular ~ ., data = train_no_na, family = binomial())

# Forward stepwise regression
forward <- step(glm(popular ~ 1, data = train_no_na, family = "binomial"),
                scope = list(upper = spotify_glm_no_na),
                direction = "forward", trace = TRUE)

# Check the summary
summary(forward)

#backward

backward <- step(glm(popular ~ 1, data = train_no_na, family = "binomial"),
                 direction = "backward", trace = TRUE)

summary(backward)


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
    
    accuracy(valid.df_pop$track_popularity,valid.df_pop$prediction)
  
  
  
  
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
    
    accuracy(valid.df_edm$track_popularity,valid.df_edm$prediction)
  
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
    
    accuracy(valid.df_rnb$track_popularity,valid.df_rnb$prediction)
  
  
  
  
  
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
  
  accuracy(valid.df_rock$track_popularity,valid.df_rock$prediction)
  
  
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
                                      hidden=c(2))
  
  valid.prediction_latin <- compute(popularity_model_latin, valid.df_latin)
  valid.df_latin$prediction <- valid.prediction_latin$net.result
  round(cor(valid.df_latin$track_popularity,valid.df_latin$prediction),digits=2)*100
  
  accuracy(valid.df_latin$track_popularity,valid.df_latin$prediction)
  
  
  ##ALL SONGS


  train_size_nn <- round(32833*.60)
  train_index_nn <- sample(32833,train_size_nn)
  
  train.df_nn <- songs.nn[train_index_nn, ]
  valid.df_nn <- songs.nn[-train_index_nn, ]
  
  "
  We must eliminate some variables in order to decrease processing times
  
  Instrumentalness captured in speechiness
  Liveness deemed unconsequential
  Loudness captured in energy
  
  "
  
  popularity_model.nn <- neuralnet(track_popularity~
                                     danceability+energy+
                                     speechiness+acousticness+ 
                                     valence+tempo+duration_ms,  
                                   data=train.df_nn,
                                   hidden=c(2,1))
  
  plot(popularity_model.nn)
  
  
  
  valid.prediction <- compute(popularity_model, valid.df)
  valid.df_nn$prediction <- valid.prediction$net.result
  
  accuracy(valid.df_nn$track_popularity,valid.df_nn$prediction)



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
                                   valence=sensitivity_factor,tempo=.5,duration_ms=.5)
  
  
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
  
  
  sensitivity.nn$Impact_Pos <- round(sensitivity.nn$Populairty_Pos-Base_Popularity.nn,digits=3)
  sensitivity.nn$Impact_Neg <- round(sensitivity.nn$Populairty_Neg-Base_Popularity.nn,digits=3)
  View(sensitivity.nn)
  

