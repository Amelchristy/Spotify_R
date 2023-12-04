library(tidymodels)
library(xgboost)
library(caret)
library(fastDummies)
library(ggplot2)

# Define the normalization function
normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Read and preprocess the dataset
getwd()
setwd()
spotify_songs <- read.csv("spotify_songs.csv", header = TRUE, sep = ",")
head(spotify_songs)

# Exploratory Data Analysis (optional steps not included here)

# Keep the desired features
features_to_keep <- c("track_popularity", "playlist_genre", "playlist_subgenre", 
                      "danceability", "energy", "key", "loudness", "mode", 
                      "speechiness", "acousticness", "instrumentalness", 
                      "liveness", "valence", "tempo", "duration_ms")
spotify_songs <- spotify_songs[, features_to_keep]

# Normalize certain columns
columns_non_normal <- c("loudness", "tempo", "duration_ms")
spotify_songs[columns_non_normal] <- as.data.frame(lapply(spotify_songs[columns_non_normal], normalize))

# Create dummy variables and remove the original categorical columns
spotify_songs <- dummy_cols(spotify_songs, select_columns = c("playlist_genre", "playlist_subgenre"), remove_selected_columns = TRUE)

# Create binary target variable and convert it to a factor
spotify_songs$popular <- as.factor(spotify_songs$track_popularity > 70)
spotify_songs$track_popularity <- NULL

# Split into training (75%) and testing set (25%)
set.seed(12345)
train_test_split <- initial_split(spotify_songs, prop = 0.75)
train <- training(train_test_split)
test <- testing(train_test_split)

# Prepare data for xgboost
train_x <- as.matrix(train[, -which(names(train) == "popular")])
train_y <- as.numeric(train$popular) - 1

test_x <- as.matrix(test[, -which(names(test) == "popular")])
test_y <- as.numeric(test$popular) - 1

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
conf_mat <- confusionMatrix(as.factor(test_pred), as.factor(test_y))

# Print evaluation results
print(conf_mat)

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


####GLM####

