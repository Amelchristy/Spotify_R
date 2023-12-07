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


#code to create binary target variable
spotify_songs$popular <- ifelse(spotify_songs$track_popularity > 70, 1, 0)
spotify_songs$track_popularity <- NULL


# Proceed with the train-test split and further analysis as before


set.seed(12345)
train_test_split <- initial_split(spotify_songs, prop = 0.6)
train <- training(train_test_split)
test <- testing(train_test_split)

# Prepare data for xgboost
train_x <- as.matrix(train[, -which(names(train) == "popular")])
train_y <- as.numeric(train$popular)

test_x <- as.matrix(test[, -which(names(test) == "popular")])
test_y <- as.numeric(test$popular)

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)


# Define parameters and perform cross-validation
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# 10-fold cross-validation
cv_model1 <- xgb.cv(params = params, data = xgb_train, nrounds = 1000, nfold = 8, verbose = 0, early_stopping_rounds = 10)

# Train the model using optimal number of rounds
optimal_nrounds <- cv_model1$best_iteration
optimal_nrounds
xgb_model <- xgboost(params = params, data = xgb_train, nrounds = optimal_nrounds, verbose = 0)

# Predict and evaluate
test_pred_probs <- predict(xgb_model, xgb_test)
test_pred <- ifelse(test_pred_probs > 0.5, 1, 0)

# Confusion matrix and additional evaluation metrics
conf_mat <- confusionMatrix(as.factor(test_pred), as.factor(test_y))

# Print evaluation results
print(conf_mat)
importance_matrix <- xgb.importance(feature_names = colnames(train_x), model = xgb_model)
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



# Split into training (75%) and testing set (25%)
set.seed(12345)
train_test_split <- initial_split(spotify_songs, prop = 0.75)
train <- training(train_test_split)
test <- testing(train_test_split)

library(xgboost)
library(caret)

# Assuming 'train' and 'test' datasets are already prepared and 'popular' is a binary numeric variable with values 0 and 1

# Prepare data for xgboost
train_x <- as.matrix(train[, -which(names(train) == "popular")])
train_y <- as.numeric(train$popular) # Convert to numeric if it's not already

test_x <- as.matrix(test[, -which(names(test) == "popular")])
test_y <- as.numeric(test$popular) # Convert to numeric if it's not already

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

# Define parameters and perform cross-validation
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# 8-fold cross-validation (use nfold = 10 for 10-fold)
cv_model1 <- xgb.cv(params = params, data = xgb_train, nrounds = 1000, nfold = 8, verbose = 0, early_stopping_rounds = 10)

# Train the model using optimal number of rounds
optimal_nrounds <- cv_model1$best_iteration
xgb_model <- xgboost(params = params, data = xgb_train, nrounds = optimal_nrounds, verbose = 0)

# Predict and evaluate
test_pred_probs <- predict(xgb_model, xgb_test)
test_pred <- ifelse(test_pred_probs > 0.5, 1, 0)

# Convert predictions and true labels to factors with explicit level ordering
test_y <- factor(test_y, levels = c(1, 0))
test_pred <- factor(test_pred, levels = c(1, 0))

# Generate the confusion matrix
conf_mat <- confusionMatrix(test_pred, test_y)

# Print the confusion matrix
print(conf_mat)

importance_matrix <- xgb.importance(feature_names = colnames(train_x), model = xgb_model)
# Feature importance
importance_plot <- xgb.plot.importance(importance_matrix)

# Modify the plot to change bar color to green
library(ggplot2)
library(xgboost)


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



# Assuming spotify_songs has 'loudness', 'instrumentalness', and 'popular' columns
# 'popular' is expected to be a binary factor or numeric variable with values 0 and 1

ggplot(spotify_songs, aes(x = loudness, y = instrumentalness, color = as.factor(popular))) +
  geom_point() +
  labs(x = "Loudness", y = "Instrumentalness", title = "Scatter plot of Loudness vs Instrumentalness, Colored by Popularity") +
  scale_color_manual(values = c("gray4", "green")) +  # Set colors for categories 0 and 1
  theme_minimal() +
  theme(legend.title = element_blank())  # Optionally remove the legend title

head(spotify_songs)
####GLM####
spotify_songs2 <- read.csv("spotify_songs.csv", header = TRUE, sep = ",")
spotify_songs2 <- spotify_songs2[, features_to_keep]

# Normalize certain columns
columns_non_normal <- c("loudness", "tempo", "duration_ms")
spotify_songs[columns_non_normal] <- as.data.frame(lapply(spotify_songs[columns_non_normal], normalize))

# Create dummy variables and remove the original categorical columns
spotify_songs <- dummy_cols(spotify_songs, select_columns = c("playlist_genre", "playlist_subgenre"), remove_selected_columns = TRUE)


#code to create binary target variable
spotify_songs2$popular <- ifelse(spotify_songs2$track_popularity > 70, 1, 0)

head(spotify_songs2)
tail(spotify_songs2)
