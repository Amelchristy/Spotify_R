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
# Modify the 'popular' column using ifelse
spotify_songs$popular <- ifelse(spotify_songs$track_popularity > 80, 1, 0)

# Remove the 'track_popularity' column
spotify_songs$track_popularity <- NULL
head(spotify_songs)

# Split into training (60%) and testing set (40%)
set.seed(12345)
train_test_split <- initial_split(spotify_songs, prop = 0.6)
train <- training(train_test_split)
test <- testing(train_test_split)

summary(train)

# Option 1: Remove rows with NA
train_no_na <- na.omit(train)
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