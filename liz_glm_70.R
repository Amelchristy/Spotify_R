getwd()
setwd("~/Desktop/Rclass")

spotify_songs <-  read.csv("spotify_songs.csv")

View(spotify_songs)
str(spotify_songs)
summary(spotify_songs)


install.packages('lattice')
library('lattice')
library(ggplot2)
library(caret)
install.packages("spotifyr")
library(spotifyr)
install.packages("ggimage")
library(ggimage)
library(gt)

# Checking for missing values in each column of the spotify_songs dataset
missing_values <- colSums(is.na(spotify_songs))
missing_values

#normalizing the function
normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

#chosen features
features <- c("track_popularity", "playlist_genre", "playlist_subgenre", 
              "danceability", "energy", "key", "loudness", "mode", 
              "speechiness", "acousticness", "instrumentalness", 
              "liveness", "valence", "tempo", "duration_ms")
spotify_songs <- spotify_songs[, features]

# Normalizing certain columns which are the most significant to popularity 
norm <- c("loudness", "danceability", "energy", "duration_ms","instrumentalness","liveness")
spotify_songs[norm] <- as.data.frame(lapply(spotify_songs[norm], normalize))

#creating the"popular" column 
spotify_songs$popular <- ifelse(spotify_songs$track_popularity > 70, 1, 0)
#deleting the "track_popularity" column 
spotify_songs$track_popularity <- NULL
head(spotify_songs)

RNGversion("3.5.2")
set.seed(12345) 
train.index <- sample(1:nrow(spotify_songs), 0.6 * nrow(spotify_songs))
train.index

train.data <- spotify_songs[train.index, ]
valid.data <- spotify_songs[-train.index, ]

#Train a logistic regression model 
model1 <- glm(popular ~ danceability + energy + key + loudness + mode +
                speechiness + acousticness + instrumentalness +
                liveness + valence + tempo + duration_ms,
              data = train.data, family = "binomial")
summary(model1)

#predictions on the valid set
valid.data$predictions <- ifelse(predict(model1, newdata = valid.data, type = "response") > 0.5, 1, 0)

# Accuracy of the model
accuracy<-mean(valid.data$predictions == valid.data$popular)
accuracy

# Confusion matrix
cmatrix_model <- table(valid.data$predictions, valid.data$popular)
cmatrix_model

#MODEL2 use all features
model2 <- glm(popular ~ ., data = train.data, family = "binomial")

summary(model2)

#predictions on the valid set
valid.data$predictions <- ifelse(predict(model2, newdata = valid.data, type = "response") > 0.5, 1, 0)

# Accuracy of the model
accuracy2<-mean(valid.data$predictions == valid.data$popular)
accuracy2

# Confusion matrix
cmatrix_model2 <- table(valid.data$predictions, valid.data$popular)
cmatrix_model2


install.packages("forecast")
library(forecast)

# Backward
model3 <- glm(popular~., data = train.data)
glm.backward <- step(model3,direction = "backward")
#Step:  AIC=11342.4
#popular ~ playlist_subgenre + danceability + energy + loudness + 
#  instrumentalness + liveness + valence + tempo
summary(glm.backward)

glm.backward.pred <- predict(glm.backward, valid.data)
accuracy(glm.backward.pred,valid.data$popular)


#forward
null <- glm(popular ~ 1, data = train.data)
glm.forward<- step(null, scope=list(lower=null, upper=model3), direction="forward")

summary(glm.forward)

glm.forward.pred <- predict(glm.forward,valid.data)
accuracy(glm.forward.pred, valid.data$popular)


#both direction
glm.step <- step(null, scope=list(upper=model3), data = train.data, direction="both")
summary(glm.step) 

glm.step.pred <- predict(glm.step, valid.data)
accuracy(glm.step.pred, valid.data$popular)