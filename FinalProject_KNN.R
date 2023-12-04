getwd()
setwd('~/BUAN448 PreditiveR')
library(caret)
library(ellipsis)
library(FNN)

songs <- read.csv('spotify_songs.csv')

head(songs)
View(songs)

#step1 - categorical -> dummy
d_genre <- dummyVars('~playlist_genre',songs)
songs_genre <- predict(d_genre,songs)

d_subgenre <- dummyVars('~playlist_subgenre',songs)
songs_subgenre <- predict(d_subgenre,songs)

songs <- cbind(songs,songs_genre,songs_subgenre)
songs$is_popular <- ifelse(songs$track_popularity>70,1,0)

songs <- songs[,-c(1,2,3,4,5,6,7,8,9,10,11,14)]

#step2 - scale the features
norm.values <- preProcess(songs[ ,c(3,10,11)], method=c("center", "scale"))
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


#what's the best k?  K=2
best <- which.max(accuracy_df$Accuracy)
print(best)

#analysis
prediction <- knn(train=train.df,
                      test = valid.df,
                      cl=train.df$is_popular,
                      k=2)
conf_matrix <- table(prediction,valid.df$is_popular)

confusionMatrix(conf_matrix)


