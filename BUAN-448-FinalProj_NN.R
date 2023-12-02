library(neuralnet)
library(forecast)

songs.df <- read.csv("spotify_songs.csv")

str(songs.df)
summary(songs.df)

normalize <- function(x) {
  if (!is.character(x)) {
    if (is.numeric(x)) {
      normalized <- (x - min(x)) / (max(x)-min(x))
      return(normalized)
    } else {
      return(x)
    }
  } else {
    return(x)
  }
}

columns_non_normal <- c("loudness", "tempo", "duration_ms", "track_popularity")
songs.df[columns_non_normal] <- as.data.frame(lapply(songs.df[columns_non_normal], 
                                                     normalize))

songs.df.pop <- subset(songs.df, songs.df$playlist_genre=="pop")
songs.df.rap <- subset(songs.df, songs.df$playlist_genre=="rap")
songs.df.rock <- subset(songs.df, songs.df$playlist_genre=="rock")
songs.df.latin <- subset(songs.df, songs.df$playlist_genre=="latin")
songs.df.rnb <- subset(songs.df, songs.df$playlist_genre=="r&b")
songs.df.edm <- subset(songs.df, songs.df$playlist_genre=="edm")

'
songs.df$playlist_pop <- ifelse(songs.df$playlist_genre=="pop", 1, 0)
songs.df$playlist_rap <- ifelse(songs.df$playlist_genre=="rap", 1, 0)
songs.df$playlist_rock <- ifelse(songs.df$playlist_genre=="rock", 1, 0)
songs.df$playlist_latin <- ifelse(songs.df$playlist_genre=="latin", 1, 0)
songs.df$playlist_rnb <- ifelse(songs.df$playlist_genre=="r&b", 1, 0)
songs.df$playlist_edm <- ifelse(songs.df$playlist_genre=="edm", 1, 0)
songs.df <- subset(songs.df, select = -playlist_genre)
'





######################################
##POP SONGS
#######################################


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




###############################################
#ROCK
###############################################

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


###############################################
##LATIN
###############################################

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

#########################
##ALL SONGS
#########################


train_size_nn <- round(32833*.60)
train_index_nn <- sample(32833,train_size)

train.df_nn <- songs.df[train_index_nn, ]
valid.df_nn <- songs.df[-train_index_nn, ]

"Instrumentalness captured in speechiness
Liveness deemed unconsequential
Loudness captured in energy

"

popularity_model <- neuralnet(track_popularity~
                                danceability+energy+
                                speechiness+acousticness+ 
                                valence+tempo+duration_ms,  
                              data=train.df_nn,
                              hidden=c(2,1))

plot(popularity_model)

valid.prediction <- compute(popularity_model, valid.df)
valid.df_nn$prediction <- valid.prediction$net.result

accuracy(valid.df_nn$track_popularity,valid.df_nn$prediction)




#############################################################
##Using requested data
#################################################

library(tidyverse)
library(lubridate)
library(janitor)
library(jsonlite)
library(skimr)
library(showtext)
library(ggridges)
library(scales)

listen_data <- fromJSON("StreamingHistory0.json") %>% 
  clean_names() %>% 
  mutate(end_time = with_tz(end_time, tzone = "America/New_York"),
         date = date(end_time),
         day_week_temp = wday(end_time),
         weekend = factor(case_when(
           day_week_temp == 6 | day_week_temp == 7 ~ "Weekend",
           day_week_temp == 1 | day_week_temp == 2 | day_week_temp == 3 | day_week_temp == 4 | day_week_temp == 5 ~ "Weekday",
           TRUE ~ NA_character_
         ))) %>% 
  filter(date >= "2021-01-01" & date <= "2023-11-26") %>% 
  arrange(desc(date)) %>% 
  select(-day_week_temp)

skim(listen_data)


listen_data %>%
  group_by(date) %>% 
  mutate(sum_ms_listen = sum(ms_played),
         mins_listen = ((sum_ms_listen / 1000)/60)) %>% 
  ggplot(aes(x = date, y = mins_listen, color = mins_listen)) +
  geom_line() +
  scale_color_viridis_c(option = "plasma") +
  theme_dark() +
  scale_x_date(limits = as.Date(c("2023-01-01","2023-4-27")),
               expand = c(0,0),
               date_breaks = "1 month",
               date_labels = "%B") +
  theme(plot.background = element_rect(fill = "black"),
        legend.position = "none",
        text = element_text(family = "oswald", color = "white", size = 15),
        panel.background = element_rect(fill = "black"),
        axis.text = element_text(color = "white"),
        axis.text.x = element_text(angle = 25),
        axis.text.y = element_text(size = 15, hjust = 1.2),
        plot.title.position = "plot") +
  labs(x = "", y = "Minutes Listened")

listen_data <- fromJSON("StreamingHistory0.json")
listen_data2 <- fromJSON("StreamingHistory1.json")
listen_data3 <- fromJSON("StreamingHistory2.json")
listen_data_total <- rbind(listen_data,listen_data2, listen_data3)

most_listened_songs <- listen_data_total %>%
  group_by(trackName, artistName) %>%
  summarize(total_plays = n()) %>%
  arrange(desc(total_plays))

