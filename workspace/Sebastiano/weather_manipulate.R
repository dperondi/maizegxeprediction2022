library(lubridate)
library(tidyverse)
library(reshape2)
trait_train <- read.csv("A:/Drive_sync/Work_NCSU/Projects/G2F/Data/Training_Data/1_Training_Trait_Data_2014_2021.csv", header = T)


trait_train$plantdate_parsed <- parse_date_time(trait_train$Date_Planted, orders = c('mdy','ymd'))


unique(trait_train$plantdate_parsed)  
unique(trait_train$Date_Planted)  

unique(trait_train$harvestdate_parsed)
trait_train$harvestdate_parsed <- parse_date_time(trait_train$Date_Harvested, orders = c('mdy','ymd'))

hist(month(trait_train$plantdate_parsed))


                             
length(trait_train[trait_train$Date_Planted=="",])


weather_data <- read.csv("A:/Drive_sync/Work_NCSU/Projects/G2F/Data/Training_Data/4_Training_Weather_Data_2014_2021.csv", header = T)
weather_data$Date_parsed <- ymd(weather_data$Date)
weather_data$Year <- year(weather_data$Date_parsed)
weather_data$month <- month(weather_data$Date_parsed)

weather_data$Env_trim <- do.call(rbind.data.frame, (strsplit(weather_data$Env, split = "_")))[,3]


a <- weather_data %>% group_by(Year, month, Env_trim) %>% summarise(T2M_mean = mean(T2M), T2M_min_mean = mean(T2M_MIN))
#a_wide <- spread(a, month, key=c(T2M_mean, T2M_min_mean))
#a_wide <- dcast(a, Env + Year ~ month, value.var = c("T2M_mean","T2M_min_mean"))
a_wide <- dcast(melt(a, id.vars=c("Year", "Env", "month")), Year+Env~variable+month)
