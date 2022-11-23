library(tidyr)
library(dplyr)
library(lubridate)
library(ggplot2)

tmp <- read.fwf("D:/workfiles/450/HW/USC00010063.dly",widths = c(11, 4, 2, 4, rep(c(5, 1, 1, 1),31)))
Weather <- tmp[,c(1:4,seq(5,125,by=4))]
colnames(Weather) <- c("ID","Year","Month","Feature",paste("Day",c(1:31),sep=''))

TMAX <- subset(Weather,Feature =="TMAX", select=-c(ID,Feature))

library(tidyr)
TMAX_long <- pivot_longer(TMAX, -c(Year, Month), values_to = "Tmax", names_to = "Day")

TMAX_long$Day <- as.integer(gsub("[Day]", "", TMAX_long$Day))

indx1 <- TMAX_long$Month %in% c(4,6,9,11) & TMAX_long$Day==31
indx2 <- TMAX_long$Month == 2 & TMAX_long$Day > 29
indx3 <- TMAX_long$Month == 2 & ((TMAX_long$Year%%4 != 0) |
                                   (TMAX_long$Year%%400 != 0) & (TMAX_long$Year%%100 == 0)) &
  TMAX_long$Day == 29
dindx <- indx1 | indx2 | indx3
TMAX_long = TMAX_long[!dindx,]

TMAX_long$tdate <- ymd(paste(TMAX_long$Year,TMAX_long$Month,TMAX_long$Day,sep='-'))
TMAX_long$Tmax[TMAX_long$Tmax == -9999] <- NA
TMAX_long %>%
  ggplot(aes(x=tdate,y=Tmax,colour=Year)) +
  geom_line() +
  scale_colour_gradientn(colours=rainbow(17))

TMAX_long$ydays <- TMAX_long$tdate-ymd(paste(TMAX_long$Year,1,1,sep='-'))+1
TMAX_long %>%
  ggplot(aes(x=ydays,y=Tmax,colour=Year)) +
  geom_line(alpha=0.3) +
  scale_colour_gradientn(colours=rainbow(17)) + theme_classic()

TMAX_long %>%
  ggplot(aes(x=ydays,y=Tmax,colour=Year)) +
  geom_line() +
  scale_colour_gradientn(colours=rainbow(17)) + theme_classic() +
  facet_wrap(~Year)