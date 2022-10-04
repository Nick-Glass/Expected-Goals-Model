# Expected-Goals-Model

## Nick Glass

#### 2022-10-03

# Overview:

The goal of this project was to create a model to predict the probability of shot becoming a goal for NHL players. The method used for this task was logistic regression that included variable selection to ensure the best subset of variables were used in the final model. New variables were created from the NHL play by play data in order to give a more detailed analysis. The data for this project came from the NHLscrape package and the NHLapi package. The data was joined in a separate script and imported for this project.

# Abstract:
This model was created in three main steps. The first step included the creation, cleaning, and examination of new variables. These variables are shown in the dictionary. The second step included the model preparation and execution. The final main step showed the model performance and compared the results to other public models.

I found in the creation of this model that each of the methods used for variable selection produced the same results when it came to area under the curve, log loss, and variables included in the model. To see the modeling steps refer to part 7 of the project. The area under the curve for the training data was 0.8214 and the log loss was 0.2029. For the validation data the area under the curve was 0.8202 and the log loss was 0.2047. When predicting the probability for the 2022 season these values were slightly lower but that could be due to the small sample size. The average difference in total expected goals compared to actual total goals was 2.12. To see the model results refer to part 8 of this project. Overall this model does a better job at predicting the probability of a shot becoming a goal than most public models. To see a comparison between this model and other public models see part 9.

![image](https://user-images.githubusercontent.com/113626253/193839636-d5ba508a-5754-4356-b473-e29d79c7b529.png)

# Variable Dictionary:

Is Last Faceoff - Was the previous event a faceoff? (TRUE or FALSE).

Event Time Difference - The time between events in seconds.

X Fixed - The fixed x coordinate corresponding to the location of the event on the rink from left to right with values of -99 to 99 in ft (Rink is 200ft long with 11ft behind each net).

Y Fixed - The fixed y coordinate corresponding to the location of the event on the rink from bottom to top with values of -42 to 42 in ft (Rink is 85ft wide).

Shot Distance - The distance in feet of the shot from the net.

Shot Angle - The angle of the shot from the net.

Shot Distance Diff - The difference in the shot distance from the last shot location.

Shot Angle Diff - The difference in the shot angle from the last shot location.

Is Home - Is the event team the home team? (TRUE or FALSE).

Is Tied - Is the game tied? (TRUE or FALSE).

Rebound - Is the play a rebound? Defined as a shot on goal followed by a shot on goal or a goal by the same event team in the same period, within 3 seconds of each other (TRUE or FALSE).

Rush Shot - Is the shot attempt off the rush? Defined as a shot on goal, a missed shot, or a goal, that occurred within 10 seconds of the opposing teams last shot attempt. (TRUE or FALSE).

Shot Side - Is the player on their strong side or off wing when taking the shot?

High Danger Attempt - Did the shot attempt occur in the slot? Defined as 30ft in front of the crease and 16ft wide between the faceoff circles. (TRUE or FALSE).

High Danger Last - Was the last event a high danger attempt? (TRUE or FALSE).

Strength - The strength of the event that occurred. (Even, Power Play, Short Handed, etc.)

Secondary Type - The type of shot that occurred. (If the event is a missed shot than the secondary type is listed as missed shot).

Is Goal - Did the event result in a goal? (This is the response variable, listed as 0 or 1).


![image](https://user-images.githubusercontent.com/113626253/193837079-9af32d93-53d4-4c67-93c7-16642076eb49.png)

# part 1:
# Load packages -----------------------------------------------------------
library(tidyverse)
library(fastDummies)
library(forecast)
library(pROC)
library(caret)
library(knitr)

# Load data ---------------------------------------------------------------
XG_Data <- read_csv("XG_Clean_Data.csv") # load csv file

# part 2:
# Subset Data -------------------------------------------------------------
#### select columns
XG_df_Sub <- XG_Data %>%
  dplyr::select(Season:Period,Period_Seconds,Event_Team,Event_Player_1_FullName,
                Event,Event_Idx,Event_Type,Secondary_Type,Strength_State,
                Event_Player_1_PositionName,Event_Player_1_Handed,Empty_Net,
                X_Fixed,Y_Fixed,Shot_Distance,Shot_Angle,Event_Team_Id,
                Home_Id,Away_Id,Home_Score,Away_Score)

#### find regular season observations & non shootout events
XG_df_Sub <- XG_df_Sub %>% 
  filter(Season_Type == "R" & Period < 5 & Event_Player_1_PositionName != "Goalie" &
         Event != "Empty Net")

#### find only missed shots, shots on goal, & goals
XG_df_Sub <- XG_df_Sub %>%
  filter(Event %in% c("Goal", "Shot", "Missed Shot", "Faceoff"))

glimpse(XG_df_Sub)

## Analysis:

The data was reduced to the necessary columns and filtered to only include regular season games. Empty net events were mot included in this model do to the vast amount of missing values. A separate model could be made for this specific situation. Shootout and goalie events were also filtered out.

# Part 3:

Create new variables and deal with missing values.

## Create New variables ----------------------------------------------------
#### find faceoffs as last event
XG_df_last_faceoff <- XG_df_Sub %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Last_Event = lag(Event),
         Is_Faceoff_Last = case_when(Last_Event == "Faceoff" ~ TRUE,
                                     Last_Event != "Faceoff" ~ FALSE))

#### count the number of faceoffs 
XG_df_last_faceoff %>%
  ungroup() %>%
  count(Is_Faceoff_Last)

#### find the time between shot attempts when last event was not a faceoff
XG_df_time_diff <- XG_df_last_faceoff %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Event_Time_Diff = (Period_Seconds - lag(Period_Seconds)))

#### remove NA values for event time difference
XG_df_time_diff <- XG_df_time_diff %>%
  filter(!is.na(Event_Time_Diff)) %>%
  dplyr::select(-Empty_Net)

#### check NA values for shot location
XG_df_time_diff %>%
  ungroup() %>%
  count(is.na(X_Fixed) & is.na(Y_Fixed)) 

#### remove NA values for shot location
XG_df_time_diff <- XG_df_time_diff %>%
  filter(!is.na(X_Fixed) & !is.na(Y_Fixed))

#### check NA values for shot distance & angle
XG_df_time_diff %>%
  ungroup() %>%
  count(is.na(Shot_Distance) & is.na(Shot_Angle)) 

#### remove NA values for shot distance & angle
XG_df_time_diff <- XG_df_time_diff %>%
  filter(!is.na(Shot_Distance) & !is.na(Shot_Angle)) 

#### find difference in distance & angle of last shot
XG_df_last_shot <- XG_df_time_diff %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Shot_Distance_Diff = abs(Shot_Distance - lag(Shot_Distance)),
         Shot_Angle_Diff = abs(Shot_Angle - lag(Shot_Angle)))

#### count NA values for shot angle difference & shot distance difference
XG_df_last_shot %>%
  ungroup() %>%
  summarise(Missing_Dist_Diff = sum(is.na(Shot_Distance_Diff)),
            Missing_Angle_Diff = sum(is.na(Shot_Angle_Diff)))

#### remove NA values for shot angle difference & shot distance difference
XG_df_last_shot <- XG_df_last_shot %>%
  filter(!is.na(Shot_Angle_Diff) & !is.na(Shot_Distance_Diff))

#### find home & away teams
XG_home_away <- XG_df_last_shot %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Is_Home = case_when(Event_Team_Id == Home_Id ~ TRUE,
                             Event_Team_Id == Away_Id ~ FALSE))

#### count NA values for home & away 
XG_home_away %>%
  ungroup() %>%
  summarise(Missing_Home_Away = sum(is.na(Is_Home)))

#### find when the games are tied
XG_tied <- XG_home_away %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Is_Tied = case_when(Home_Score == Away_Score ~ TRUE,
                             Home_Score != Away_Score ~ FALSE))

#### count NA values for ties
XG_tied %>%
  ungroup() %>%
  summarise(Missing_Tied = sum(is.na(Is_Tied)))
            
#### find rebounds
XG_df_rebounds <- XG_tied %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Rebound = case_when(Event_Team == lag(Event_Team) & 
                             Period == lag(Period) &
                             Last_Event == "Shot" &
                             Event %in% c("Shot","Goal") &
                             Event_Time_Diff <= 3 ~ TRUE,
                             TRUE ~ FALSE))
                             
#### count NA values for event time difference & rebound
XG_df_rebounds %>%
  ungroup() %>%
  summarise(Missing_Diff = sum(is.na(Event_Time_Diff)),
            Missing_Rebound = sum(is.na(Rebound)))

#### find rush shots
XG_df_rush_shots <- XG_df_rebounds %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Rush_Shot = case_when(Event_Team != lag(Event_Team) & 
                               Period == lag(Period) & 
                               Last_Event %in% c("Shot","Missed Shot") &
                               Event %in% c("Shot","Missed Shot","Goal") &
                               Event_Time_Diff <= 10 ~ TRUE,
                               TRUE ~ FALSE))

#### count NA values for rush shots
XG_df_rush_shots %>%
  ungroup() %>%
  summarise(Missing_Rush = sum(is.na(Rush_Shot)))

#### find shot side
XG_shot_side <- XG_df_rush_shots %>%
  mutate(Shot_Side = case_when(Event_Player_1_Handed == "R" & Y_Fixed >  0 | 
                               Event_Player_1_Handed == "L" & Y_Fixed <= 0 ~ "Off_Wing",
                               Event_Player_1_Handed == "R" & Y_Fixed <= 0 | 
                               Event_Player_1_Handed == "L" & Y_Fixed >  0 ~ "Strong_Side"))
                                   
#### find the number of positive values for Y fixed
XG_shot_side %>%
  ungroup() %>%
  filter(Shot_Side == "Off_Wing") %>%
  count()

#### find the number of negitive values for Y fixed
XG_shot_side %>%
  ungroup() %>%
  filter(Shot_Side == "Strong_Side") %>%
  count()

#### count NA values for shot side
XG_shot_side %>%
  ungroup() %>%
  summarise(Missing_wing = sum(is.na(Shot_Side)))

#### find high danger shots
XG_high_danger <- XG_shot_side %>%
  mutate(High_Danger_Attempt = case_when((8 > Y_Fixed & Y_Fixed > -8) &
                                        (X_Fixed > -89 & X_Fixed < -59)| 
                                        (X_Fixed > 59 & X_Fixed < 89) &
                                        Event %in% c("Shot","Missed Shot","Goal")
                                        ~ TRUE,
                                        TRUE ~ FALSE))
  
#### count NA values for high danger attempt
XG_high_danger %>%
  ungroup() %>%
  summarise(Missing_High_Danger_Attempt= sum(is.na(High_Danger_Attempt))) 

#### find if last event was high danger attempt
XG_high_danger_last <- XG_high_danger %>%
  mutate(High_Danger_Last = case_when(lag(High_Danger_Attempt) & 
                                      Last_Event %in% c("Shot","Missed Shot")
                                      ~ TRUE, TRUE ~ FALSE))

#### count NA values for high danger last
XG_high_danger_last %>%
  ungroup() %>%
  summarise(Missing_High_Danger_Last = sum(is.na(High_Danger_Last)))

#### examine most common strengths
XG_high_danger_last %>%
  ungroup() %>%
  count(Strength_State) %>%
  arrange(desc(n))

#### select most common strengths
XG_df_strength <- XG_high_danger_last %>%
  filter(Strength_State %in% c("5v5","4v4","3v3","6v5","6v4","5v6","4v6",
                               "5v4","5v3","4v5","3v5","4v3","3v4"))

#### combine common strengths
XG_df_strength <- XG_df_strength %>%
  mutate(Strength = case_when(Strength_State %in% c("5v5","4v4","3v3") ~ "Even",
                              Strength_State %in% c("5v4","5v3","4v3") ~ "Power_Play",
                              Strength_State %in% c("4v5","3v5","3v4") ~ "Short_Handed",
                              Strength_State %in% c("6v5","6v4") ~ "Extra_Attacker_For",
                              Strength_State %in% c("5v6","4v6") ~ "Extra_Attacker_Against"))

#### count NA values for secondary type
XG_df_strength %>%
  ungroup() %>%
  summarise(Missing_Secondary = sum(is.na(Secondary_Type)))

#### count missed shot values overall
XG_df_strength %>%
  ungroup() %>%
  summarise(Missed_Shot = sum(is.na(Event == "Missed Shot")))

#### count NA values for missed shot & secondary type
XG_df_strength %>%
  ungroup() %>%
  filter(Event == "Missed Shot") %>%
  summarise(Missing_Secondary = sum(is.na(Secondary_Type)))

#### handle NA values for secondary type
XG_df_strength$Secondary_Type[which(XG_df_strength$Event == "Missed Shot")] <- "Missed Shot" 

#### count NA values for secondary type
XG_df_strength %>%
  ungroup() %>%
  summarise(Missing_Secondary = sum(is.na(Secondary_Type)))

#### remove NA values for secondary type
XG_df_strength <- XG_df_strength %>%
  filter(!is.na(Secondary_Type))

#### make IS_Goal column for response variable
XG_df_goal <- XG_df_strength %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Is_Goal = case_when(Event == "Goal" ~ 1,
                             Event != "Goal" ~ 0))

#### double check NA values 
sort(colSums(is.na(XG_df_goal)), decreasing = TRUE)

#### save as final data frame
XG_df <- XG_df_goal

#### find number of rows & columns of final data frame
number_rows <- nrow(XG_df)

number_col <- ncol(XG_df)

#### look at the new data frame
glimpse(XG_df)

## Analysis:

In this step new variables were created to be included in the model. The missing values were handled in a specific order that the last amount of rows were deleted in order to preserve as many observations as possible. Most of the missing values for the secondary event column were linked to the missed shot events. It appears that if a player misses a shot the shot type is not recorded. In order to deal with these missing values "Missed Shot" was imputed for the respective NA values in the secondary attempt column. The strength state column was reduced to only the major types and combined to form categories such as "Even" or "Power Play". There were r number_rows and r number_col in the final data frame. Please refer to the dictionary for the description of these variables.

# Part 4:

## Exploratory data analysis.

# EDA ---------------------------------------------------------------------
#### summary statistics
summary(XG_df)

### standard deviation
sd_df <- XG_df %>%
  ungroup() %>%
  dplyr::select(X_Fixed:Shot_Angle,Event_Time_Diff:Shot_Angle_Diff)
  
sd <- sort(sapply(sd_df, sd, na.rm = TRUE), decreasing = TRUE)
sd

#### specify columns to change to factor
cols <- c("Season","Season_Type","Period_Type","Period","Event_Team",
          "Event_Player_1_Handed","Event","Event_Type","Secondary_Type",
          "Strength_State","Event_Player_1_PositionName","Is_Faceoff_Last",
          "Event_Player_1_Handed","Is_Faceoff_Last","Is_Home","Is_Tied",
          "Rebound","Rush_Shot","Shot_Side","Strength","High_Danger_Attempt",
          "High_Danger_Last")
          
#### convert columns to factors
XG_df <- XG_df %<>%
  ungroup() %>%
  mutate_each_(funs(factor(.)),cols)

## Discrete Variables ------------------------------------------------------
#### create a data frame for discrete variables
XG_bar_charts <- XG_df %>%
  ungroup() %>%
  dplyr::select(Season,Period,Event,Secondary_Type,Last_Event,Is_Faceoff_Last,
                Strength,Event_Player_1_PositionName,Rush_Shot,Rebound,Shot_Side,
                Is_Home,High_Danger_Attempt,High_Danger_Last,Is_Tied,Is_Goal)

#### Use map, which directly generates a list of plots
bar_charts <- map(names(XG_bar_charts)[1:16],
             ~ggplot(XG_bar_charts, aes(x = !!sym(.x))) +
               geom_bar(color = "#00d100", fill="#24ff24",alpha=0.95) +
               labs(title = .x) +
               theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
                     axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
                     axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
                     axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
                     axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
                     panel.background=element_rect(fill="white"),
                     panel.margin=unit(0.05, "lines"),
                     panel.border = element_rect(color="black",fill=NA,size=1), 
                     strip.background = element_rect(color="black",fill="white",size=1),
                     panel.grid.major=element_blank(),
                     panel.grid.minor = element_blank(),
                     axis.ticks=element_blank()))

bar_charts


## Continuous Variables -----------------------------------------------------
#### create a data frame for continuous variables
XG_hist <- XG_df %>%
  ungroup() %>%
  dplyr::select(X_Fixed:Shot_Angle,Event_Time_Diff:Shot_Angle_Diff)

#### Use map, which directly generates a list of plots
histograms <- map(names(XG_hist)[1:7],
                  ~ggplot(XG_hist, aes(x = !!sym(.x))) +
                    geom_histogram(color = "#00d100", fill="#24ff24",alpha=0.95) +
                    labs(title = .x) +
                    theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
                          axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
                          axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
                          axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
                          axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
                          panel.background=element_rect(fill="white"),
                          panel.margin=unit(0.05, "lines"),
                          panel.border = element_rect(color="black",fill=NA,size=1), 
                          strip.background = element_rect(color="black",fill="white",size=1),
                          panel.grid.major=element_blank(),
                          panel.grid.minor = element_blank(),
                          axis.ticks=element_blank()))

histograms

#### Use map, which directly generates a list of plots
box_plot <- map(names(XG_hist)[1:7],
                  ~ggplot(XG_hist, aes(x=factor(0),y = !!sym(.x))) +
                    geom_boxplot(color = "#00d100", fill="#24ff24",alpha=0.95) +
                    labs(title = .x) +
                    theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
                          axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
                          axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
                          axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
                          axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
                          panel.background=element_rect(fill="white"),
                          panel.margin=unit(0.05, "lines"),
                          panel.border = element_rect(color="black",fill=NA,size=1), 
                          strip.background = element_rect(color="black",fill="white",size=1),
                          panel.grid.major=element_blank(),
                          panel.grid.minor = element_blank(),
                          axis.ticks=element_blank()))

box_plot


glimpse(XG_df)

## Analysis:

Looking at the plots it can be seen that many of the predictor variables are skewed towards the right. For the purpose of this model this was not much of an issue. It would be a problem if the response variable was skewed but that was not the case. Certain columns were converted to factors in order to be turned into binary dummy variables later.

## Part 5:

#### Model preparation

## Model Prep -------------------------------------------------------------
#### remove first three columns
XG_sub <- XG_df %>%
  dplyr::select(-Season_Type,-Event,-Period_Type,-Event_Idx,-Strength_State,
                -Event_Team_Id,-Home_Id,-Away_Id,-Home_Score,-Away_Score)

#### create binary dummy variables
XG_Sub <- dummy_cols(XG_sub, select_columns = c("Secondary_Type",
                                                "Rush_Shot","Shot_Side",
                                                "Rebound","Strength",
                                                "Is_Faceoff_Last",
                                                "Is_Home","Is_Tied",
                                                "Event_Player_1_PositionName",
                                                "High_Danger_Attempt",
                                                "High_Danger_Last"),
                     remove_first_dummy = TRUE, remove_selected_columns = TRUE)

options(scipen = 999)

#### split data into years 2017-2021 & 2021-2022
XG_sub_df <- XG_Sub %>%
  filter(Season %in% c("20172018","20182019","20192020","20202021"))

XG_sub_20212022 <- XG_Sub %>%
  filter(Season %in% c("20212022"))

#### save XG sub df file as CSV
write.csv(XG_sub_df,'XG_sub_df.csv')

#### save XG sub 20212022 file as CSV
write.csv(XG_sub_20212022,'XG_sub_20212022.csv')

## Analysis:

In this step unnecessary variables were removed and binary dummy variables were created for the factors that would be include in the model. The data was separated into two data sets; one that contained the data from 2017-2021 used to train the models, and the other for 2022 data used to predict the expected goals for players in order to compare the prediction with the actual goals scored.

## Part 6:

#### Create the logistic regression models.
# Full Model --------------------------------------------------------------
#### predictive analysis of expected goals data using logistic regression
#### randomly split the data into training (80%) and validation (20%) datasets
set.seed(1)
train_data <- sample(rownames(XG_sub_df), nrow(XG_sub_df) * 0.8)
XG_train <- XG_sub_df[train_data, ]
valid_data <- setdiff(rownames(XG_sub_df), train_data)
XG_valid <- XG_sub_df[valid_data, ]                  

#### train a logistic regression with all predictors
XG_full <- glm(Is_Goal ~ Shot_Distance + Shot_Angle + Rebound_TRUE + Rush_Shot_TRUE 
               + `Secondary_Type_Slap Shot` + `Secondary_Type_Snap Shot` + 
                 `Secondary_Type_Tip-In` + `Secondary_Type_Wrap-around` +
                 `Secondary_Type_Wrist Shot` + Secondary_Type_Deflected +
                 `Secondary_Type_Missed Shot` + Strength_Extra_Attacker_Against +
                 + Strength_Extra_Attacker_For + Strength_Power_Play + 
                 Strength_Short_Handed + Shot_Distance_Diff + Shot_Angle_Diff
                + X_Fixed + Y_Fixed + Event_Time_Diff + Shot_Side_Strong_Side +
                 Is_Faceoff_Last_TRUE + Is_Home_TRUE + Is_Tied_TRUE +
                 Event_Player_1_PositionName_Defenseman +
                 `Event_Player_1_PositionName_Left Wing` + 
                 `Event_Player_1_PositionName_Right Wing` + 
                 High_Danger_Attempt_TRUE + High_Danger_Last_TRUE
               , data = XG_train, family = "binomial")

summary(XG_full)

#### metrics for model fitting
XG_full$deviance

#### AIC
AIC(XG_full)

#### BIC
BIC(XG_full)

#### in-sample prediction 
pred_XG_full_train <- round(predict(XG_full, type = "response"),4)

#### ROC curve
r <- roc(XG_train$Is_Goal, pred_XG_full_train)
plot.roc(r)

#### area under curve
auc(r)

#### out-of-sample prediction
pred_XG_full_valid <- round(predict(XG_full, newdata = XG_valid, 
                                    type = "response"),4)

#### ROC curve
r <- roc(XG_valid$Is_Goal, pred_XG_full_valid)

#### plot ROC
plot.roc(r)

#### area under curve
auc(r)

#### in sample log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_train$Is_Goal, pred_XG_full_train + 0.000000000000001),4)


#### out-of-sample log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_valid$Is_Goal, pred_XG_full_valid + 0.000000000000001),4)

## Analysis:

In this step the data was separated into the training and validation data were it was used to create the full model. The training set included 80% of the data and the validation set included the remaining 20%. The AIC, BIC, area under the curve, and the log loss were calculated for both sets of data.

## Part 7:

#### Preform variable selection to find the best model.

# Variable Selection -------------------------------------------------------
#### forward selection ####
XG_glm_Null <- glm(Is_Goal ~ 1, data = XG_train, family = "binomial")
XG_fwd <- step(XG_glm_Null, scope = list(XG_glm_Null, upper = XG_full), 
               direction = "forward")

# Variable Selection -------------------------------------------------------
#### forward selection model summary
summary(XG_fwd)

#### measures
XG_fwd$deviance
AIC(XG_fwd)
BIC(XG_fwd)

#### in-sample prediction 
pred_XG_fwd_train <- round(predict(XG_fwd, type = "response"),4)

#### ROC curve
r <- roc(XG_train$Is_Goal, pred_XG_fwd_train)
plot.roc(r)

#### area under curve
auc(r)

#### out-of-sample prediction
pred_XG_fwd_valid <- round(predict(XG_fwd, newdata = XG_valid, 
                                   type = "response"),4)

#### ROC curve
r <- roc(XG_valid$Is_Goal, pred_XG_fwd_valid)

#### plot ROC
plot.roc(r)

#### area under curve
auc(r)

#### in sample log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_train$Is_Goal, pred_XG_fwd_train + 0.000000000000001),4)


#### out-of-sample log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_valid$Is_Goal, pred_XG_fwd_valid + 0.000000000000001),4)

#### backward elimination logistic model ####
XG_back <- step(XG_full, direction = "backward")

#### backward elimination model summary
summary(XG_back)

#### measures
XG_back$deviance
AIC(XG_back)
BIC(XG_back)

#### in-sample prediction 
pred_XG_back_train <- round(predict(XG_back, type = "response"),4)

#### ROC curve
r <- roc(XG_train$Is_Goal, pred_XG_back_train)
plot.roc(r)

#### area under curve
auc(r)

#### out-of-sample prediction
pred_XG_back_valid <- round(predict(XG_back, newdata = XG_valid, 
                              type = "response"),4)

#### ROC curve
r <- roc(XG_valid$Is_Goal, pred_XG_back_valid)

#### plot ROC
plot.roc(r)

#### area under curve
auc(r)

#### in sample log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_train$Is_Goal, pred_XG_back_train + 0.000000000000001),4)


#### out-of-sample log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_valid$Is_Goal, pred_XG_back_valid + 0.000000000000001),4)

#### stepwise logistic model ####
XG_step <- step(XG_glm_Null, scope = list(XG_glm_Null, upper = XG_full), direction = "both")

#### stepwise model summary
summary(XG_step)

#### measures
XG_step$deviance
AIC(XG_step)
BIC(XG_step)

#### in-sample prediction 
pred_XG_step_train <- round(predict(XG_step, type = "response"),4)

#### ROC curve
r <- roc(XG_train$Is_Goal, pred_XG_step_train)
plot.roc(r)

#### area under curve
auc(r)

#### out-of-sample prediction
pred_XG_step_valid <- round(predict(XG_step, newdata = XG_valid, 
                                    type = "response"),4)

#### ROC curve
r <- roc(XG_valid$Is_Goal, pred_XG_step_valid)

#### plot ROC
plot.roc(r)

#### area under curve
auc(r)

#### in sample log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_train$Is_Goal, pred_XG_step_train + 0.000000000000001),4)

## Analysis:

The training and validation data was used to preform variable selection with forward selection, backward elimination, and stepwise methods. Like the full model the AIC, BIC, AUC, and log loss were calculated to examine the models performance. Each of these models had the same performance so the stepwise model was selected to predict the 2022 expected goals.

#### create a table for the training model results
Method <- c("Full","Forward","Backward","Stepwise","Step_2022")
Number_varibles <- c(29,25,25,25,25)
AIC <- c(XG_full$aic,XG_fwd$aic,XG_back$aic,XG_step$aic,NA)
BIC <- c(BIC(XG_full),BIC(XG_fwd),BIC(XG_back),BIC(XG_step),NA)
AUC <- c(0.8214,0.8214,0.8214,0.8214,0.8144)
Log_Loss <- c(0.2029,0.2029,0.2029,0.2029,0.2122)

Results_Train <- data.frame(Method,Number_varibles,AIC,BIC,AUC,Log_Loss)
Results_Train

#### create a table for the validation model results
Method <- c("Full","Forward","Backward","Stepwise","Step_2022")
Number_varibles <- c(29,25,25,25,25)
AIC <- c(XG_full$aic,XG_fwd$aic,XG_back$aic,XG_step$aic,NA)
BIC <- c(BIC(XG_full),BIC(XG_fwd),BIC(XG_back),BIC(XG_step),NA)
AUC <- c(0.8201,0.8202,0.8202,0.8202,0.8144)
Log_Loss <- c(0.2047,0.2047,0.2047,0.2047,0.2122)

Results_Valid <- data.frame(Method,Number_varibles,AIC,BIC,AUC,Log_Loss)
Results_Valid

Training_Step_AUC <- 0.8214
Valid_Step_AUC <- 0.8202
Step_2022_AUC <- 0.8144

Training_Step_LL <- 0.2029
Valid_Step_LL <- 0.2047
Step_2022_LL <- 0.2122

#### create tables with results
##### 2022 data was predicted with step wise model
kable(Results_Train, caption = "Training data results")
kable(Results_Valid, caption = "Validation data results")

## Part 8:

Use the data from 2022 NHL season to predict expected goals.

# Prediction Using 2022 Data ----------------------------------------------
XG_df_2022 <- read_csv("XG_sub_20212022.csv") # load csv file

#### out-of-sample prediction for 2022 data
pred_XG_step_2022 <- round(predict(XG_step, newdata = XG_df_2022, 
                                    type = "response"),4)

#### ROC curve
r <- roc(XG_df_2022$Is_Goal, pred_XG_step_2022)
plot.roc(r)

#### area under curve
auc(r)

#### 2022 log loss
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

round(LogLoss(XG_df_2022$Is_Goal, pred_XG_step_2022 + 0.000000000000001),4)

#### combine predicted values with data
XG_players_values <- cbind(XG_df_2022,pred_XG_step_2022)

#### subset data
XG_players_values <- XG_players_values %>%
  ungroup() %>%
  group_by(Event_Player_1_FullName,Event_Team) %>%
  rename(XG = pred_XG_step_2022) %>%
  summarise(Total_XG = sum(XG),
            Total_Actual_Goals = sum(Is_Goal)) %>%
  mutate(XG_difference = abs(Total_Actual_Goals - Total_XG)) %>%
  dplyr::select(Event_Team,Event_Player_1_FullName,Total_XG,
                Total_Actual_Goals,XG_difference) %>%
  arrange(desc(Total_XG))
  
  XG_players_values

#### find average difference
Average_XG_Diff <- XG_players_values %>%
  ungroup() %>%
  summarise(Average_XG_diff = mean(XG_difference))

## look at Lightning players for 2022
XG_players_values %>%
  filter(Event_Team == "Tampa Bay Lightning") %>%
  arrange(desc(Total_XG))
  
## Analysis:

After using the stepwise model to predict the 2022 NHL players expected goals it can be seen that the average difference between the predicted expected goals and the actual goals scored was 2.12. This difference is not too large in the big picture considering that there is a great amount of variability and luck in hockey when it comes to scoring goals. The model also seemed to under estimate the total expected goals for the players that scored a high number of goals. This could have been because these players are some of the most talented players and they could make the most of the play. Another explanation is that this model did not capture certain aspects of the game like passing and therefore would not have been able to predict accordingly.

![image](https://user-images.githubusercontent.com/113626253/193839852-07eede77-82f5-468e-ba35-7855e957a2e9.png)

![image](https://user-images.githubusercontent.com/113626253/193839958-5d4cf48f-baf1-4b13-82ff-a7c646addeb5.png)

![image](https://user-images.githubusercontent.com/113626253/193840110-dacbd899-3364-4f08-96ed-e55680912bb4.png)



## Part 9:

Compare the expected goals from this model to Natural Stat Trick's expected goals.

# comparing predicted values with Natural Stat Trick 2022 Data ------------
NST2022 <- read_csv("NaturalStatTrick_XG_2022.csv") # load csv file

#### find natural stat trick XG by player
NST_XG <- NST2022 %>%
  dplyr::select(Player,Team,Position,Goals,ixG) %>%
  arrange(desc(ixG))

#### find natural stat trick XG difference by player
NST_XG <- NST_XG %>%
  mutate(XG_difference = abs(ixG - Goals))

head(NST_XG,10)

#### find average XG difference for natural stat trick data
NST_XG_diff <- NST_XG %>%
  summarise(Average_XG_Diff = mean(XG_difference))
  
## Analysis:

Comparing this models average expected goals difference of 2.12 to Natural Stat Tricks difference of 2.20, it can be seen that the models are comparable. It is important to note that the data used to model this project had observations deleted that contained missing values, possibly deleting goal events, therefore the total actual goals from the players did not match their real values. The Natural Stat Trick difference included all goals that the players scored in the season thus this might not be a direct comparison.

![image](https://user-images.githubusercontent.com/113626253/193840303-d3811662-57d1-431a-a040-c665e3cc7351.png)

## Part 10:

## Compare model with other public models

#### Harry Shomer's Model

![image](https://user-images.githubusercontent.com/113626253/193840472-b683d671-01a2-4748-aa7a-c2865dc1db05.png)

#### Patrick Bacon's Model

![image](https://user-images.githubusercontent.com/113626253/193840623-809332d7-ab8b-4ed4-b028-0953e91071d6.png)

#### Evolving Hockey's Model

![image](https://user-images.githubusercontent.com/113626253/193840769-36b95a6c-4353-42db-8f90-f2b8192cd4f0.png)

# Conclusion:

Comparing the model created in this project to other public models such as evolving hockey's model, Harry Shomer's model, and Patrick Bacon's model, it can be seen that this model has a higher area under the curve with a value of r Training_Step_AUC for the training set and r Step_2022_AUC for the 2022 prediction. The log loss for this model is comparable with other public models with a value of r Training_Step_LL for the training data and a value of r Step_2022_LL for the 2022 expected goals prediction. Overall this model is better at predicting expected goals and includes a wider range of variables than most public models.
  
