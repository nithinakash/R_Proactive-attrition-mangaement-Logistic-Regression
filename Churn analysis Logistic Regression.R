setwd('C:/Users/Nithin/Downloads/R - Logistic Regression case study')

#**************************************************************************************************************#

#########################
#-->Required Packages<--#
#########################
require(dplyr)
require(stringr)
require(fastDummies)
require(ggplot2)
require(caret)
require(car)
require(Metrics)
require(MLmetrics)
require(sqldf)
require(InformationValue)

#**************************************************************************************************************#

################
#-->Datasets<--#
################

df <- read.csv('Proactive Attrition Management-Logistic Regression Case Study.csv')

#**************************************************************************************************************#

#################
#-->Data Prep<--#
#################

str(df)

df$CSA <- NULL

#**************************************************************************************************************#

#############
#--> UDF <--#
#############

#cont_var_summary
cont_var_summary <- function(x){
  n = length(x)
  nmiss = sum(is.na(x))
  nmiss_pct = mean(is.na(x))
  sum = sum(x, na.rm=T)
  mean = mean(x, na.rm=T)
  median = quantile(x, p=0.5, na.rm=T)
  std = sd(x, na.rm=T)
  var = var(x, na.rm=T)
  range = max(x, na.rm=T)-min(x, na.rm=T)
  pctl = quantile(x, p=c(0, 0.01, 0.05,0.1,0.25,0.5, 0.75,0.9,0.95,0.99,1), na.rm=T)
  return(c(N=n, Nmiss =nmiss, Nmiss_pct = nmiss_pct, sum=sum, avg=mean, meidan=median, std=std, var=var, range=range, pctl=pctl))
}

#outlier_treatment
outlier_treatment <- function(x){
  UC = quantile(x, p=0.99, na.rm=T)
  LC = quantile(x, p=0.01, na.rm=T)
  x = ifelse(x>UC, UC, x)
  x = ifelse(x<LC, LC, x)
  return(x)
}

#missing_value_treatment continuous
missing_value_treatment = function(x){
  x[is.na(x)] = mean(x, na.rm=T)
  return(x)
}

#*#**************************************************************************************************************#

#######################
#-->Data Treatment <--#
#######################

cont_col <- colnames(select_if(df,is.numeric))

df_cont <- df[,cont_col]

#Outlier Treatment & Missing Value treatment for continuous variables

num_sum <- data.frame(t(round(apply(df_cont,2,cont_var_summary),2)))

df_cont <- data.frame(apply(df_cont,2,outlier_treatment))
df_cont <- data.frame(apply(df_cont,2,missing_value_treatment))

#*#**************************************************************************************************************#

########################
#--> Data Splitting <--#
########################

samp <- sample(1:nrow(df_cont), floor(nrow(df_cont)*0.7))

dev <-df_cont[samp,]
val <-df_cont[-samp,]

#*#**************************************************************************************************************#

########################
#--> Model Building <--#
########################
                       
M0 <- glm(CHURN~MARRYNO+
          PHONES+
          NEWCELLY+
          RECCHRGE+
          DROPBLK+
          REFURB+
          MAILRES+
          RETCALL+
          DIRECTAS+
          CUSTCARE+
          EQPDAYS+
          CREDITC+
          WEBCAP+
          CALIBRAT+
          OVERAGE+
          THREEWAY+
          CUSTOMER+
          CREDITDE+
          INCMISS+
          ROAM+
          AGE1+
          INCOME+
          CHANGEM+
          MONTHS+
          CHANGER+
          INCALLS+
          UNIQSUBS+
          CREDITAD,data = dev,
          family = binomial(logit))     

summary(M0)

#--> Columns Removed <--#
#RETACCPT
#OCCCLER
#PRIZMUB
#MARRYYES
#RETCALLS
#PCOWN
#CHURNDEP
#OCCSTUD
#CALLFWDV
#CALLWAIT
#OCCHMKR
#MAILORD
#MARRYUN
#RV
#MAILFLAG
#OPEAKVCE
#CREDITGY
#MOUREC
#CREDITZ
#CREDITCD
#MODELS
#TRAVEL
#OUTCALLS
#MCYCLE
#NEWCELLN
#OCCRET
#OCCCRFT
#PRIZMTWN
#UNANSVCE
#TRUCK
#OCCSELF
#OCCPROF
#OWNRENT
#REFER
#MOU
#SETPRCM
#CREDITAA
#SETPRC
#CREDITB
#PRIZMRUR
#AGE2
#BLCKVCE
#CHILDREN

#*#*#**************************************************************************************************************#

#*#####################
#--> Model Scoring <--#
#######################


dev1 <- cbind(dev, prob = predict(M0, type = "response"))

prob <- predict(M0, val, type = 'response')
test_Y <- val$CHURN
val1 <- cbind(test_Y,prob)
colnames(val1) = c("CHURN","prob")
val1 <- data.frame(val1)

#--> Concordance <--#
Concordance(dev1$CHURN,dev1$prob)
Concordance(val1$CHURN,val1$prob)

#--> AUC <--#
roc_obj <- roc(dev1$CHURN,dev1$prob)
auc(roc_obj)

roc_obj <- roc(val1$CHURN,val1$prob)
auc(roc_obj)

#--> Predicted Class <--#
dev1$pred = ifelse(dev1$prob > 0.5 , 1, 0)
val1$pred = ifelse(val1$prob > 0.5 , 1, 0)

#--> Confusion Matrix <--#
confusionMatrix(dev1$pred,dev1$CHURN)
confusionMatrix(val1$pred,val1$CHURN)

#--> Accuracy <--#
accuracy(dev1$CHURN,dev1$pred)
accuracy(val1$CHURN,val1$pred)

#--> Sensitivity <--#
sensitivity(dev1$CHURN,dev1$pred)
sensitivity(val1$CHURN,val1$pred)

#--> Specificity <--#
specificity(dev1$CHURN,dev1$pred)
specificity(val1$CHURN,val1$pred)

#*#*#**************************************************************************************************************#

#*######################
#--> Best Threshold <--#
########################

roc_obj <-  roc(dev1$CHURN,dev1$prob)
plot(roc_obj)
roc_values <- coords(roc_obj, "best", "threshold")
roc_values

#Best Value is 0.31

#*#*#**************************************************************************************************************#

#*#####################
#--> KS Statistics <--#
#######################

y_prob <- as.data.frame(cbind(dev1$CHURN,dev1$prob))
colnames(y_prob) <- c('CHURN','prob')

decLocations <- quantile(y_prob$prob, probs = seq(0.1,0.9,by=0.1))
y_prob$decile <- findInterval(y_prob$prob,c(-Inf,decLocations, Inf))

decile_grp <- group_by(y_prob,decile)

decile_summ <- summarize(decile_grp, total_cnt=n(), 
                         min_prob=min(p=prob), 
                         max_prob=max(prob), 
                         CHURN_cnt=sum(CHURN),
                         non_CHURN_cnt=total_cnt-CHURN_cnt, 
                         CHURN_rate=(CHURN_cnt/total_cnt)*100)

decile_summ<-arrange(decile_summ, desc(decile))

sum1 <- sum(decile_summ$CHURN_cnt)
sum2 <- sum(decile_summ$non_CHURN_cnt)

decile_summ$CHURN_pct <- ((decile_summ$CHURN_cnt)/sum1)*100
decile_summ$non_CHURN_pct <- ((decile_summ$non_CHURN_cnt)/sum2)*100
decile_summ$cum_CHURN_pct <- cumsum(decile_summ$CHURN_pct)
decile_summ$cum_non_CHURN_pct <- cumsum(decile_summ$non_CHURN_pct)
decile_summ$ks_stats <- abs(decile_summ$cum_CHURN_pct-decile_summ$cum_non_CHURN_pct)

View(decile_summ)

#*#*#**************************************************************************************************************#

#*################################
#--> Using the best Threshold <--#
##################################

#--> Predicted Class <--#
dev1$pred <- ifelse(dev1$prob > 0.31,1,0)
val1$pred <- ifelse(val1$prob > 0.31,1,0)

#--> Concordance <--#
Concordance(dev1$CHURN,dev1$prob)
Concordance(val1$CHURN,val1$prob)

#--> AUC <--#
roc_obj <- roc(dev1$CHURN,dev1$prob)
auc(roc_obj)

roc_obj <- roc(val1$CHURN,val1$prob)
auc(roc_obj)

#--> Confusion Matrix <--#
confusionMatrix(dev1$pred,dev1$CHURN)
confusionMatrix(val1$pred,val1$CHURN)

#--> Accuracy <--#
accuracy(dev1$CHURN,dev1$pred)
accuracy(val1$CHURN,val1$pred)

#--> Sensitivity <--#
sensitivity(dev1$CHURN,dev1$pred)
sensitivity(val1$CHURN,val1$pred)

#--> Specificity <--#
specificity(dev1$CHURN,dev1$pred)
specificity(val1$CHURN,val1$pred)

#*#*#**************************************************************************************************************#

#*###################################
#--> KS Statistics for test data <--#
#####################################

y_prob <- as.data.frame(cbind(val1$CHURN,val1$prob))
colnames(y_prob) <- c('CHURN','prob')

decLocations <- quantile(y_prob$prob, probs = seq(0.1,0.9,by=0.1))
y_prob$decile <- findInterval(y_prob$prob,c(-Inf,decLocations, Inf))

decile_grp <- group_by(y_prob,decile)

decile_summ <- summarize(decile_grp, total_cnt=n(), 
                         min_prob=min(p=prob), 
                         max_prob=max(prob), 
                         CHURN_cnt=sum(CHURN),
                         non_CHURN_cnt=total_cnt-CHURN_cnt, 
                         CHURN_rate=(CHURN_cnt/total_cnt)*100)

decile_summ<-arrange(decile_summ, desc(decile))

sum1 <- sum(decile_summ$CHURN_cnt)
sum2 <- sum(decile_summ$non_CHURN_cnt)

decile_summ$CHURN_pct <- ((decile_summ$CHURN_cnt)/sum1)*100
decile_summ$non_CHURN_pct <- ((decile_summ$non_CHURN_cnt)/sum2)*100
decile_summ$cum_CHURN_pct <- cumsum(decile_summ$CHURN_pct)
decile_summ$cum_non_CHURN_pct <- cumsum(decile_summ$non_CHURN_pct)
decile_summ$ks_stats <- abs(decile_summ$cum_CHURN_pct-decile_summ$cum_non_CHURN_pct)

View(decile_summ)

#*#*#**************************************************************************************************************#