library(ROCR)
library("e1071")
library(sas7bdat)
library(ggplot2)
library("plotly")
library(reshape2)
library(caret)
library(broom)

set.seed(4)
getwd()
dirpath <- "D:/Sid_Documents/Knowledge/Syllabus And Lecture/Business Statistics/Project"
setwd(dirpath)

rm(list=ls())
infile.sas<-"develop.sas7bdat"
data.sas <- read.sas7bdat(infile.sas, debug=FALSE)
#EDA
#Get Summary of data to identify variables with missing values
data.summary <- summary(data.sas)
data.summary
#Variable with NANS
input.na <- c("AcctAge", "Phone", "POS", "POSAmt",
           "Inv", "InvBal",  "CC", "CCBal",
           "CCPurc", "Income", "HMOwn", "LORes",
           "HMVal", "Age" ,"CRScore")
#Missing Value imputation using Median
for(i in input.na){
    median.col <- median(data.sas[,i],na.rm = TRUE)
    data.sas[is.na(data.sas[,i]), i] <- median.col
}
summary(is.na(data.sas))

#Converted the character column to factors.
Branch_attr<- as.factor(data.sas$Branch)
Res_attr <- as.factor(data.sas$Res)
data.sas$Branch <- as.numeric(Branch_attr)
data.sas$Res <- as.numeric(Res_attr)

#Normalize Data using Z-Score
y_actual <- data.sas$Ins
data.sas$Ins <- NULL
data.sas <- data.frame(scale(data.sas))
data.sas$Ins <- y_actual
#Convert to Matrix
data.sas.matrix <- as.matrix(data.sas)
data.sas.matrix <- apply(data.sas.matrix,2,as.numeric)


#Get Correlation Matrix
mycorr <- round(cor(data.sas.matrix),2)

melted_cormat <- melt(mycorr)
head(melted_cormat)
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
}
upper_tri <- get_upper_tri(mycorr)
melted_cormat <- melt(upper_tri, na.rm = TRUE)

#Correlation Matrix
corr_plot <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+geom_tile(color = "white")
corr_plot <- corr_plot +scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation") 
corr_plot <- corr_plot+  theme(text= element_text(size=7),axis.text.y = element_blank(),axis.text.x = element_blank())
corr_plot <- corr_plot + ylab("Variable")+xlab("Variable")
corr_plot <- corr_plot +coord_fixed()
corr_plot

interactive_plot <- ggplotly(corr_plot)
interactive_plot

#Correlation Plot for Firms with Strong Positive Correlation
h_corr <- melted_cormat[(melted_cormat$value < 1) & (melted_cormat$value >0.5),]
h_corr_plot <- ggplot(h_corr, aes( Var1,Var2,fill =value,size = value),legend = FALSE) + geom_point(alpha = 0.7,color = "red1")
h_corr_plot <- h_corr_plot + theme(text= element_text(size=7))
h_corr_plot <- h_corr_plot + ylab("Variable")+xlab("Variable")
h_corr_plot
ggplotly(h_corr_plot)

#Remove Highly Correlated Variables
reduced.input <- c("Teller","MM","Income","ILSBal","LOCBal", "POSAmt","NSFAmt","CD","LORes",
                   "CCPurc","ATMAmt","Inv","Dep","CashBk","Moved","IRA","CRScore","IRABal",
                   "AcctAge","SavBal","DDABal","SDB","InArea","Sav","Phone","CCBal","InvBal",
                   "MTG","HMOwn","DepAmt","DirDep","ATM","Age","Ins")
data.sas.reduced <- data.sas[reduced.input]

#Split Data
trainIndex <- createDataPartition(data.sas.reduced$Ins, p = .6667,
                                  list = FALSE,
                                  times = 1)

data.sas.reduced$strata[trainIndex] <- 1
data.sas.reduced$strata[is.na(data.sas.reduced$strata)] <- 2
table(data.sas.reduced$Ins,data.sas.reduced$strata)
train.data <- data.sas.reduced[data.sas.reduced$strata ==1,]
valid.data <- data.sas.reduced[data.sas.reduced$strata ==2,]

train.data$strata <- NULL
valid.data$strata <- NULL

#Fit Logit On Training Data As an Explanatory Approach
model <- glm( Ins ~.,family=binomial(link='logit'),data=train.data)
summary(model)
logit.model <- tidy(model)
write.csv(logit.model,file = "Initial_Logistic.csv",row.names = FALSE)
p_plot <- ggplot(logit.model, aes( term,p.value,fill =p.value,size =p.value),legend = FALSE) + geom_point(alpha = 0.5,color = "red1")
p_plot <- p_plot + theme(text= element_text(size=7),axis.text.x = element_text(angle=90, vjust=0.3))
p_plot <- p_plot + ylab("P-Values")+xlab("Variable")
p_plot + geom_hline(yintercept = 0.05)

#Select Significant Variables from Initial fitting
reduced.input.logit <- c("Teller","MM","ILSBal","LOCBal", "POSAmt","CD",
                   "CCPurc","ATMAmt","Inv","Dep","CashBk","IRA",
                   "AcctAge","SavBal","DDABal","InArea","Sav","Phone",
                   "MTG","DirDep","ATM","Ins")
train.data.reduced <- train.data[reduced.input.logit]
valid.data.reduced <- valid.data[reduced.input.logit]

model <- glm( Ins ~.,family=binomial(link='logit'),data=train.data.reduced)
summary(model)
logit.model.reduced <- tidy(model)
write.csv(logit.model.reduced,file = "Initial_Logistic.csv",row.names = FALSE)

y_train <- train.data.reduced$Ins
y_valid <- valid.data.reduced$Ins
train.data.reduced$Ins <- NULL
valid.data.reduced$Ins <- NULL


###PCA Implementation######
#Computing the Covariance Matrix.
PCA <- function(data){
    covariance <- cov(data)
    #Computing the Eigen values and Eigen vectors of the Covariance Matrix
    eigen_Vals <- eigen(covariance)$values
    eigen_vecs <- eigen(covariance)$vectors
    
    pca_plot <- data.frame(eigen_Vals)
    range <- c(1:nrow(pca_plot))
    pca_plot$range <- range
    plot_pca <- ggplot(data = pca_plot, aes(range, eigen_Vals))+geom_line(color = "red")
    plot_pca <- plot_pca + ylab("Eigen Values")+xlab("PCA Components")
    d <-c(0,0)
    for(i in 1:length(eigen_Vals)){
        var <- sum(eigen_Vals[1:i])/sum(eigen_Vals)
        c <- c(i,var)
        d<-rbind(d,c)
    }
    d <- data.frame(d)
    d <- d[2:22,]
    #d<-d[order(d$X2,decreasing = TRUE),]
    pca_var <- ggplot(data = d, aes(X1, X2))+geom_line(color = "red")
    pca_var <- pca_var + ylab("Variance")+xlab("PCA Components")
    
    return (list(plot_pca,pca_var,eigen_vecs))}
##Calling the PCA function on the training data
return_pca_train <- PCA(train.data.reduced)
return_pca_train[1]
return_pca_train[2]
eigen_vecs_train <- return_pca_train[[3]]
#Calling PCA Function on Valid Data
return_pca_test <- PCA(valid.data.reduced)
eigen_vecs_test <- return_pca_test[[3]]

Select_PC <-function(d,data,vec){
    Projected <-as.matrix(data) %*% vec[,1:d]
    Projected <- data.frame(Projected) 
    return (Projected)}
#ROC AUC Function
Plot_ROC <- function(model,data,y){
    p <- predict(model, newdata=data, type="response")
    pr <- prediction(p, y)
    prf <- performance(pr, measure = "tpr", x.measure = "fpr")
    auc <- performance(pr, measure = "auc")
    auc <- auc@y.values[[1]]
    return(list(auc,prf)) }

#Function to build logistic model
model.build <- function(comp,eigen_vecs_train,eigen_vecs_test){
    PCA_Output <- Select_PC(d=comp,train.data.reduced,eigen_vecs_train)
    PCA_Output$Ins <- y_train
    model.pca <- glm( Ins ~.,family=binomial(link='logit'),data=PCA_Output)
    train.summary <- summary(model.pca)
    bic.model <- BIC(model.pca)
    #Prepare Validation Input
    PCA_Output_test <- Select_PC(d=comp,valid.data.reduced,eigen_vecs_test)
    PCA_Output_test$Ins <- y_valid
    ##Training Accuracy##
    fitted.results <- predict(model.pca,newdata=PCA_Output,type='response')
    fitted.results <- ifelse(fitted.results > 0.5,1,0)
    train.misClasificError <- mean(fitted.results != PCA_Output$Ins)
    train.Accuracy <- 1-train.misClasificError
    #Validation Accuracy
    fitted.results <- predict(model.pca,newdata=PCA_Output_test,type='response')
    fitted.results <- ifelse(fitted.results > 0.5,1,0)
    valid.misClasificError <- mean(fitted.results != PCA_Output_test$Ins)
    valid.accuracy <- 1-valid.misClasificError
    ROC_Logistic <- Plot_ROC(model=model.pca,PCA_Output,y_train)
    roc.plot<- ROC_Logistic[[2]]
    auc<-ROC_Logistic[[1]]
    return(list(model.pca,bic.model,train.Accuracy,valid.accuracy,roc.plot,auc))
}

d <- c("Components","Train.Accuracy","Test.Accuracy","AIC","AUC")
pca.components <- c(17,18,19,20,21)
for(i in pca.components){
        pca.model.build <- model.build(comp = i,eigen_vecs_train=eigen_vecs_train,eigen_vecs_test=eigen_vecs_test)
        analysis <- c(i,pca.model.build[[3]],pca.model.build[[4]],pca.model.build[[2]],pca.model.build[[6]])
        d <- rbind(d,analysis)
}
d<-data.frame(d)
modeling.data <- d
colnames(modeling.data)[1]<-"Components"
colnames(modeling.data)[2]<-"Train.Accuracy"
colnames(modeling.data)[3]<-"Test.Accuracy"
colnames(modeling.data)[4]<-"BIC"
colnames(modeling.data)[5]<-"AUC"
modeling.data<-modeling.data[2:nrow(modeling.data),]
modeling.data$Components <- as.factor(modeling.data$Components)
modeling.data$Train.Accuracy <- as.numeric(as.character(modeling.data$Train.Accuracy))
modeling.data$Test.Accuracy <- as.numeric(as.character(modeling.data$Test.Accuracy))
modeling.data$BIC <- as.numeric(as.character(modeling.data$BIC))
modeling.data$AUC <- as.numeric(as.character(modeling.data$AUC))
modeling.data$Acc_Variance <-modeling.data$Train.Accuracy - modeling.data$Test.Accuracy

bic_plot <- ggplot(modeling.data, aes(Components,log(BIC),group=1),legend = FALSE) + geom_line()
bic_plot <- aic_plot + theme(text= element_text(size=10))
bic_plot <- aic_plot + ylab("log(BIC)")+xlab("Components")
bic_plot
auc_plot <- ggplot(modeling.data, aes(Components,AUC,group=1),legend = FALSE) + geom_line()
auc_plot <- auc_plot + theme(text= element_text(size=10))
auc_plot <- auc_plot + ylab("AUC")+xlab("Components")
auc_plot

acc_plot <- ggplot(modeling.data, aes(Components,group=1)) + 
    geom_line(aes(y = Train.Accuracy, colour = "Training Accuracy")) + 
    geom_line(aes(y = Test.Accuracy, colour = "Validation Accuracy")) +
    ylab("Accuracy")
    
acc_plot
write.csv(modeling.data,file = "ModelingMeasures.csv",row.names = FALSE)

confusion.glm <- function(data, model) {
    prediction <- ifelse(predict(model, data, type='response') > 0.5, TRUE, FALSE)
    confusion  <- table(prediction, as.logical(model$y))
    confusion  <- cbind(confusion, c(1 - confusion[1,1]/(confusion[1,1]+confusion[2,1]), 1 - confusion[2,2]/(confusion[2,2]+confusion[1,2])))
    confusion  <- as.data.frame(confusion)
    names(confusion) <- c('FALSE', 'TRUE', 'class.error')
    confusion
}

#PCA_Output.con <- Select_PC(d=comp,train.data.reduced,eigen_vecs_train)
#PCA_Output.com$y <- y_train
#Prepare Validation Input
PCA_Output_test.con.18 <- Select_PC(d=18,train.data.reduced,eigen_vecs_train)
#PCA_Output_test.con.18$y <- y_valid
PCA_Output_test.con.19 <- Select_PC(d=19,train.data.reduced,eigen_vecs_train)
#PCA_Output_test.con.19$y <- y_valid
pca.18 <- model.build(comp = 18,eigen_vecs_train=eigen_vecs_train,eigen_vecs_test=eigen_vecs_test)
pca.19 <- model.build(comp = 19,eigen_vecs_train=eigen_vecs_train,eigen_vecs_test=eigen_vecs_test)
confusion.18 <- confusion.glm(PCA_Output_test.con.18,model = pca.18[[1]])
confusion.19 <- confusion.glm(PCA_Output_test.con.19,model = pca.19[[1]])

confusion.18
confusion.19

plot(pca.18[[5]])
title("ROC Curve with 18 Components")
plot(pca.19[[5]])
title("ROC Curve with 19 Components")
########Support Vector Machine Classification.##################
#### Radial Basis Function Kernel ##############################
PCA_Output <- Select_PC(d=18,train.data.reduced,eigen_vecs_train)
PCA_Output$Ins <- y_train
PCA_Output_test <- Select_PC(d=18,valid.data.reduced,eigen_vecs_test)
PCA_Output_test$Ins <- y_valid
PCA_Output$Ins[PCA_Output$Ins == 0]<- -1
PCA_Output_test$Ins[PCA_Output_test$Ins == 0]<- -1

y_train_svm <- PCA_Output$Ins
y_test_svm <- PCA_Output_test$Ins
PCA_Output$Ins <- NULL
PCA_Output_test$Ins <- NULL

svm_model <- svm(y_train_svm~.,data =PCA_Output)
summary(svm_model)


pred <- predict(svm_model,PCA_Output_test)
pred <- ifelse(pred > 0 ,1,-1)

tot <- 0
for(i in 1:length(y_test_svm)){
    if(y_test_svm[i] == pred[i]){
        tot <- tot+1
    }
}
accuracy <- (tot/length(y_test_svm))*100
accuracy
# Statistics : Accuracy, Precision, Recall, F-measure, AIC.
# Plot Confusion Matrix. 




