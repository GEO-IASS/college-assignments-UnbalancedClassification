rm(list=ls())
library('clusterSim')
library('e1071')
library('mlbench')
library('readr')
library('caret')
library('rgl')

replaceNAsByMedian <- function(X)
{
  for(i in 1:ncol(X))
  {
    X[is.na(X[,i]),i] <- median(na.omit(X[,i])) 
  }
  return(X)
}

splitDataTrainAndTest<-function(X,Y)
{
  #Reordenando os vetores
  rndIdx <- sample(nrow(X))
  X <- X[rndIdx,]
  Y <- Y[rndIdx]
  
  #Índice dos dados de cada classe
  classRowIDX<-list()
  for(i in 1:length(levels(factor(Y))))
  {
    classRowIDX[[i]]<-which(Y==i)
  }
  
  #Proporção de separação
  proportion <- c(0.8,0.2) 
  
  #Separação dos dados de cada classe
  xclass<-list()
  yclass<-list()
  for(i in 1:length(classRowIDX))
  {
    xclass[[i]]<-X[classRowIDX[[i]],]
    yclass[[i]]<-Y[classRowIDX[[i]]]
  }
  
  #Definindo os limites
  output <- list()
  xtrain <- list()
  xtest <- list()
  ytrain <- list()
  ytest <- list()
  for(i in 1:length(levels(factor(Y))))
  {
    l1<-1:(proportion[1]*nrow(xclass[[i]]))
    l2<-(l1[length(l1)]+1):(nrow(xclass[[i]]))
    
    xtrain[[i]] <- xclass[[i]][l1,]
    xtest[[i]] <- xclass[[i]][l2,]
    
    ytrain[[i]] <- yclass[[i]][l1]
    ytest[[i]] <- yclass[[i]][l2]
    
  }
  
  #Combinando classes para output
  aux1 <- xtrain[[1]]
  aux2 <- xtest[[1]]
  aux3 <- ytrain[[1]]
  aux4 <- ytest[[1]]
  for(i in 1:(length(levels(factor(Y)))-1))
  {
    trainX <- rbind(aux1,xtrain[[i+1]])
    aux1 <- trainX
    
    testX <- rbind(aux2,xtest[[i+1]])
    aux2 <- testX
    
    trainY <- c(aux3,ytrain[[i+1]])
    aux3 <- trainY
    
    testY <- c(aux4,ytest[[i+1]])
    aux4 <- testY
  }
  
  output <- list(trainX = trainX, testX = testX,
                 trainY = trainY, testY = testY)
  
  return(output)
}

trainAndTestSVM <- function(trainX,trainY,testX,testY,C,g)
{
  svm.model<-svm(trainY ~ ., data=trainX,
                 cost=C,gamma=g)
  
  yhat.svm<-as.numeric(predict(svm.model,testX))
  
  yhat.svm[yhat.svm<0]<--1
  yhat.svm[yhat.svm>=0]<-1
  
  acc<-sum(diag(table(yhat.svm,testY)))/sum(table(yhat.svm,testY))
  auc<-AUC::auc(AUC::roc(yhat.svm,factor(testY)))
  
  out<-list(acc=acc,auc=auc)
  
  return(out)
}

crossValidation <- function(trainX,trainY,testX,testY,Crange,gammarange)
{
  idx<-createFolds(1:nrow(trainX), k = 10)
  acc.svm<-matrix(0, nrow=length(gammarange),ncol=length(Crange))
  auc.svm<-acc.svm
  for(i in 1:length(gammarange))
  {
    for(j in 1:length(Crange))
    {
      acc<-c()
      auc<-c()
      t0<-Sys.time()
      for(k in 1:10)
      {
        output <- trainAndTestSVM(trainX = trainX[-idx[[k]],],
                                  trainY = trainY[-idx[[k]]],
                                  testX = trainX[idx[[k]],],
                                  testY = trainY[idx[[k]]],
                                  C = Crange[j],g = gammarange[i])
        
        acc[k] <- output[[1]]
        auc[k] <- output[[2]]
        
      }
      acc.svm[i,j]<-mean(acc)
      auc.svm[i,j]<-mean(auc)
      print(Sys.time()-t0)
      print(c("Internal LOOP:",j,"External LOOP:",i))
      print(c(acc.svm[i,j],auc.svm[i,j]))
    }
  }
  
  output <- list(AUC = auc.svm, ACC = acc.svm)
  return(output)

}

BaseCar <- read_csv("~/Reconhecimento de Padrões/Desbalanceamento/BaseCar.csv")
BaseCar <- data.matrix(BaseCar)

hist(BaseCar[,8])

#Definição dos dados
X <- BaseCar[,2:7]
X <- replaceNAsByMedian(X)
X <- data.Normalization(X,type="n1",normalization="column")
Y <- BaseCar[,8]

#Separação em teste e treino
splitIndex <- createDataPartition(Y,times=1,p=0.7,list=FALSE)
trainX <- X[splitIndex,]
testX <- X[-splitIndex,]
trainY <- Y[splitIndex]
testY <- Y[-splitIndex]

#Ditribuição das classes
trainY[trainY=='0'] <- -1
testY[testY=='0'] <- -1
table(trainY)
table(testY)

#Grid para variação dos parâmetros
gammarange <- c(2 %o% 10^(-1:1))
Crange <- c(2 %o% 10^(-1:1))

out.grid <- crossValidation(trainX,trainY,testX,testY,Crange,gammarange)
persp3d(log(gammarange),log(Crange),
        alpha=0.9,out.grid[[1]],col='lightblue')

persp3d(log(gammarange),log(Crange),
        alpha=0.7,out.grid[[2]],col='red',add=T)

#Seleção de parâmetros e teste com a métrica AUC
c.best <- Crange[which(out.grid[[1]]==max(out.grid[[1]]),arr.ind=TRUE)[1,2]]
gamma.best <- gammarange[which(out.grid[[1]]==max(out.grid[[1]]),arr.ind=TRUE)[1,1]]

svm.model<-svm(trainY ~ ., data=trainX,
               cost=c.best,gamma=gamma.best)

yhat.svm<-as.numeric(predict(svm.model,testX))

yhat.svm[yhat.svm<0]<--1
yhat.svm[yhat.svm>=0]<-1

acc<-sum(diag(table(yhat.svm,testY)))/sum(table(yhat.svm,testY))
auc<-AUC::auc(AUC::roc(yhat.svm,factor(testY)))

#Seleção de parâmetros e teste com a métrica Acurácia
c.best.acc <- Crange[which(out.grid[[2]]==max(out.grid[[2]]),arr.ind=TRUE)[1,2]]
gamma.best.acc <- gammarange[which(out.grid[[2]]==max(out.grid[[2]]),arr.ind=TRUE)[1,1]]

svm.model.acc<-svm(trainY ~ ., data=trainX,
               cost=c.best.acc,gamma=gamma.best.acc)

yhat.svm.acc<-as.numeric(predict(svm.model.acc,testX))

yhat.svm.acc[yhat.svm.acc<0]<--1
yhat.svm.acc[yhat.svm.acc>=0]<-1

acc2<-sum(diag(table(yhat.svm.acc,testY)))/sum(table(yhat.svm.acc,testY))
auc2<-AUC::auc(AUC::roc(yhat.svm.acc,factor(testY)))

table(yhat.svm.acc)
table(yhat.svm)
table(testY)


