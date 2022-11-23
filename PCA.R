set.seed(0)
n = 50
p = 30
x = matrix(rnorm(n*p),nrow=n)

bstar = c(runif(10,0.5,1),runif(20,0,0.3))
mu = as.numeric(x%*%bstar)

par(mar=c(4.5,4.5,0.5,0.5))
hist(bstar,breaks=30,col="gray",main="",
     xlab="True coefficients")

library(MASS)

set.seed(1)
R = 100#replications
nlam = 60
lam = seq(0,25,length=nlam)

fit.ls = matrix(0,R,n)
fit.rid  = array(0,dim=c(R,nlam,n))
err.ls = numeric(R)
err.rid = matrix(0,R,nlam)

for (i in 1:R) {
  cat(c(i,", "))
  y = mu + rnorm(n)
  ynew = mu + rnorm(n)
  
  a = lm(y~x+0)#no intercept
  bls = coef(a)
  fit.ls[i,] = x%*%bls
  err.ls[i] = mean((ynew-fit.ls[i,])^2)
  
  aa = lm.ridge(y~x+0,lambda=lam)
  brid = coef(aa)
  fit.rid[i,,] = brid%*%t(x)
  err.rid[i,] = rowMeans(scale(fit.rid[i,,],center=ynew,scale=F)^2)#sweeping columns
}

aveerr.ls = mean(err.ls)#sample of mse
aveerr.rid = colMeans(err.rid)

bias.ls = sum((colMeans(fit.ls)-mu)^2)/n#mean of bias^2
var.ls = sum(apply(fit.ls,2,var))/n#mean of variance

bias.rid = rowSums(scale(apply(fit.rid,2:3,mean),center=mu,scale=F)^2)/n#
var.rid = rowSums(apply(fit.rid,2:3,var))/n

mse.ls = bias.ls + var.ls
mse.rid = bias.rid + var.rid
prederr.ls = mse.ls + 1
prederr.rid = mse.rid + 1

bias.ls
var.ls
p/n

prederr.ls
aveerr.ls

cbind(prederr.rid,aveerr.rid)

par(mar=c(4.5,4.5,0.5,0.5))
plot(lam,prederr.rid,type="l",
     xlab="Amount of shrinkage",ylab="Prediction error")
abline(h=prederr.ls,lty=2)
text(c(1,24),c(1.48,1.48),c("Low","High"))
legend("topleft",lty=c(2,1),
       legend=c("Linear regression","Ridge regression"))

par(mar=c(4.5,4.5,0.5,0.5))
plot(lam,mse.rid,type="l",ylim=c(0,max(mse.rid)),
     xlab=expression(paste(lambda)),ylab="")
lines(lam,bias.rid,col="red")
lines(lam,var.rid,col="blue")
abline(h=mse.ls,lty=2)
legend("bottomright",lty=c(2,1,1,1),
       legend=c("Linear MSE","Ridge MSE","Ridge Bias^2","Ridge Var"),
       col=c("black","black","red","blue"))
#################
###################### load and normalize the prostate cancer data
prostate=read.table(file="prostate.data",header=T)
prostate.train=prostate[prostate$train==TRUE,]
prostate.test=prostate[prostate$train==FALSE,]

names(prostate.train)
dim(prostate.train)

prostate.X0=as.matrix(prostate.train[,1:8])
prostate.y0=as.vector(prostate.train[,9])


###################### linear regression, subset selection
prostate.ls=lm(prostate.y0~prostate.X0)
summary(prostate.ls)
prostate.ls.0=lm(lpsa~1,data=prostate.train)
prostate.ls.forward=step(prostate.ls.0,scope=list(lower=~1,upper=~lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45),k=2,direction="forward",data=prostate.train)
prostate.ls.forward$coefficients
######forward: lcavol lweight svi lbph pgg45 lcp age gleason

prostate.ls.8=lm(lpsa~lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45,data=prostate.train)
## prostate.ls.backward=step(prostate.ls.8,scope=list(lower=~1,upper=~lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45),direction="backward",data=prostate.train,trace=TRUE)
summary(prostate.ls.8)## gleason dropped
prostate.ls.7=lm(lpsa~lcavol+lweight+age+lbph+svi+lcp+pgg45,data=prostate.train)
summary(prostate.ls.7)## age dropped
prostate.ls.6=lm(lpsa~lcavol+lweight+lbph+svi+lcp+pgg45,data=prostate.train)
summary(prostate.ls.6)## lcp dropped
prostate.ls.5=lm(lpsa~lcavol+lweight+lbph+svi+pgg45,data=prostate.train)
summary(prostate.ls.5)##pgg45 dropped
prostate.ls.4=lm(lpsa~lcavol+lweight+lbph+svi,data=prostate.train)
summary(prostate.ls.4)##lbph dropped
prostate.ls.3=lm(lpsa~lcavol+lweight+svi,data=prostate.train)
summary(prostate.ls.3)##svi dropped
prostate.ls.2=lm(lpsa~lcavol+lweight,data=prostate.train)
summary(prostate.ls.2)##lweight dropped
prostate.ls.1=lm(lpsa~lcavol,data=prostate.train)
summary(prostate.ls.1)##lcavol dropped


###################### load the lars package
library(lars)
prostate.y=prostate.y0-mean(prostate.y0)
prostate.X=scale(prostate.X0,center=TRUE,scale=TRUE)


prostate.lasso=lars(prostate.X,prostate.y,type="lasso",trace=FALSE,normalize=TRUE,intercept=TRUE)
prostate.lars=lars(prostate.X,prostate.y,type="lar",trace=FALSE,normalize=TRUE,intercept=TRUE)
prostate.fs=lars(prostate.X,prostate.y,type="forward.stagewise",trace=FALSE,normalize=TRUE,intercept=TRUE)
prostate.lars
prostate.fs


###################### plot solution paths
par(mar=c(4.5,4.5,0.5,3))

plot(prostate.lasso,xvar="step",breaks=TRUE,plottype="coefficients")
plot(prostate.lasso,xvar="norm",breaks=TRUE,plottype="coefficients")
plot(prostate.lasso,xvar="arc.length",breaks=TRUE,plottype="coefficients")
plot(prostate.lasso,xvar="df",breaks=TRUE,plottype="coefficients")

plot(prostate.lars,xvar="arc.length")

par(mfrow=c(1,2))
plot(prostate.lars,xvar="arc.length")
plot(prostate.lasso,xvar="arc.length")


###################### extract estimated coefficients
predict(prostate.lasso,s=4,type="coefficients",mode="step") ##s=s-1 for our notation
coef(prostate.lasso,s=4,mode="step")
###################### fitted values
predict(prostate.lasso,as.vector(prostate.test[1,1:8]),s=3,type="fit",mode="step")
predict(prostate.lasso,t(prostate.X[1,]),s=3,type="fit",mode="step") ## new input has to be a row vector
predict(prostate.lasso,as.vector(prostate.test[1,1:8]),s=.5,type="fit",mode="fraction")
predict(prostate.lasso,as.vector(prostate.test[1,1:8]),s=2,type="fit",mode="lambda")
predict(prostate.lasso,as.vector(prostate.test[1,1:8]),s=10,type="fit",mode="norm")
###################### get all fitted values on the training set
prostate.lasso.fitted=predict(prostate.lasso,prostate.X,s=3,type="fit",model="step")

#################################################################
###### You have to be careful when using the lars function. #####
###### It gives coefficients in term of the original input  #####
###### matrix X (prostate.X here). If necessary, you need   #####
###### to figure out the right intercept by yourself.       #####
#################################################################


###################### example on the sign change of LARS coefficients
x1=c(1,0,0,0,0)
x1=(x1-mean(x1))/(sd(x1)*sqrt(4))
x2=c(1,0,-.5,0,0)
x2=(x2-mean(x2))/(sd(x2)*sqrt(4))
x3=c(0,1,0,0,0)
x3=(x3-mean(x3))/(sd(x3)*sqrt(4))

X3=cbind(x1,x2,x3)
w3=solve((t(X3)%*%X3),rep(1,3))
y=rnorm(5)
y=(lm(y~X3))$residuals+X3%*%w3+2*x1+.5*x2
y=y-mean(y)
cor(y,X3)
m=lm(y~X3-1)
summary(m)

###### 1st step
plot(0,0,xlim=c(0,1),ylim=c(0,1),type="n",xlab=expression(gamma[1]),ylab="correlations",main="Step 1")
legend("topright",c("x1","x2","x3"),col=c("black","blue","yellow"),lty=c(1,1,1),bg="gray90")
abline(0,0)
abline(cor(y,x1),-1)
abline(cor(y,x2),-t(x2)%*%x1,col="blue")
abline(cor(y,x3),-t(x3)%*%x1,col="yellow")

