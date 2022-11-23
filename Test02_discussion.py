# -*- coding: utf-8 -*-
"""
Math450 - Test02
Model Selection in Multiple Regression
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Model Specification
eqn = "Y~X1+X2+X3+X4"

# Data Source
mydata = pd.read_excel("D:\\workfiles\\450\\Test02\\play.xlsx",header=0,nrows=30,dtype='object')

# Response and Design Matrix preparation 
yvar = eqn.split('~')[0]
xlst = eqn.split('~')[1].split('+')

y = np.matrix(mydata[[yvar]],dtype='float')
X = np.matrix(mydata[xlst],dtype='float')
n = X.shape[0]

X = np.hstack([np.ones((n,1)),X])
b_hat = np.linalg.inv(X.T*X)*X.T*y
print(b_hat)

H = X*np.linalg.inv(X.T*X)*X.T
SSR = y.T*(H-1/n*np.ones((n,n)))*y
SSE = y.T*(np.eye(n)-H)*y
print(SSR)
print(SSE)


# forming the index set for all power subsets 
count = 0
mlst1 = list()
for i in range(4):
    mlst1.insert(count,[i])
    count += 1

count = 0
mlst2 = list()
for i in range(3):
    for j in range(i+1,4):
        mlst2.insert(count,[i,j])
        count += 1

count = 0
mlst3 = list()
for i in range(2):
    for j in range(i+1,3):
        for k in range(j+1,4):
            mlst3.insert(count,[i,j,k])
            count += 1
mlst4=[[i for i in range(4)]]

mlst = mlst1 + mlst2 + mlst3 + mlst4
print(mlst)

subxlst = [xlst[i] for i in mlst[3]]
print(subxlst)
p = len(subxlst)+1
X = np.matrix(mydata[subxlst],dtype='float')

b_hat = np.linalg.inv(X.T*X)*X.T*y
H = X*np.linalg.inv(X.T*X)*X.T
SSR = y.T*(H-1/n*np.ones((n,n)))*y
SSE = y.T*(np.eye(n)-H)*y
adjrsq = 1-(SSE/(n-p))/((SSE+SSR)/(n-1))

def OLS(X,y):
    X = np.hstack([np.ones((X.shape[0],1)),X])
    b_hat = np.linalg.inv(X.T*X)*X.T*y
    return(b_hat)

OLS(X,y)

