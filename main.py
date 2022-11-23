# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 21:33:50 2022

@author: 37402
"""

import sys
sys.path.append("D:\\workfiles\\450\\Test02\\")

import numpy as np
import pandas as pd

from evaluation import method_eval
from selection import forward_regression
from selection import backward_elimination
from Preprocessing import coef

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', 1000)

n = 0
p = 0
R_squaredv = list()
ADJR_squaredv = list()
AICv = list()
CPv = list()
SBCv = list()
outlist = list()
outp = list()

df = coef.dataload("play.csv")
df = pd.DataFrame(df,dtype='float')
subc = df.shape[1]


eqn = "Y~X1+X2+X3+X4"

yvar = eqn.split('~')[0]
xlst = eqn.split('~')[1].split('+')

y = np.matrix(df[[yvar]],dtype='float')
X = np.matrix(df[xlst],dtype='float')
n = X.shape[0]
X_train = df.loc[:,("X1","X2","X3","X4")]
Y_train = df.loc[:,("Y")]



X = np.hstack([np.ones((n,1)),X])
b_hat = np.linalg.inv(X.T*X)*X.T*y
H = X*np.linalg.inv(X.T*X)*X.T
sigmas = coef.Sigma(X, y, H, n, p)


count = 0
mlst1 = list()
for i in range((subc-1)):
    mlst1.insert(count,[i])
    count += 1

count = 0
mlst2 = list()
for i in range((subc-2)):
    for j in range(i+1,(subc-1)):
        mlst2.insert(count,[i,j])
        count += 1

count = 0
mlst3 = list()
for i in range((subc-3)):
    for j in range(i+1,(subc-2)):
        for k in range(j+1,(subc-1)):
            mlst3.insert(count,[i,j,k])
            count += 1
mlst4=[[i for i in range((subc-1))]]

mlst = mlst1 + mlst2 + mlst3 + mlst4
print(mlst)


for l in range(len(mlst)):
    subxlst = [xlst[i] for i in mlst[l]]
    print(subxlst)
    outlist.append(subxlst)
    p = len(subxlst)+1
    outp.append(p)
    X = np.matrix(df[subxlst],dtype='float')
    b_hat = np.linalg.inv(X.T*X)*X.T*y
    H = X*np.linalg.inv(X.T*X)*X.T
    SSR = coef.SSR(y, H, n)
    SSE = coef.SSE(y, H, n)
    R_squaredv.append(method_eval.RSQ(SSE, SSR)) 
    ADJR_squaredv.append(method_eval.ADJRSQ(SSE, SSR, n, p)) 
    AICv.append(method_eval.AIC(SSE, n, p)) 
    CPv.append(method_eval.CP(SSE, n, p, sigmas)) 
    SBCv.append(method_eval.SBC(SSE, n, p))

output = pd.DataFrame({"Model":outlist, "p":outp,"R Squared":R_squaredv, "Adjusted R squared":ADJR_squaredv,
          "Cp":CPv, "AIC":AICv, "SBC":SBCv})   
print(output)


outputF = forward_regression(X_train, Y_train, 0.01,idetail=True)
print(outputF)
outputB = backward_elimination(X_train, Y_train, 0.01,idetail=True)
print(outputB)
