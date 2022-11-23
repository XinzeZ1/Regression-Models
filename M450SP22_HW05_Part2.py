# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 20:06:27 2022

@author: 37402
"""

import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:/workfiles/450/HW/HW05/binary.csv')
print(df.shape)
df.head()
df.columns
df.describe()
corr = df.corr()
corr


feature_cols = ['gre','gpa','rank']
X = df[feature_cols]
y = df.admit

formula = 'admit ~ gre + gpa + rank'
model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())
print("Coefficeients")
print(result.params)
print()
print("p-Values")
print(result.pvalues)
print()
print("Dependent variables")
print(result.model.endog_names)

formula2 = 'admit ~ gre +gpa'
model2 = smf.glm(formula = formula2, data=df, family=sm.families.Binomial())
result2 = model2.fit()
print(result.summary())
print("Coefficeients")
print(result2.params)
print()
print("p-Values")
print(result2.pvalues)
print()
print("Dependent variables")
print(result2.model.endog_names)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)



mylogit = smf.glm(formula = formula2, data = df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

mydata = np.array(df)
rows, columns = mydata.shape
N = rows
Y = mydata[:,0]
x0 = np.ones((400,1))
X = np.concatenate([x0, mydata[:,1:2]])


BETA = np.array([0.3175,0,0])
BETA.reshape((3,1))
print(BETA)
DIFF = 1
while DIFF>0.0001:
    MU = 1/np.exp(-np.multiply(X,BETA)+1)
    V = np.diag(np.multiply(MU,(x0-MU)))
    DL = np.trace(X)*np.subtract(x0-MU)
    D2L = -np.trace(X)*V*X
    NBETA = BETA-np.linalg.solve(D2L)*DL
    DIFF = max(NBETA-BETA)
    BETA = NBETA
print(BETA)
    



