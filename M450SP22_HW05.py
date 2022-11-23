# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:18:02 2022

@author: 37402
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from itertools import combinations

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

data = np.loadtxt("D:/workfiles/450/HW/HW04/PSA.txt")
psa = pd.DataFrame(data, columns=['ID','PSA_level','Cancer_Volume','Weight','Age',"Benign_Prostatic_Hyperplasia",
                                  'Seminal_Vesicle_Invasion','Capsular_Penetration','Gleason_Score'])
print(psa.axes)
print(psa.shape)
print(psa.isnull().any())
print(psa.describe())
print(psa.head())
print(psa.tail())


psa = psa.drop(['ID'],axis=1)
print(psa.corr(method="pearson"))
psa.columns = ['PL','CV','Weight','Age','BPH','SVI','CP','GS']
string_cols = '+'.join(psa.columns[1:8])
full = smf.ols('PL~{}'.format(string_cols),data=psa).fit()
print(full.summary())


results = pd.DataFrame({'PL': psa.PL,
                        'resids': full.resid,
                        'std_resids': full.resid_pearson,
                        'fitted': full.predict()})


## raw residuals vs. fitted
with plt.style.context('ggplot'):
    residsvfitted = plt.plot(results['fitted'], results['resids'],'o',color='blue')
l = plt.axhline(y = 0, color = 'red', linestyle = 'solid')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

## q-q plot
with plt.style.context('ggplot'):
    qqplot = sm.qqplot(results['std_resids'], line='s')
plt.title('Normal Q-Q Plot')
plt.show()
    

## scale-location
with plt.style.context('ggplot'):
    scalelocplot = plt.plot(results['fitted'], abs(results['std_resids'])**.5, 'o',color='blue')
plt.xlabel('Fitted values')
plt.ylabel('Square Root of |standardized residuals|')
plt.title('Scale-Location')
plt.show()

## residuals vs. leverage
with plt.style.context('ggplot'):
    residsvlevplot = sm.graphics.influence_plot(full, criterion = 'Cooks', size = 2, color='blue')
plt.title('Residuals vs. Leverage')
plt.show()

durbin_watson(full.resid)

stats.shapiro(full.resid)
stats.kstest(full.resid,'norm')



psa['PL'] = np.log(psa['PL'])

string_cols = '+'.join(psa.columns[1:8])
infull = smf.ols('PL~{}'.format(string_cols),data=psa).fit()
print(infull.summary())

results = pd.DataFrame({'PL': psa.PL,
                        'resids': infull.resid,
                        'std_resids': infull.resid_pearson,
                        'fitted': infull.predict()})


## raw residuals vs. fitted
with plt.style.context('ggplot'):
    residsvfitted = plt.plot(results['fitted'], results['resids'],'o',color='blue')
l = plt.axhline(y = 0, color = 'red', linestyle = 'solid')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

## q-q plot
with plt.style.context('ggplot'):
    qqplot = sm.qqplot(results['std_resids'], line='s')
plt.title('Normal Q-Q Plot')
plt.show()
    

## scale-location
with plt.style.context('ggplot'):
    scalelocplot = plt.plot(results['fitted'], abs(results['std_resids'])**.5, 'o',color='blue')
plt.xlabel('Fitted values')
plt.ylabel('Square Root of |standardized residuals|')
plt.title('Scale-Location')
plt.show()

## residuals vs. leverage
with plt.style.context('ggplot'):
    residsvlevplot = sm.graphics.influence_plot(infull, criterion = 'Cooks', size = 2, color='blue')
plt.title('Residuals vs. Leverage')
plt.show()

durbin_watson(infull.resid)

stats.shapiro(infull.resid)
stats.kstest(infull.resid,'norm')

est_CV = smf.ols('CV ~ Age+Weight+BPH+SVI+CP+GS', psa).fit()
est_Weight = smf.ols('Weight ~ CV+Age+BPH+SVI+CP+GS', psa).fit()
est_Age = smf.ols('Age ~ CV+Weight+BPH+SVI+CP+GS', psa).fit()
est_BPH = smf.ols('BPH ~ CV+Weight+Age+SVI+CP+GS', psa).fit()
est_SVI = smf.ols('SVI ~ CV+Weight+BPH+Age+CP+GS', psa).fit()
est_CP = smf.ols('CP ~ CV+Weight+BPH+SVI+Age+GS', psa).fit()
est_GS = smf.ols('GS ~ CV+Weight+BPH+SVI+CP+Age', psa).fit()

print(1/(1-est_CV.rsquared))
print(1/(1-est_Weight.rsquared))
print(1/(1-est_Age.rsquared))
print(1/(1-est_BPH.rsquared))
print(1/(1-est_SVI.rsquared))
print(1/(1-est_CP.rsquared))
print(1/(1-est_GS.rsquared))



import itertools
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error



def fit_linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared

from tqdm import tnrange, tqdm_notebook


#Initialization variables
Y = psa.PL
X = psa.drop(columns = 'PL', axis = 1)
k = 7
RSS_list, R_squared_list, feature_list = [],[], []
numb_features = []

#Looping over k = 1 to k = 11 features in X
for k in tnrange(1,len(X.columns) + 1, desc = 'Loop...'):

    #Looping over all possible combinations: from 11 choose k
    for combo in combinations(X.columns,k):
        tmp_result = fit_linear_reg(X[list(combo)],Y)   #Store temp result 
        RSS_list.append(tmp_result[0])                  #Append lists
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))   

#Store in DataFrame
df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
# Although what you need to do is to sort the df, on the basis of RSS, than figure out what are the num_features, 
# with lowest RSS, DO the same for R_squared.

df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
print(df_min.head(3))
print(df_max.tail(3))

df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)
df.head()

with plt.style.context('ggplot'):
    fig = plt.figure(figsize = (16,6))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(df.numb_features,df.RSS, alpha = .2, color = 'darkblue' )
    ax.set_xlabel('# Features')
    ax.set_ylabel('RSS')
    ax.set_title('RSS - Best subset selection')
    ax.plot(df.numb_features,df.min_RSS,color = 'r', label = 'Best subset')
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(df.numb_features,df.R_squared, alpha = .2, color = 'darkblue' )
    ax.plot(df.numb_features,df.max_R_squared,color = 'r', label = 'Best subset')
    ax.set_xlabel('# Features')
    ax.set_ylabel('R squared')
    ax.set_title('R_squared - Best subset selection')
    ax.legend()
    plt.show()



remaining_features = list(X.columns.values)
features = []
RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
features_list = dict()

for i in range(1,k+1):
    best_RSS = np.inf
    
    for combo in itertools.combinations(remaining_features,1):

            RSS = fit_linear_reg(X[list(combo) + features],Y)   #Store temp result 

            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1] 
                best_feature = combo[0]

    #Updating variables for next loop
    features.append(best_feature)
    remaining_features.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    features_list[i] = features.copy()

print('Forward stepwise subset selection')
print('Number of features |', 'Features |', 'RSS')
print([(i,features_list[i], round(RSS_list[i])) for i in range(1,5)])


df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
df1['numb_features'] = df1.index

m = len(Y)
p = 7
hat_sigma_squared = (1/(m - p -1)) * min(df1['RSS'])

#Computing
df1['C_p'] = (1/m) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['AIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['BIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] +  np.log(m) * df1['numb_features'] * hat_sigma_squared )
df1['R_squared_adj'] = 1 - ( (1 - df1['R_squared'])*(m-1)/(m-df1['numb_features'] -1))
df1

df1['R_squared_adj'].idxmax()
df1['R_squared_adj'].max()

with plt.style.context('ggplot'):
    variables = ['C_p', 'AIC','BIC','R_squared_adj']
    fig = plt.figure(figsize = (12,12))

    for i,v in enumerate(variables):
        ax = fig.add_subplot(2, 2, i+1)
        ax.plot(df1['numb_features'],df1[v], color = 'lightblue')
        ax.scatter(df1['numb_features'],df1[v], color = 'darkblue')
        if v == 'R_squared_adj':
            ax.plot(df1[v].idxmax(),df1[v].max(), marker = 'x', markersize = 20)
        else:
            ax.plot(df1[v].idxmin(),df1[v].min(), marker = 'x', markersize = 20)
        ax.set_xlabel('Number of predictors')
        ax.set_ylabel(v)
        fig.suptitle('Subset selection using C_p, AIC, BIC, Adjusted R2', fontsize = 16)
        plt.show()

finalm = smf.ols('PL~CV+BPH+SVI+GS',data=psa).fit()
print(finalm.summary())

results = pd.DataFrame({'PL': psa.PL,
                        'resids': finalm.resid,
                        'std_resids': finalm.resid_pearson,
                        'fitted': finalm.predict()})


## raw residuals vs. fitted
with plt.style.context('ggplot'):
    residsvfitted = plt.plot(results['fitted'], results['resids'],'o',color='blue')
l = plt.axhline(y = 0, color = 'red', linestyle = 'solid')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

## q-q plot
with plt.style.context('ggplot'):
    qqplot = sm.qqplot(results['std_resids'], line='s')
plt.title('Normal Q-Q Plot')
plt.show()
    

## scale-location
with plt.style.context('ggplot'):
    scalelocplot = plt.plot(results['fitted'], abs(results['std_resids'])**.5, 'o',color='blue')
plt.xlabel('Fitted values')
plt.ylabel('Square Root of |standardized residuals|')
plt.title('Scale-Location')
plt.show()

## residuals vs. leverage
with plt.style.context('ggplot'):
    residsvlevplot = sm.graphics.influence_plot(infull, criterion = 'Cooks', size = 2, color='blue')
plt.title('Residuals vs. Leverage')
plt.show()

durbin_watson(finalm.resid)

stats.shapiro(finalm.resid)
stats.kstest(finalm.resid,'norm')

est_CV = smf.ols('CV ~ BPH+SVI+GS', psa).fit()
est_BPH = smf.ols('BPH ~ CV+SVI+GS', psa).fit()
est_SVI = smf.ols('SVI ~ CV+BPH+GS', psa).fit()
est_GS = smf.ols('GS ~ CV+SVI+BPH', psa).fit()

print(1/(1-est_CV.rsquared))
print(1/(1-est_BPH.rsquared))
print(1/(1-est_SVI.rsquared))
print(1/(1-est_GS.rsquared))

