# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 03:44:21 2022

@author: 37402
"""

class coef(object):
    
    def dataload(datapath): 
        import pandas as pd
        mydata = pd.read_csv(datapath,header=0,dtype='object')
        return mydata
    
    def getFunc(file, Y_name):
        import csv
        with open(file, 'r') as f:
            header = next(csv.reader(f))
            if 'ID' in header:
                del header[header.index('ID')]
            try:
                del header[header.index(Y_name)]
            except ValueError:
                raise ValueError(f'The Y column "{Y_name}" is not exist!!!')
            valueDict = {'Y': Y_name}
            for i, column in enumerate(header):
                valueDict[f'X{i + 1}'] = column
            return f'Y~{"+".join([f"X{i}" for i in range(1, len(header) + 1)])}', valueDict  

    def Sigma(X,y,H,n,p):
        import numpy as np
        sigma = (y.T*(np.eye(n)-H)*y)/(n-p)   
        return sigma

    def SSR(y,H,n):
        import numpy as np
        SSR = y.T*(H-1/n*np.ones((n,n)))*y
        return SSR

    def SSE(y,H,n):
        import numpy as np
        SSE = y.T*(np.eye(n)-H)*y
        return SSE


    