# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:57:29 2022

@author: 37402
"""

class method_eval(object):
    
    def CP(SSE,n,p,sigmas):
        cp = SSE/(sigmas)-n+2*p
        return cp

    def RSQ(SSE,SSR):
        rsq = 1-SSE/(SSE+SSR)
        return rsq

    def ADJRSQ(SSE,SSR,n,p):
        adjrsq = 1-(SSE/(n-p))/((SSE+SSR)/(n-1))
        return adjrsq

    def AIC(SSE,n,p):
        import numpy as np
        aic = n*np.log(SSE/n)+2*p
        return aic

    def SBC(SSE,n,p):
        import numpy as np
        sbc = n*np.log(SSE/n)+np.log(n)*p
        return sbc
    

    
    
    
    