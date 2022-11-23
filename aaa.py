# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 03:52:48 2022

@author: 37402
"""

class car():
     
    # init method or constructor
    def __init__(self, model, color):
        self.model = model
        self.color = color
         
    def show(self):
        print("Model is", self.model )
        print("color is", self.color )
         
# both objects have different self which
# contain their attributes
