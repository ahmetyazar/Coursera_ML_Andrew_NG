#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:08:22 2017

@author: yazar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gradient_descent():
    """function(a, b) -> list"""
    print("gradient descent algorithm.")
    print("a new line")


train = pd.read_csv('ex1data2.txt', names=['x1','x2','y']) # pylint: disable-msg=C0103
print('train.shape = {0}'.format(train.shape))
X = train.ix[:,train.columns != 'y']
print('X.shape = {0}'.format(X.shape))
y = train['y']
print('y.shape = {0}'.format(y.shape))
