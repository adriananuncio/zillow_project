#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# standard imports
import pandas as pd
import numpy as np
# stats
from scipy import stats
# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[ ]:


def retrieve_IQR(df, col_list, k):
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


# In[ ]:


def hypothesis_test(x, y):
    α = 0.05
    r, p = stats.pearsonr(x, y)
    if p > α:
        print(f'P-value: {p} \nr-value: {r} \nI fail to reject the null hypothesis.')
    else:
        print(f'P-value: {p} \nr-value: {r} \nI reject the null hypothesis.')

