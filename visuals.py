#!/usr/bin/env python
# coding: utf-8

# In[1]:


# standard imports
import pandas as pd
import numpy as np
# models
import wrangle
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# stats
from scipy import stats
# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[ ]:


def zillow_heatmap():
    corr = train.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap='Purples', annot=True, linewidths=0.5, mask=np.triu(corr))
    plt.ylim
    plt.show()


# In[ ]:


def tax_scatterplot():
    plt.figure(figsize=(15,15))
    sns.jointplot(x="tax_value", y="tax_amount", data=train,  kind='reg')
    plt.show()


# In[ ]:


def variables_box_bar():
    plt.figure(figsize=(15,15))
    plt.subplot(421)
    sns.barplot(x='bathrooms', y='sqr_ft', data= train)

    plt.subplot(422)
    sns.boxplot(x='bathrooms', y='sqr_ft', data=train)

    plt.subplot(423)
    sns.barplot(x='bathrooms', y='year_built', data= train)

    plt.subplot(424)
    sns.boxplot(x='bathrooms', y='year_built', data=train)

    plt.subplot(425)
    sns.barplot(x='bedrooms', y='sqr_ft', data= train)

    plt.subplot(426)
    sns.boxplot(x='bedrooms', y='sqr_ft', data=train)

    plt.subplot(427)
    sns.barplot(x='bedrooms', y='bathrooms', data= train)

    plt.subplot(428)
    sns.boxplot(x='bedrooms', y='bathrooms', data=train)
    plt.tight_layout()
    plt.show()


# In[ ]:


def iqr_variables():
    plt.figure(figsize=(15,15))
    plt.subplot(421)
    sns.barplot(x='bathrooms', y='sqr_ft', data= train)

    plt.subplot(422)
    sns.boxplot(x='bathrooms', y='sqr_ft', data=train)

    plt.subplot(423)
    sns.barplot(x='bedrooms', y='sqr_ft', data= train)

    plt.subplot(424)
    sns.boxplot(x='bedrooms', y='sqr_ft', data=train)

    plt.subplot(425)
    sns.barplot(x='bedrooms', y='bathrooms', data= train)

    plt.subplot(426)
    sns.boxplot(x='bedrooms', y='bathrooms', data=train)
    plt.tight_layout()
    plt.show()


# In[ ]:


def iqr_tax():
    plt.figure(figsize=(15,15))
    sns.jointplot(x="tax_value", y="tax_amount", data=train,  kind='reg')
    plt.show()


# In[ ]:


def tax_resid():
    sns.scatterplot(data = train, x='tax_value', y = 'residuals')
    plt.xlabel('Tax Value')
    plt.ylabel('Residuals')
    plt.show()

