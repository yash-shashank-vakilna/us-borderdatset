# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:34:45 2020

@author: yashv
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
import math

plt.style.use('fivethirtyeight') 

df = pd.read_csv('./Border_Crossing_Entry_Data.csv', 
                 index_col='Date', parse_dates=['Date'])
df.head()


# Preprocessing
del df['State']    
del df['Port Name']
del df['Border']

# Getting daily sums and constructing useful measures
df.groupby(['Date','Port Code', 'Measure']).sum()
df.reset_index()
df = df.reset_index()
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['day'] = pd.DatetimeIndex(df['Date']).dayofweek

# Extracting X, and y
X = df.loc[:,['Port Code','Measure','Year','Month','day']].to_numpy()
y = df['Value'].to_numpy()

# Encoding Measure (col 1) into one-hot encodded
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Evaluating model
f"RMSE = {np.round(math.sqrt(mean_squared_error(y_test, y_pred)),2)}"