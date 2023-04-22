#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:08:45 2023

@author: admin
"""

#pull in data as a dataframe 

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


# diabetes_X = diabetes_X[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print("Coefficients: \n", regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
# plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
# diabetes_y_pred = regr.predict(diabetes_X_test)

data = pd.read_csv('data.csv')
features = ['Term', 'NoEmp', 'NewExist', 'CreateJob'\
            , 'RetainedJob', 'FranchiseCode', 'DisbursementGross' \
                , 'GrAppv', 'SBA_Appv', 'Recession']
data = data.dropna()

feature_name = ''

# df = pd.DataFrame({
#     feature_name:np.concatenate((X_train.loc[:,feature_name],X_test.loc[:,feature_name])),
#     'set':['training']*X_train.shape[0] + ['test']*X_test.shape[0]
#     })

X = data[features]
y = data.Selected

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# # fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn import metrics
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()




