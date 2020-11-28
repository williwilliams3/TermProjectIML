#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:17:49 2020

@author: bwilliams
"""


import numpy as np
import pandas as pd
import os

os.chdir('/Users/bwilliams/GoogleDrive/UniversityOfHelsinki/Fall2020/IML/TeamProject/TermProjectIML/bin')
from SplitData import *

from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import plot_confusion_matrix

import xgboost as xgb

from scipy.stats import uniform, randint

# Cross validation on grid of values XGB

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 400), # default 100
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_trainval, y_trainval_bin)
xgb_params = search.best_params_

print(search.best_params_)



xgb_model = search.best_estimator_

y_pred_train = xgb_model.predict(X_trainval)

print("Accuracy on Train:",metrics.accuracy_score(y_trainval_bin, y_pred_train))


y_pred = xgb_model.predict(X_test)
# probabilities of class event
y_pred_proba = xgb_model.predict_proba(X_test)[:,0]

print('5-Fold CV Accuracy: ', search.best_score_)

print("Accuracy on Test:",metrics.accuracy_score(y_test_bin, y_pred))


plot_confusion_matrix(xgb_model, X_test, y_test_bin,
                                 cmap=plt.cm.Blues,
                                 # normalize='true'
                                 )



y_test_01 = pd.get_dummies(y_test_bin)['event']

fpr, tpr, thresholds = metrics.roc_curve(y_test_01, y_pred_proba)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr,tpr , marker='.', label='DT')
plt.title('AUC: '+ str(auc) )
plt.plot(fpr, fpr, linestyle='--', label='Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


########################################################
# Multiclass

xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 400), # default 100
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_trainval, y_trainval)
xgb_params = search.best_params_

print(search.best_params_)



xgb_model = search.best_estimator_

y_pred_train = xgb_model.predict(X_trainval)

print("Accuracy on Train:",metrics.accuracy_score(y_trainval, y_pred_train))


y_pred = xgb_model.predict(X_test)

print('5-Fold CV Accuracy: ', search.best_score_)

print("Accuracy on Test:",metrics.accuracy_score(y_test, y_pred))


plot_confusion_matrix(xgb_model, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 # normalize='true'
                                 )


