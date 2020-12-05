#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 03:14:32 2020

@author: bwilliams

Ensamble of models

"""

# 1. SVM Binary 


import numpy as np
import pandas as pd
import os

os.chdir('/Users/bwilliams/GoogleDrive/UniversityOfHelsinki/Fall2020/IML/TeamProject/TermProjectIML/bin')
from SplitData import *
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

svm_model = SVC(C= 810.2342569581549, gamma= 0.0005183165833793786, kernel= 'rbf', probability=True)
svm_model.fit(X_trainval_norm, y_trainval_bin)

y_pred_train_proba_svm = svm_model.predict_proba(X_train_norm)[:,0]
y_pred_val_proba_svm = svm_model.predict_proba(X_val_norm)[:,0]
y_pred_test_proba_svm = svm_model.predict_proba(X_test_norm)[:,0]

print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_svm>0.5))
print('Accuracy on test set: ',metrics.accuracy_score(y_test_bin=='event', y_pred_proba_svm>0.5))


# 2. XGB Binary 
import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
                              colsample_bytree= 0.8854654189948783, gamma=0.05056133806139512,
                              learning_rate= 0.05523204183449923, max_depth= 2, 
                              n_estimators= 267, subsample= 0.6291052025456774)

# {'colsample_bytree': 0.8854654189948783, 'gamma': 0.05056133806139512, 'learning_rate': 0.05523204183449923, 'max_depth': 2, 'n_estimators': 267, 'subsample': 0.6291052025456774}
xgb_model.fit(X_trainval, y_trainval_bin)
y_pred_train_proba_xgb = xgb_model.predict_proba(X_train)[:,0]
y_pred_val_proba_xgb = xgb_model.predict_proba(X_val)[:,0]
y_pred_test_proba_xgb = xgb_model.predict_proba(X_test)[:,0]

print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_xgb>0.5))
print('Accuracy on test set: ',metrics.accuracy_score(y_test_bin=='event', y_pred_proba_xgb>0.5))




# 3. Log Binary 
from sklearn.linear_model import LogisticRegression
import numpy.random as npr
# Accuracy on test set:  0.8846153846153846
# Accuracy on validation set:  0.8461538461538461
# Accuracy on train set:  0.9415584415584416
npr.seed(42)
logistic_regression= LogisticRegression()
logistic_regression.fit(X_trainval_norm, y_trainval_bin)
y_pred_train_proba_log = logistic_regression.predict_proba(X_train_norm)[:,0]
y_pred_val_proba_log = logistic_regression.predict_proba(X_val_norm)[:,0]
y_pred_test_proba_log = logistic_regression.predict_proba(X_test_norm)[:,0]

print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_log>0.5))
print('Accuracy on test set: ',metrics.accuracy_score(y_test_bin=='event', y_pred_proba_log>0.5))



# 4. NB Binary
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
pca1 = PCA(n_components=12)
X_train_i = pca1.fit_transform(X_train)
X_test_i = pca1.transform(X_test)
X_val_i = pca1.transform(X_val)

model = GaussianNB()
model.fit(X_train_i, y_train_bin)
y_pred_train_proba_nb = model.predict_proba(X_train_i)[:,0]
y_pred_val_proba_nb = model.predict_proba(X_val_i)[:,0]
y_pred_test_proba_nb = model.predict_proba(X_test_i)[:,0]

print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_nb>0.5))
#print('Accuracy on train set: ',metrics.accuracy_score(y_val_bin=='event', y_pred_val_proba_nb>0.5))
print('Accuracy on test set: ',metrics.accuracy_score(y_test_bin=='event', y_pred_proba_nb>0.5))

# Blend of Models 

df_train_bin = pd.DataFrame(dict( xbb = y_pred_train_proba_xgb, log = y_pred_train_proba_log, nb = y_pred_train_proba_nb))
df_val_bin = pd.DataFrame(dict( xbb = y_pred_val_proba_xgb, log = y_pred_val_proba_log, nb = y_pred_val_proba_nb))
df_test_bin = pd.DataFrame(dict( xbb = y_pred_test_proba_xgb, log = y_pred_test_proba_log, nb = y_pred_test_proba_nb))

y_pred_train_proba_blend = df_train_bin.mean(axis = 1)
y_pred_val_proba_blend = df_val_bin.mean(axis = 1)
y_pred_test_proba_blend = df_test_bin.mean(axis = 1)


print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_blend>0.5))
print('Accuracy on validation set: ',metrics.accuracy_score(y_val_bin=='event', y_pred_val_proba_blend>0.5))
print('Accuracy on test set: ',metrics.accuracy_score(y_test_bin=='event', y_pred_test_proba_blend>0.5))





