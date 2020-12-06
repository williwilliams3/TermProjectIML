#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:25:59 2020

@author: bwilliams
"""

# 1. SVM MultiClass 


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

svm_model = SVC(**{'C': 109.53031576544694, 'gamma': 0.0005494254346819604, 'kernel': 'rbf'}, probability=True)
svm_model.fit(X_trainval_norm, y_trainval)

y_pred_train_proba_svm = svm_model.predict_proba(X_train_norm)
y_pred_val_proba_svm = svm_model.predict_proba(X_val_norm)
y_pred_test_proba_svm = svm_model.predict_proba(X_test_norm)


print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_svm, axis = 1)) )
print('Accuracy on test set: ',  metrics.accuracy_score(pd.factorize(y_test, sort=True)[0], np.argmax(y_pred_test_proba_svm, axis = 1)) )
# 0.7093023255813954

# 2. XGB Multiclass
import xgboost as xgb

xgb_model =  xgb.XGBClassifier(**{'colsample_bytree': 0.7604881960143233, 'gamma': 0.08182797143285225, 'learning_rate': 0.07927973937929789, 'max_depth': 2, 'n_estimators': 177, 'subsample': 0.8092261699076477}, random_state=42)
xgb_model.fit(X_trainval, y_trainval)

y_pred_train_proba_xgb = xgb_model.predict_proba(X_train)
y_pred_val_proba_xgb = xgb_model.predict_proba(X_val)
y_pred_test_proba_xgb = xgb_model.predict_proba(X_test)


print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_xgb, axis = 1)) )
print('Accuracy on test set: ',  metrics.accuracy_score(pd.factorize(y_test, sort=True)[0], np.argmax(y_pred_test_proba_xgb, axis = 1)) )


# 3. NB PCA
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
pca4 = PCA(n_components=14)
X_trainval_i = pca4.fit_transform(X_trainval)
X_train_i = pca4.transform(X_train)
X_test_i = pca4.transform(X_test)
X_val_i = pca4.transform(X_val)

model = GaussianNB()
model.fit(X_trainval_i, y_trainval)

y_pred_train_proba_nb = model.predict_proba(X_train_i)
y_pred_val_proba_nb = model.predict_proba(X_val_i)
y_pred_test_proba_nb = model.predict_proba(X_test_i)

y_pred_trainval_proba_nb = model.predict_proba(X_trainval_i)


print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_nb, axis = 1)) )
print('Accuracy on test set: ',  metrics.accuracy_score(pd.factorize(y_test, sort=True)[0], np.argmax(y_pred_test_proba_nb, axis = 1)) )


# Blend of Models 

df_train_bin = pd.DataFrame(np.c_[y_pred_train_proba_svm, y_pred_train_proba_xgb, y_pred_train_proba_nb])
df_val_bin = pd.DataFrame(np.c_[y_pred_val_proba_svm, y_pred_val_proba_xgb, y_pred_val_proba_nb])
df_test_bin = pd.DataFrame(np.c_[y_pred_test_proba_svm, y_pred_test_proba_xgb, y_pred_test_proba_nb])


y_pred_train_proba_blend = np.c_[df_train_bin[[0,4,8]].mean(axis = 1), df_train_bin[[1,5,9]].mean(axis = 1), df_train_bin[[2,6,10]].mean(axis = 1) , df_train_bin[[3,7,11]].mean(axis = 1)]
y_pred_val_proba_blend = np.c_[df_val_bin[[0,4,8]].mean(axis = 1), df_val_bin[[1,5,9]].mean(axis = 1), df_val_bin[[2,6,10]].mean(axis = 1) , df_val_bin[[3,7,11]].mean(axis = 1)]
y_pred_test_proba_blend = np.c_[df_test_bin[[0,4,8]].mean(axis = 1), df_test_bin[[1,5,9]].mean(axis = 1), df_test_bin[[2,6,10]].mean(axis = 1) , df_test_bin[[3,7,11]].mean(axis = 1)]

# Multiclass Acuracy 
print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_blend, axis = 1)) )
print('Accuracy on test set: ',  metrics.accuracy_score(pd.factorize(y_test, sort=True)[0], np.argmax(y_pred_test_proba_blend, axis = 1)) )


# Binary Accuracy 
print('Binary accuracy on train set: ', metrics.accuracy_score(y_train == 'nonevent', np.argmax(y_pred_train_proba_blend, axis = 1)==3))
print('Binary accuracy on test set: ',metrics.accuracy_score(y_test == 'nonevent', np.argmax(y_pred_test_proba_blend, axis = 1)==3))

# Accuracy on umbalanced dataset 
# 38% of events  72% of nonevents
y_test == 'nonevent'
y_test_hat = np.argmax(y_pred_test_proba_blend, axis = 1)==3


acc_nonevent = np.mean(y_test_hat[y_test == 'nonevent']==True)
acc_event = np.mean(y_test_hat[~(y_test == 'nonevent')]==False)
acc = 0.5*acc_nonevent + 0.5*acc_event

acc_umbalanced = .60*acc_nonevent + 0.5*acc_event





