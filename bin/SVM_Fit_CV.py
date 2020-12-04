#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:19:53 2020

@author: bwilliams
"""

import numpy as np
import pandas as pd
import os

os.chdir('/Users/bwilliams/GoogleDrive/UniversityOfHelsinki/Fall2020/IML/TeamProject/TermProjectIML/bin')
from SplitData import *


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from sklearn.utils.fixes import loguniform

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

params = {'C': loguniform(1e0, 1e3),
 'gamma': loguniform(1e-4, 1e-3),
 'kernel': ['linear','sigmoid','rbf']}



svclassifier = SVC(probability=True)

search = RandomizedSearchCV(svclassifier, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_trainval_norm, y_trainval_bin)
svm_params = search.best_params_

print(search.best_params_)
# {'C': 810.2342569581549, 'gamma': 0.0005183165833793786, 'kernel': 'rbf'}

svm_model = search.best_estimator_

y_pred_train = svm_model.predict(X_trainval_norm)

print("Accuracy on Train:",metrics.accuracy_score(y_trainval_bin, y_pred_train))

y_pred = svm_model.predict(X_test_norm)
# probabilities of class event
y_pred_proba = svm_model.predict_proba(X_test_norm)[:,0]

print('5-Fold CV Accuracy: ', search.best_score_)

print("Accuracy on Test:",metrics.accuracy_score(y_test_bin, y_pred))


plot_confusion_matrix(svm_model, X_test_norm, y_test_bin,
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

params = {'C': loguniform(1e0, 1e3),
 'gamma': loguniform(1e-4, 1e-3),
 'kernel': ['linear','sigmoid','rbf']}



svclassifier = SVC(probability=True)

search = RandomizedSearchCV(svclassifier, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_trainval_norm, y_trainval)
svm_params = search.best_params_

print(search.best_params_)

svm_model = search.best_estimator_

y_pred_train = svm_model.predict(X_trainval_norm)

print("Accuracy on Train:",metrics.accuracy_score(y_trainval, y_pred_train))

y_pred = svm_model.predict(X_test_norm)
# probabilities of class event
y_pred_proba = svm_model.predict_proba(X_test_norm)[:,0]

print('5-Fold CV Accuracy: ', search.best_score_)

print("Accuracy on Test:",metrics.accuracy_score(y_test, y_pred))


plot_confusion_matrix(svm_model, X_test_norm, y_test,
                                 cmap=plt.cm.Blues,
                                 # normalize='true'
                                 )

