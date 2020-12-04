#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 01:11:59 2020

@author: bwilliams

Choosing the best parameters found for SVM, we try pca from 1-100 variables, finding optimal reduction

We also try K-Best Varibles 
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
from sklearn.decomposition import PCA


params = {'C': 810.2342569581549, 'gamma': 0.0005183165833793786, 'kernel': 'rbf'}
svm_model = SVC(C= 810.2342569581549, gamma= 0.0005183165833793786, kernel= 'rbf', probability=True)

# PCA
results = np.zeros((100, 4))
for i in range(X_trainval.shape[1]):
    pca = PCA(n_components=i+1)
    X_trainval_i = pca.fit_transform(X_trainval_norm)
    X_test_i = pca.transform(X_test_norm)
    
    
    svm_model.fit(X_trainval_i, y_trainval_bin)
    
    y_pred_train = svm_model.predict(X_trainval_i)
    y_pred = svm_model.predict(X_test_i)
    
    
    acc_train = metrics.accuracy_score(y_pred_train, y_trainval_bin)
    acc_val = cross_val_score(svm_model, X_trainval_i, y_trainval_bin, cv=5).mean()
    acc_test = metrics.accuracy_score(y_test_bin, y_pred)
    results[i] = np.array([i+1, acc_train, acc_val, acc_test])

df_res =  pd.DataFrame(results, columns = ['Components','AccTrain','AccVal','AccTest'])
df_res.drop(columns = 'Components').plot()

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
def normalize_0to1(df):
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_norm


def find_n_best_feature(X, y, n):
    bestfeatures = SelectKBest(score_func=chi2, k=n)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['col_name','Score']
    featureScores= featureScores.nlargest(n,'Score')
    cols = list(featureScores["col_name"])
    return cols
# Best K features
results = np.zeros((100, 4))
X_trainval_0to1 = normalize_0to1(X_trainval)

for i in range(X_trainval.shape[1]):
    bestFeatureCols = find_n_best_feature(X_trainval_0to1, y_trainval, i+1)
    X_trainval_i = X_trainval_norm[bestFeatureCols]
    X_test_i = X_test_norm[bestFeatureCols]
    
    
    svm_model.fit(X_trainval_i, y_trainval_bin)
    
    y_pred_train = svm_model.predict(X_trainval_i)
    y_pred = svm_model.predict(X_test_i)
    
    
    acc_train = metrics.accuracy_score(y_pred_train, y_trainval_bin)
    acc_val = cross_val_score(svm_model, X_trainval_i, y_trainval_bin, cv=5).mean()
    acc_test = metrics.accuracy_score(y_test_bin, y_pred)
    results[i] = np.array([i+1, acc_train, acc_val, acc_test])

df_res =  pd.DataFrame(results, columns = ['BestKFeatures','AccTrain','AccVal','AccTest'])
df_res.drop(columns = 'BestKFeatures').plot()


#############################################################
# Fit without normalization, full hyperparameter optimization 
'''
Much slower to converge than with normalized variables, accuracy decreases
Must present numerical problems. Importance of standarizing data
'''


params = {'C': loguniform(1e0, 1e3),
 'gamma': loguniform(1e-4, 1e-3),
 'kernel': ['linear','sigmoid','rbf']}



svclassifier = SVC(probability=True)

search = RandomizedSearchCV(svclassifier, param_distributions=params, random_state=42, n_iter=10, cv=5, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_trainval, y_trainval_bin)
svm_params = search.best_params_
print(search.best_params_)
# {'C': 15.833718339012057, 'gamma': 0.00011134370364229915, 'kernel': 'rbf'}

svm_model = search.best_estimator_

y_pred_train = svm_model.predict(X_trainval)

print("Accuracy on Train:",metrics.accuracy_score(y_trainval_bin, y_pred_train))

y_pred = svm_model.predict(X_test)
# probabilities of class event
y_pred_proba = svm_model.predict_proba(X_test)[:,0]

print('5-Fold CV Accuracy: ', search.best_score_)

print("Accuracy on Test:",metrics.accuracy_score(y_test_bin, y_pred))


plot_confusion_matrix(svm_model, X_test, y_test_bin,
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


