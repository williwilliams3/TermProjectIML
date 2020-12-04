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

from sklearn.utils.fixes import loguniform

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

svm_model = SVC(C= 810.2342569581549, gamma= 0.0005183165833793786, kernel= 'rbf', probability=True)
svm_model.fit(X_trainval_norm, y_trainval)
y_pred_train = svm_model.predict(X_trainval_norm)
y_pred = svm_model.predict(X_test_norm)

y_pred_train_proba = svm_model.predict_proba(X_trainval_norm)[:,0]
y_pred_proba = svm_model.predict_proba(X_test_norm)[:,0]

print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin, y_pred_train))
print('Accuracy on test set: ',metrics.accuracy_score(y_test_bin, y_pred))


# 2. XGB Binary 
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
import xgboost as xgb
from scipy.stats import uniform, randint

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
                              colsample_bytree= 0.8854654189948783, gamma=0.05056133806139512,
                              learning_rate= 0.05523204183449923, max_depth= 2, 
                              n_estimators= 267, subsample= 0.6291052025456774)

# {'colsample_bytree': 0.8854654189948783, 'gamma': 0.05056133806139512, 'learning_rate': 0.05523204183449923, 'max_depth': 2, 'n_estimators': 267, 'subsample': 0.6291052025456774}

search.fit(X_trainval, y_trainval_bin)
y_pred_train = xgb_model.predict(X_trainval)
y_pred = xgb_model.predict(X_test)

y_pred_train_proba = xgb_model.predict_proba(X_trainval)[:,0]
y_pred_proba = xgb_model.predict_proba(X_test)[:,0]

print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin, y_pred_train))
print('Accuracy on test set: ',metrics.accuracy_score(y_test_bin, y_pred))




# 3. Log Binary 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy.random as npr
# Accuracy on test set:  0.8846153846153846
# Accuracy on validation set:  0.8461538461538461
# Accuracy on train set:  0.9415584415584416
npr.seed(42)
logistic_regression= LogisticRegression()
logistic_regression.fit(X_trainval_norm, y_train_bin)

y_pred_train = logistic_regression.predict(X_trainval_norm)
y_pred = logistic_regression.predict(X_test_norm)

y_pred_train_proba = logistic_regression.predict_proba(X_trainval_norm)[:,0]
y_pred_proba = logistic_regression.predict_proba(X_test_norm)[:,0]

print('Accuracy on train set: ', metrics.accuracy_score(y_train_bin, y_pred_train))
print('Accuracy on test set: ', metrics.accuracy_score(y_test_bin, y_pred))


# 4. NB Binary

# 1. Gaussian Naive Bayes for Binary Classification (class2)
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import preprocessing 
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
def bestK_and_graph(X_train, y_train, X_val, y_val, X_test, y_test, title):
    len_cols = len(X_train.columns)+1
    acc_train = []
    acc_test = []
    acc_val = []

    for i in range(1, len_cols):
        
        # Do PCA
        pca = PCA(n_components=i)
        X_train_i = pca.fit_transform(X_train)
        X_test_i = pca.transform(X_test)
        X_val_i = pca.transform(X_val)
        
        # Train Gaussian Naive Bayes model
        model = GaussianNB()
        model.fit(X_train_i, y_train)
    
        # Save accurancy to list
        acc_train.append(accuracy_score(y_train, model.predict(X_train_i)))
        acc_test.append(accuracy_score(y_test, model.predict(X_test_i)))
        acc_val.append(accuracy_score(y_val, model.predict(X_val_i)))
    
    # Get max value
    max_val = max(acc_val)
    
    # Loop max values and select largest index with max value
    acc_val_max = 0
    acc_val_max_index = 0
    for i in range(len(acc_val)):
        if acc_val[i] >= acc_val_max:
            acc_val_max = acc_val[i]
            acc_val_max_index = i
    PCA_max = acc_val_max_index+1
    
    # Make PCA model with n_components that maximizes validation accuracy
    pca = PCA(n_components=PCA_max)

    # Plot the results
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    x = range(1, len_cols)
    plt.plot(x, acc_train, color="black")
    plt.plot(x, acc_test, color="blue")
    plt.plot(x, acc_val, color="green")
    # Vertical line to points with max values
    plt.axvline(PCA_max, color='pink', linestyle='-')
    plt.title(title)
    plt.xlabel('Number of features')
    plt.ylabel('Accurancy')
    plt.legend(["train", "test", "val"], loc ="lower right") 
    plt.show()
    
    return pca
pca1 = bestK_and_graph(X_train, y_train_bin, X_val, y_val_bin, X_test, y_test_bin, "1. Gaussian Naive Bayes for Binary (class2) classification accurancy with n number of PCA features")

X_train_i = pca1.fit_transform(X_train)
X_test_i = pca1.transform(X_test)
X_val_i = pca1.transform(X_val)

model = GaussianNB()
model.fit(X_train_i, y_train_bin)

print("Gaussian Naive Bayes binary classification with PCA best features")
print("Train set accurancy:", round(accuracy_score(y_train_bin, model.predict(X_train_i)),2))
print("Test set accurancy:", round(accuracy_score(y_test_bin, model.predict(X_test_i)),2))
print("Validation set accurancy:", round(accuracy_score(y_val_bin, model.predict(X_val_i)), 2))

