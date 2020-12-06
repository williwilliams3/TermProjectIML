#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:27:38 2020

@author: bwilliams

Predictions on npf_test.csv

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy.random as npr
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


def data_cleaner(df):
    data = df.copy()
    #remove the id, partlybad columns
    data.drop(["id","partlybad"],1,inplace=True)
    #set new index using the date column
    data.set_index(["date"],inplace=True)
    #add binary 'class2' col for event/non-event
    data["class2"] = np.where(data["class4"] == "nonevent", "nonevent","event")
    #arrange to put class2 col at the front
    cols = ['class2'] + [col for col in data if col != 'class2']
    data = data[cols]
    return data



def convert_binary(y):
    '''
    Parameters
    ----------
    y : repsonse variable with 4 classes

    Returns
    -------
    y : reponse variable 2 classes

    '''
    y_bin = np.where(y == "nonevent", "nonevent","event")
    return y_bin


def binary_predictions_fulldata():
    '''
    This function returns probabilities of blend of 3 models. 
    This will be used to compute perplexity over npf_hidden
    Maybe it will be better to use XGB as it shows smaller perplexity
    '''
    
    # 0. Full data preparation
    
    url = "https://raw.githubusercontent.com/williwilliams3/TermProjectIML/master/data/npf_train.csv"
    df = pd.read_csv(url)
    df = data_cleaner(df)
    
    X_train = df.drop(["class2","class4"], axis=1)
    y_train = df['class4']
    y_train_bin = convert_binary(y_train)

    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    
    url2 = 'https://raw.githubusercontent.com/williwilliams3/TermProjectIML/master/data/npf_test_hidden.csv'
    df2 = pd.read_csv(url2)
    df2 = df2.drop(["id","partlybad",'date'],axis = 1)
    
    X_test = df2.drop(["class4"], axis=1)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    # 1. SVM Binary 
 
    # svm_model = SVC(C= 810.2342569581549, gamma= 0.0005183165833793786, kernel= 'rbf', probability=True)
    # svm_model.fit(X_train_norm, y_train_bin)
    
    # y_pred_train_proba_svm = svm_model.predict_proba(X_train_norm)[:,0]
    # y_pred_test_proba_svm = svm_model.predict_proba(X_test_norm)[:,0]
    
    # print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_svm>0.5))
    
    # 2. XGB Binary 

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
                                  colsample_bytree= 0.8854654189948783, gamma=0.05056133806139512,
                                  learning_rate= 0.05523204183449923, max_depth= 2, 
                                  n_estimators= 267, subsample= 0.6291052025456774)
    
    # {'colsample_bytree': 0.8854654189948783, 'gamma': 0.05056133806139512, 'learning_rate': 0.05523204183449923, 'max_depth': 2, 'n_estimators': 267, 'subsample': 0.6291052025456774}
    xgb_model.fit(X_train, y_train_bin)
    y_pred_train_proba_xgb = xgb_model.predict_proba(X_train)[:,0]
    y_pred_test_proba_xgb = xgb_model.predict_proba(X_test)[:,0]
    
    print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_xgb>0.5))
    print('Perplexity on train set: ', np.exp(-np.mean(np.log(np.where(y_train_bin=='event',y_pred_train_proba_xgb,1-y_pred_train_proba_xgb )))))
    
    
    
    # 3. Log Binary 
    
    npr.seed(42)
    logistic_regression= LogisticRegression()
    logistic_regression.fit(X_train_norm, y_train_bin)
    y_pred_train_proba_log = logistic_regression.predict_proba(X_train_norm)[:,0]
    y_pred_test_proba_log = logistic_regression.predict_proba(X_test_norm)[:,0]
    
    print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_log>0.5))
    print('Perplexity on train set: ', np.exp(-np.mean(np.log(np.where(y_train_bin=='event',y_pred_train_proba_log,1-y_pred_train_proba_log )))))
    
    
    # 4. NB Binary
    
    pca1 = PCA(n_components=12)
    X_train_i = pca1.fit_transform(X_train)
    X_test_i = pca1.transform(X_test)
    
    model = GaussianNB()
    model.fit(X_train_i, y_train_bin)
    y_pred_train_proba_nb = model.predict_proba(X_train_i)[:,0]
    y_pred_test_proba_nb = model.predict_proba(X_test_i)[:,0]
    
    print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_nb>0.5))
    print('Perplexity on train set: ', np.exp(-np.mean(np.log(np.where(y_train_bin=='event',y_pred_train_proba_nb,1-y_pred_train_proba_nb )))))
    
    # Blend of Models 
    
    df_train_bin = pd.DataFrame(dict( xgb = y_pred_train_proba_xgb, log = y_pred_train_proba_log, nb = y_pred_train_proba_nb))
    df_test_bin = pd.DataFrame(dict( xgb = y_pred_test_proba_xgb, log = y_pred_test_proba_log, nb = y_pred_test_proba_nb))
    
    y_pred_train_proba_blend = df_train_bin.mean(axis = 1)
    y_pred_test_proba_blend = df_test_bin.mean(axis = 1)
    
    print('Accuracy on train set: ',metrics.accuracy_score(y_train_bin=='event', y_pred_train_proba_blend>0.5))
     
    
    
    print('Perpelxity on train set: ', np.exp(-np.mean(np.log(np.where(y_train_bin=='event',y_pred_train_proba_blend,1-y_pred_train_proba_blend )))))
    return y_pred_test_proba_blend


def multiclass_predictions_fulldata():
    '''
    This function returns probabilities of blend of 3 models. 
    This will be used to compute perplexity over npf_hidden
    Maybe it will be better to use XGB as it shows smaller perplexity
    '''
    
    # 0. Full data preparation
    
    url = "https://raw.githubusercontent.com/williwilliams3/TermProjectIML/master/data/npf_train.csv"
    df = pd.read_csv(url)
    df = data_cleaner(df)
    
    X_train = df.drop(["class2","class4"], axis=1)
    y_train = df['class4']
    y_train_bin = convert_binary(y_train)

    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    
    url2 = 'https://raw.githubusercontent.com/williwilliams3/TermProjectIML/master/data/npf_test_hidden.csv'
    df2 = pd.read_csv(url2)
    df2 = df2.drop(["id","partlybad",'date'],axis = 1)
    
    X_test = df2.drop(["class4"], axis=1)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    # 1. SVM MultiClass 
 
    svm_model = SVC(**{'C': 109.53031576544694, 'gamma': 0.0005494254346819604, 'kernel': 'rbf'}, probability=True)
    svm_model.fit(X_train_norm, y_train)
    
    y_pred_train_proba_svm = svm_model.predict_proba(X_train_norm)
    y_pred_test_proba_svm = svm_model.predict_proba(X_test_norm)
    
    print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_svm, axis = 1)) )
    print('Binary Accuracy train: ', metrics.accuracy_score(y_train == 'nonevent', np.argmax(y_pred_train_proba_svm, axis = 1)==3 ))
    
    # 2. XGB MultiClass 
    #{'colsample_bytree': 0.9915346248162882, 'gamma': 0.4812236474710556, 'learning_rate': 0.10553468874760924, 'max_depth': 3, 'n_estimators': 212, 'subsample': 0.6592347719813599}
    xgb_model =  xgb.XGBClassifier(**{'colsample_bytree': 0.9915346248162882, 'gamma': 0.4812236474710556, 'learning_rate': 0.10553468874760924, 'max_depth': 3, 'n_estimators': 212, 'subsample': 0.6592347719813599}, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_train_proba_xgb = xgb_model.predict_proba(X_train)
    y_pred_test_proba_xgb = xgb_model.predict_proba(X_test)
    
    print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_xgb, axis = 1)) )
    print('Binary Accuracy train: ', metrics.accuracy_score(y_train == 'nonevent', np.argmax(y_pred_train_proba_xgb, axis = 1)==3 ))
    
    
    # 3. NB MultiClass
    
    pca1 = PCA(n_components=14)
    X_train_i = pca1.fit_transform(X_train)
    X_test_i = pca1.transform(X_test)
    
    model = GaussianNB()
    model.fit(X_train_i, y_train)
    y_pred_train_proba_nb = model.predict_proba(X_train_i)
    y_pred_test_proba_nb = model.predict_proba(X_test_i)
    
    print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_nb, axis = 1)) )
    print('Binary Accuracy train: ', metrics.accuracy_score(y_train == 'nonevent', np.argmax(y_pred_train_proba_nb, axis = 1)==3 ))
    
    # Blend of Models 

    df_train_bin = pd.DataFrame(np.c_[y_pred_train_proba_svm, y_pred_train_proba_xgb, y_pred_train_proba_nb])
    df_test_bin = pd.DataFrame(np.c_[y_pred_test_proba_svm, y_pred_test_proba_xgb, y_pred_test_proba_nb])
        
    y_pred_train_proba_blend = np.c_[df_train_bin[[0,4,8]].mean(axis = 1), df_train_bin[[1,5,9]].mean(axis = 1), df_train_bin[[2,6,10]].mean(axis = 1) , df_train_bin[[3,7,11]].mean(axis = 1)]
    y_pred_test_proba_blend = np.c_[df_test_bin[[0,4,8]].mean(axis = 1), df_test_bin[[1,5,9]].mean(axis = 1), df_test_bin[[2,6,10]].mean(axis = 1) , df_test_bin[[3,7,11]].mean(axis = 1)]
    
    print('Accuracy on train set: ', metrics.accuracy_score(pd.factorize(y_train, sort=True)[0], np.argmax(y_pred_train_proba_nb, axis = 1)) )
    print('Binary Accuracy train: ', metrics.accuracy_score(y_train == 'nonevent', np.argmax(y_pred_train_proba_nb, axis = 1)==3 ))
    
    return y_pred_test_proba_blend


p = binary_predictions_fulldata()
y_pred_test_proba_blend = multiclass_predictions_fulldata()




y_pred_test_proba_blend_int = np.argmax(y_pred_test_proba_blend, axis = 1)
levels = {0:'II',1:'Ia',2:'Ib',3:'nonevent'}
class4 = np.vectorize(levels.get)(y_pred_test_proba_blend_int)

#  Estimate number of events in test
p_nonevent = np.sum(y_pred_test_proba_blend_int==3)/len(y_pred_test_proba_blend_int)

# Accuracy with umblanced classes
# Following are class specific accuracies obtained in test of multiclass selected model
acc_nonevent = 0.9767441860465116
acc_event = 0.8372093023255814

df = pd.DataFrame(columns = [0,1], )
acc = p_nonevent*acc_nonevent + (1-p_nonevent)*acc_event

df.loc[0] = acc
df.loc[1] =['class4', 'p']
df0 = pd.DataFrame(np.c_[ class4, p], )

df = df.append(df0)

# By hand delete second entry of first row for it to be valid submission
# I couldnt find a way to automate it easily
df.to_csv('answers.csv', index=False, header=False)

