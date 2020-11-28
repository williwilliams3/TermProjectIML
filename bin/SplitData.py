#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:11:36 2020

@author: bwilliams
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

##### New way to generate the data

def data_generator(df):
    '''
    input:
        df: dataframe to split stratified by variable class
        class: string with variable to use as response and as variable to stratify by
    output:
        X_train, y_train: variables and response 60% data 
        X_val, y_val: variables and response 20% data 
        X_trainval, y_trainval: variables and response 80% data (for cross validation)
        X_test, y_test: variables and response 20% data 
    '''
    
    X = df.drop(["class2","class4"], axis=1)
    y = df['class4']
    
    # 60%, 20%, 20%
    # Generate TrainVal 80% and test 20% 
    X_trainval, X_test, y_trainval, y_test = train_test_split( X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Furthermore split train into Train and Val
    X_train, X_val, y_train, y_val= train_test_split( X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)
    
    return X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test


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

url = "https://raw.githubusercontent.com/williwilliams3/TermProjectIML/master/data/train.csv"
df = pd.read_csv(url)
df = data_cleaner(df)

X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test = data_generator(df)

# If needed conversion from 4 classes to binary classes
y_train_bin = convert_binary(y_train)
y_val_bin = convert_binary(y_val)
y_trainval_bin = convert_binary(y_trainval)
y_test_bin = convert_binary(y_test)

'''
Count the classes by group
y_trainval.value_counts(normalize=True)
y_test.value_counts(normalize=True)
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
'''


'''
import os
os.chdir('/Users/bwilliams/GoogleDrive/UniversityOfHelsinki/Fall2020/IML/TeamProject/TermProjectIML/')

df = pd.read_csv('data/npf_train.csv')

train, validation, test = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.6*len(df)), int(.8*len(df))])
              
train.to_csv('data/train.csv', index=False)
validation.to_csv('data/validation.csv', index=False)
test.to_csv('data/test.csv', index=False)




train = data_cleaner(train)
validation = data_cleaner(validation)
test = data_cleaner(test)



train['class2'].value_counts(normalize=True)
validation['class2'].value_counts(normalize=True)
test['class2'].value_counts(normalize=True)

train['class4'].value_counts(normalize=True)
validation['class4'].value_counts(normalize=True)
test['class4'].value_counts(normalize=True)
'''