# TermProjectIML
 Fall 2020 Term Project
 
 Objective:
 Train a ML model on npf_train.csv
 Generate the following predictions for npf_test_hidden.csv dataset at most on December 6. 
 
  	1. Binary class predictions
	2. Multi-class predictions
	3. Prediction of our accuracy on the test set.


SplitData.py

	- Splits npf_train.csv randomly into train.csv 60%, validation.csv 20% and test.csv 20%

data_cleaner.py 

To set up the raw data file for use, following the steps from Exercises 1. 

Imports: numpy, pandas 

Input: npf data set 
- Removes ID and partlybad columns
- sets new index as the date
- adds binary 'class2' column, separating event and non-event days
- arranges columns to place 'class2' at the start of the data frame. 

Output: npf data set with changes made as above. 

(Modified data_clear.py to work on a copy of data set and not make changes on original dataset)

DecisionTreeFit_CV.py
	
	- Fits Decision tree classifier over binary Class2 using 10-Fold Cross Validation over train+validation set 
	- Computes unbiased accuracy and AUC on test set
