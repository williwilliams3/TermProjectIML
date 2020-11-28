# TermProjectIML
 Fall 2020 Term Project
 
 Objective:
 Train a ML model on npf_train.csv
 Generate the following predictions for npf_test_hidden.csv dataset at most on December 6. 
 
  	1. Binary class predictions
	2. Multi-class predictions
	3. Prediction of our accuracy on the test set.


SplitData.py

	- Stratified sample npf_train.csv randomly into train.csv 60%, validation.csv 20% and test.csv 20% with respect to variable class4
	- Also generates datasets into memory, wiht binary response and multiclass response 
	- Uses functions data_generator(df), convert_binary(y)
	- If the directory is set to where the file is can be run by from SplitData import * 

	
def data_generator(df):
   
    input:
        df: dataframe to split stratified by variable class
        class: string with variable to use as response and as variable to stratify by
    output:
        X_train, y_train: variables and response 60% data 
        X_val, y_val: variables and response 20% data 
        X_trainval, y_trainval: variables and response 80% data (for cross validation)
        X_test, y_test: variables and response 20% data 
    

def convert_binary(y):
    
    Parameters
    ----------
    y : repsonse variable with 4 classes

    Returns
    -------
    y : reponse variable 2 classes

    
    y_bin = np.where(y == "nonevent", "nonevent","event")
    return y_bin


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
	- Fits Decision tree classifier over Class4 using 10-Fold Cross Validation over train+validation set 
	- Computes unbiased accuracy and AUC on test set
	
	
### Accuracies for the binary class models	
	
| Accuracy   | DT Binary  | RF Binary   | XGB Binary   | KNN Binary   | Log Binary   |
|------------|------------|-------------|-------------|-------------|-------------|
| Training   |        89% |        100% |        100% |        89.6% |        88%    | 
| Validation |        83% |        86%  |        89%  |        86.5%  |        82%    |  
| Test       |        81% |        83%  |        88%  |        84.6%  |        87%    |

### Accuracies for the multi-classification models

| Accuracy   | DT Multiclass | RF Multiclass | XGB Multiclass | KNN Multiclass |
|------------|---------------|---------------|----------------|----------------|
| Training   |        66%    |        100%   |        100%    |        66%    |
| Validation |        61%    |        61%    |        67%     |        57.7%  |
| Test       |        60%    |        67%    |        62%     |        57.7%  |
