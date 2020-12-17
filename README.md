# Predicting new particle formation events with Machine Learning

#### The course this project is done for:<br/>
University of Helsinki <br/>
Introduction to Machine Learning<br/>
Fall 2020 Term Project<br/>

#### Project members: <br/>
Bernardo Williams (GitHub: williwilliams3) <br/>
Julia Sanders (GitHub: julia-sand) <br/>
Mikko Saukkoriipi (GitHub: Saukkoriipi) <br/>

#### Where to find full project report: <br/>
Full project report with all the details can be found from: Project_report_and_presentation/Project_report.pdf
 
#### Objective: 
To create the best possible machine learning model to predict potential new particle formation events based on the 100 daily features.

#### Abstract of project:
Several machine learning classification models were used over the dataset npf_train.csv divided with the purpose of predicting a binary and a multi-class label. The objective was to extend the model and predictions to unseen data, and also to give an estimate of the accuracy the model would have on the unseen data.

For fitting the models we used two data reduction techniques, PCA and bestK feature selection and two normalization methods, min-max and standardizing normalization. We tried fitting algorithmic, generative and discriminative methods using either validation or cross validation to measure accuracy for both the binary and multiclass classifiers and found which ones performed the best in terms of accuracy over an unbiased test set. Lastly, we found that taking the average prediction of the best algorithmic, discriminative and generative methods gave estimates with higher accuracy and more consistent accuracy over train, validation and test.
	
#### Final accuracies for the binary class models	
	
| Accuracy   | DT Binary  | RF Binary   | XGB Binary  | KNN Binary | Log Binary  | NB bestK    | NB PCA  | SVM         | Ensamble    | 
|------------|------------|-------------|-------------|------------|-------------|-------------|---------|-------------|-------------|
| Training   |        88% |        100% |        100% |       85%  |        86%  |        81%  |     84% |        98%  |        96%  | 
| Validation |        84% |        87%  |        90%  |       78%  |        85%  |        85%  |     87% |        90%  |        96%  | 
| Test       |        88% |        88%  |        87%  |       80%  |        85%  |        87%  |     93% |        83%  |        92%  | 

#### Final accuracies for the multi-classification models

| Accuracy   | DT Multiclass | RF Multiclass | XGB Multiclass | KNN Multiclass | NB bestK    | NB PCA      | SVM         | Ensamble    | 
|------------|---------------|---------------|----------------|----------------|-------------|-------------|-------------|-------------|
| Training   |        66%    |        100%   |        100%    |        66%     |        62%  |        69%  |        83%  |        94%  | 
| Validation |        64%    |        66%    |        70%     |        57.7%   |        64%  |        62%  |        69%  |        98%  |
| Test       |        67%    |        72%    |        70%     |        57.7%   |        62%  |        65%  |        68%  |        70%  |
