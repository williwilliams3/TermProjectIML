#%reset
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

def data_cleaner(df, hidden):
    data = df.copy()
    data.set_index(["id"],inplace=True)
    data.drop(["date","partlybad"],1,inplace=True)
    # If hidden="N", then create col class2
    if hidden=="N":
        # Set create col class2
        data["class2"] = np.where(data["class4"] == "nonevent", "nonevent","event")
        # Set coll2 to first place
        cols = ['class2'] + [col for col in data if col != 'class2']
        data = data[cols]
    # If hidden==Y, then drop empty col class4
    if hidden=="Y":
        data.drop(["class4"],1,inplace=True)
    return data

def normalize_0to1(df):
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_norm



def split_data(df):
    # Define X and y
    X = df.drop(["class2","class4"], axis=1)
    y = df[["class2", "class4"]]
    # Split df to test 60%, val 20% and test 20%
    # Generate TrainVal 80% and test 20% 
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    # Furthermore split train into Train and Val
    X_train, X_val, y_train, y_val= train_test_split( X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)
    return X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test

# Define folder of the raw data
os.chdir('/Users/bwilliams/GoogleDrive/UniversityOfHelsinki/Fall2020/IML/TeamProject/TermProjectIML')

# Read data
npf_train = pd.read_csv("data/npf_train.csv")
npf_test_hidden = pd.read_csv("data/npf_test_hidden.csv")

# Clean npf_train and add cols class2 and class4
npf_train = data_cleaner(npf_train, "N")
npf_test_hidden = data_cleaner(npf_test_hidden, "Y")

# Create normalized npf_train and npf_test_hidden
scaler = StandardScaler()
scaler.fit(npf_train.iloc[:,2:])

npf_train_norm = npf_train.copy()
npf_train_norm.iloc[:,2:] = scaler.transform(npf_train_norm.iloc[:,2:])

npf_test_hidden_norm = npf_test_hidden.copy()
npf_test_hidden_norm.iloc[:,] = scaler.transform(npf_test_hidden_norm.iloc[:,])

# Split npf_train to train, val and test
X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test = split_data(npf_train)

# Create 0to1 normalized npf_trainval for bestK search
X_trainval_0to1 = normalize_0to1(X_trainval)

# Split npf_train_norm to train, val and test
X_train_norm, X_val_norm, X_trainval_norm, X_test_norm, y_train, y_val, y_trainval, y_test = split_data(npf_train_norm)

# Save all the file to folder cleaned_data
#npf_train.to_csv("cleaned_data/npf_train.csv", index=True)
#npf_test_hidden.to_csv("cleaned_data/npf_test_hidden.csv", index=True)

#npf_train_norm.to_csv("cleaned_data/npf_train_norm.csv", index=True)
#npf_test_hidden_norm.to_csv("cleaned_data/npf_test_hidden_norm.csv", index=True)

#X_trainval.to_csv("cleaned_data/X_trainval.csv", index=True)
#X_trainval_norm.to_csv("cleaned_data/X_trainval_norm.csv", index=True)
#y_trainval.to_csv("cleaned_data/y_trainval.csv", index=True)

#X_test.to_csv("cleaned_data/X_test.csv", index=True)
#X_test_norm.to_csv("cleaned_data/X_test_norm.csv", index=True)
#y_test.to_csv("cleaned_data/y_test.csv", index=True)

# Print sets distribution
names = ["Train set", "Test set", "Validation set", "Train+validation set"]
sets = [y_train, y_test, y_val, y_trainval]
for i in range(len(names)):
    df = sets[i]
    print(names[i])
    print("Len:",len(df))
    print(df["class4"].value_counts(normalize=True),"\n")
    
    
corrmat = npf_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(npf_train[top_corr_features].corr(),annot=False,cmap="RdYlGn")
#plt.savefig('npf_train_correlationmatrix.pdf')  


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


# 1. Gaussian Naive Bayes for Binary Classification (class2)
pca1 = bestK_and_graph(X_train, y_train["class2"], X_val, y_val["class2"], X_test, y_test["class2"], "1. Gaussian Naive Bayes for Binary (class2) classification accurancy with n number of PCA features")

# 2. Gaussian Naive Bayes for Binary Classification (class2) with normalized features
pca2 = bestK_and_graph(X_train_norm, y_train["class2"], X_val_norm, y_val["class2"], X_test_norm, y_test["class2"], "2. Gaussian Naive Bayes for Binary Classification (class2) with normalized features classification accurancy with n number of PCA features")

# 3. Gaussian Naive Bayes for Multiclass Classification (class4)
pca3 = bestK_and_graph(X_train, y_train["class4"], X_val, y_val["class4"], X_test, y_test["class4"], "3. Gaussian Naive Bayes for Multiclass Classification (class4) classification accurancy with n number of PCA features")

# 4. Gaussian Naive Bayes for Multiclass Classification (class4) with normalized feature
pca4 = bestK_and_graph(X_train_norm, y_train["class4"], X_val_norm, y_val["class4"], X_test_norm, y_test["class4"], "4. Gaussian Naive Bayes for Multiclass Classification (class4) with normalized feature classification accurancy with n number of PCA features")


X_train_i = pca1.fit_transform(X_train)
X_test_i = pca1.transform(X_test)
X_val_i = pca1.transform(X_val)

model = GaussianNB()
model.fit(X_train_i, y_train["class2"])

print("Gaussian Naive Bayes binary classification with PCA best features")
print("Train set accurancy:", round(accuracy_score(y_train["class2"], model.predict(X_train_i)),2))
print("Test set accurancy:", round(accuracy_score(y_test["class2"], model.predict(X_test_i)),2))
print("Validation set accurancy:", round(accuracy_score(y_val["class2"], model.predict(X_val_i)), 2))

# Make confusion matrix
ax = plt.axes()
y_pred = model.predict(X_test_i)
confusion_matrix = pd.crosstab(y_test["class2"], y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, ax=ax)
ax.set_title('Test set confusion matrix')
plt.show()