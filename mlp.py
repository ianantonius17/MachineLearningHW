########################
# Black Box Models (MLP, RF, etc.)
########################

from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

# Read in Data
data = pd.read_csv('mlproject.csv', header=0)

# Extract y
y = data['Result']

# Extract X
X = data.iloc[:, 0:29]

# 80/20 Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Convert everything to numpy arrays
X_train = np.asanyarray(X_train)
X_test = np.asanyarray(X_test)
y_train = np.asanyarray(y_train)
y_test = np.asanyarray(y_test)

# Set Variables Prior to Model Training
predictions = []
solver = 'lbfgs'
activation = 'logistic'

# MLP Single Layer
clf1 = MLPClassifier(activation=activation, solver=solver, hidden_layer_sizes=(29, ), alpha=1e-5, random_state=1)
clf1.fit(X_train, y_train)
results1 = clf1.predict(X_test)
predictions.append(results1)

# MLP Two Layers
clf2 = MLPClassifier(activation=activation, solver=solver, hidden_layer_sizes=(29, 29), alpha=1e-5, random_state=1)
clf2.fit(X_train, y_train)
results2 = clf2.predict(X_test)
predictions.append(results2)

# SVM Linear
svc1 = svm.SVC(kernel='linear', gamma='auto')
svc1.fit(X_train, y_train)
results3 = svc1.predict(X_test)
predictions.append(results3)

# SVM RBF
svc2 = svm.SVC(kernel='rbf', gamma='auto')
svc2.fit(X_train, y_train)
results4 = svc2.predict(X_test)
predictions.append(results4)

# KNN-3
classifier1 = KNeighborsClassifier(n_neighbors=3)
classifier1.fit(X_train, y_train)
results5 = classifier1.predict(X_test)
predictions.append(results5)

# KNN-5
classifier2 = KNeighborsClassifier(n_neighbors=5)
classifier2.fit(X_train, y_train)
results6 = classifier2.predict(X_test)
predictions.append(results6)

# Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
results7 = clf.predict(X_test)
predictions.append(results7)

# Generate Confusion Matrices
cm1 = confusion_matrix(y_test, results1)
cm2 = confusion_matrix(y_test, results2)
cm3 = confusion_matrix(y_test, results3)
cm4 = confusion_matrix(y_test, results4)
cm5 = confusion_matrix(y_test, results5)
cm6 = confusion_matrix(y_test, results6)
cm7 = confusion_matrix(y_test, results7)


# Generate Metrics Function
def metrics(y_pred):
    print('Accuracy')
    print(accuracy_score(y_test, y_pred, normalize=True))
    print('Balanced Accuracy')
    print(balanced_accuracy_score(y_test, y_pred, adjusted=True))
    print('F1')
    print(f1_score(y_test, y_pred))
    print('Precision')
    print(precision_score(y_test, y_pred))
    print('Recall')
    print(recall_score(y_test, y_pred))
    print('AUC')
    print(roc_auc_score(y_test, y_pred))
    print('\n\n')

# Print Metrics
metrics(results1)
metrics(results2)
metrics(results3)
metrics(results4)
metrics(results5)
metrics(results6)
metrics(results7)

########################
# Naive Bayes
########################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
file = 'mlproject.csv'

cols = pd.read_csv(file, nrows=1).columns
df = pd.read_csv(file, usecols=cols[:-1])
Y = pd.read_csv(file, usecols = cols[-1:]).values.ravel()

xtrain, xtest = train_test_split(df, test_size = 0.2, random_state=42)
ytrain, ytest = train_test_split(Y, test_size = 0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xtrain, ytrain)

pred = gnb.predict(xtest)
cm1 = confusion_matrix(ytest, pred)

def metrics(y_pred):
    print('Accuracy')
    print(accuracy_score(ytest, y_pred, normalize=True))
    print('Balanced Accuracy')
    print(balanced_accuracy_score(ytest, y_pred, adjusted=True))
    print('F1')
    print(f1_score(ytest, y_pred))
    print('Precision')
    print(precision_score(ytest, y_pred))
    print('Recall')
    print(recall_score(ytest, y_pred))
    print('AUC')
    print(roc_auc_score(ytest, y_pred))
    print('\n\n')

metrics(pred)

########################
# Bernoulli Naive Bayes
########################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
file = 'mlproject.csv'

cols = pd.read_csv(file, nrows=1).columns
df = pd.read_csv(file, usecols=cols[:-1])
Y = pd.read_csv(file, usecols = cols[-1:]).values.ravel()

xtrain, xtest = train_test_split(df, test_size = 0.2, random_state=42)
ytrain, ytest = train_test_split(Y, test_size = 0.2, random_state=42)

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(xtrain, ytrain)

pred = bnb.predict(xtest)
cm1 = confusion_matrix(ytest, pred)

def metrics(y_pred):
 print('Accuracy')
 print(accuracy_score(ytest, y_pred, normalize=True))
 print('Balanced Accuracy')
 print(balanced_accuracy_score(ytest, y_pred, adjusted=True))
 print('F1')
 print(f1_score(ytest, y_pred))
 print('Precision')
 print(precision_score(ytest, y_pred))
 print(‘Recall’)
 print(recall_score(ytest, y_pred))
 print(‘AUC’)
 print(roc_auc_score(ytest, y_pred))
 print(‘\n\n’)

metrics(pred)

########################
# Logisitic Regression
########################

 # imports
 import numpy as np
 from numpy import genfromtxt
 from sklearn.linear_model import LogisticRegression
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import accuracy_score
 # from sklearn.metrics import balanced_accuracy_score
 from sklearn.metrics import f1_score
 from sklearn.metrics import precision_score
 from sklearn.metrics import recall_score
 from sklearn.metrics import roc_auc_score
 from sklearn.model_selection import train_test_split
 import math

 # build dataset

 # load data from csv
 dataset = genfromtxt("mlproject.csv", delimiter=',')

 # remove feature definitions
 dataset = dataset[1:, ]

 # get labels of dataset
 labels = dataset[:, dataset.shape[1] - 1:]
 # remove label column from dataset
 dataset = dataset[:, :dataset.shape[1] - 1]

 # run logistic regression on the dataset with the percentage of the dataset
 # used for testing denoted by percent_train
 def run_logistic_regression():
     dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
     dataset_train_labels, dataset_test_labels = train_test_split(labels, test_size=0.2, random_state=42)

     # do logistic regression
     lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
     # train
     lr.fit(dataset_train, dataset_train_labels)
     # predict
     predicted_lr = lr.predict(dataset_test)
     # print metrics
     print('Accuracy')
     print(accuracy_score(dataset_test_labels, predicted_lr, normalize=True))
     print('Balanced Accuracy')
     # print(balanced_accuracy_score(dataset_test_labels, predicted_lr, adjusted=True))
     print('F1')
     print(f1_score(dataset_test_labels, predicted_lr))
     print('Precision')
     print(precision_score(dataset_test_labels, predicted_lr))
     print('Recall')
     print(recall_score(dataset_test_labels, predicted_lr))
     print('AUC')
     print(roc_auc_score(dataset_test_labels, predicted_lr))
     print('\n\n')

 run_logistic_regression()
