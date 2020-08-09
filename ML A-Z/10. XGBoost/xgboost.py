# XGBoost

# 1. Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')

# Importing the dataset
dataset = pd.read_csv('/home/mayank/Documents/Projects/Machine-Learning/Part 10 - XGBoost/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2 = LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])

oneHotEncoder = OneHotEncoder(categorical_features=[1])
X=oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

#Predicting the result
y_pred = classifier.predict(X_test)

#Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
mean = accuracies.mean()
std = accuracies.std()

