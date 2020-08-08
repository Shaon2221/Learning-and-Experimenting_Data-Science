#Data Pre-Processing

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data
dataset = pd.read_csv('data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Missing values
'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,0]=labelencoder_X.fit_transform(x[:,0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
x=oneHotEncoder.fit_transform(x).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)'''

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''