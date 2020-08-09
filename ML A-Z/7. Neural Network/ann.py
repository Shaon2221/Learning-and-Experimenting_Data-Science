#Artificial Neural Network

# 1. Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')

# Importing the dataset
dataset = pd.read_csv('/home/mayank/Documents/Projects/Machine-Learning/Part 7 - Neural Networks/Artifical Neural Network/Churn_Modelling.csv')

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


# 2. Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising ANN
classifier = Sequential()

# Add one input layer and one hidden layer with Dropout
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
classifier.add(Dropout(p=0.1))

# Add second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))

# output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Training set
classifier.fit(X_train,y_train, batch_size=10, nb_epoch=100)

# Making the predictions and evaluating the model

# Predict the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Calculating accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)

# 3. Evaluating. Improving and Tuning the ANN

# Evaluating the ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()

    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))

    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()

    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))

    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,32],'nb_epoch':[100,500],'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_