#Simple Linear Regression

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=1/3, random_state=0)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predict the test set results
y_pred = regressor.predict(X_test)

#Visualising the training set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()