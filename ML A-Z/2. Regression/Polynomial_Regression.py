#Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Splitting data
'''from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=1/3, random_state=0)
'''
#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Fitting Polynomial Regression to the dataset.
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree=4)
x_poly = Poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualising Linear Regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial Regression results
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg2.predict(Poly_reg.fit_transform(x_grid)),color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting on Linear Regression
lin_reg.predict(6.5)

#Predicting on Polynomial Regression
lin_reg2.predict(Poly_reg.fit_transform(6.5))