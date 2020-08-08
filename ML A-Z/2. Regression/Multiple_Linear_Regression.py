#Multiple Linear Regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Read data
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,3]=labelencoder_X.fit_transform(x[:,3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
x=oneHotEncoder.fit_transform(x).toarray()

#Avoiding Dummy variable trap
x = x[:,1:]

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=1/3, random_state=0)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Multiple Linear Regression Model to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int), values=x,axis=1)
X_opt = x[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt = x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt = x[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt = x[:,[0,3,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt = x[:,[0,3]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()