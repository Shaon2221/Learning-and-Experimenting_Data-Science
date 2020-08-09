# Recurrent Neural Network

# Part -1 : Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the training set
train_df = pd.read_csv('/home/mayank/Documents/Projects/Machine-Learning/Part 7 - Neural Networks/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
train = train_df.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_scaled = sc.fit_transform(train)

# Create a data structures with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(train_scaled[i-60:i,0])
    y_train.append(train_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))



# Part -2 : Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(rate=0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(rate=0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(x_train,y_train, epochs=100, batch_size=32)

# Part -3 : Making the predictions and visualising the results

# Getting the real stock price of 2017
test = pd.read_csv('/home/mayank/Documents/Projects/Machine-Learning/Part 7 - Neural Networks/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv')
real_stock_price = test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
total_data = pd.concat((train_df['Open'], test['Open']),axis=0)
inputs = total_data[len(total_data)-len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
pred_stock_price = regressor.predict(x_test)
pred_stock_price = sc.inverse_transform(pred_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label = 'Real Google Stock Price')
plt.plot(pred_stock_price, color='blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()