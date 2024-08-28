import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

import pandas as pd


def linear_regresion(y, num_pred=100):
  # Convert y to a numpy array and check if it's not empty
  y = np.array(y)
  if y.size == 0:
      raise ValueError("The input array y is empty.")
  
  # Generate x values as a 2D array with one feature (i.e., time steps)
  x = np.arange(len(y)).reshape(-1, 1)
  
  # Reshape y to be a 1D array for the target variable
  y = y.reshape(-1, 1)
  
  # Check if the dataset is large enough for splitting
  if len(y) <= 1:
      raise ValueError("The dataset is too small to split.")
  
  # Split data into training and test sets
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  
  # Initialize and fit the model
  model = LinearRegression()
  model.fit(x_train, y_train)
  
  # Predict on the test set
  y_pred = model.predict(x_test)
  mse = mean_squared_error(y_test, y_pred)

  # Predict future values
  future_time = np.arange(len(y), len(y) + num_pred).reshape(-1, 1)  # Generate future time steps
  predicted_pm25 = model.predict(future_time)
  
  return list(predicted_pm25.flatten()), float(mse)  # Flatten predicted_pm25 for easy reading

def arima(y, order=(5, 1, 0), forecast_steps=100):
  data = pd.Series(y)
  if not isinstance(data, pd.Series):
      raise ValueError("Data should be a pandas Series.")

  # Fit the ARIMA model
  model = ARIMA(data, order=order)
  model_fit = model.fit()

  # Forecast the next values
  forecast = model_fit.forecast(steps=forecast_steps)
  
  # To compute the MSE, we need to have the true values; assuming we have a dataset to compare with
  # For this example, we'll assume the last part of the data is held out for testing
  train_size = len(data) - forecast_steps
  train_data = data[:train_size]
  test_data = data[train_size:]
  
  # Fit the model again on the training data and forecast
  model_test = ARIMA(train_data, order=order)
  model_test_fit = model_test.fit()
  predictions = model_test_fit.forecast(steps=forecast_steps)
  
  # Compute the Mean Squared Error
  mse = mean_squared_error(test_data, predictions[:len(test_data)])
  
  return list(forecast), float(mse)
