import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Lasso
from statsmodels.tsa.stattools import adfuller

import warnings


import pandas as pd
num_pred=200

def linear_regresion(y, num_pred=num_pred):
  """
  Perform linear regression on a time series data and predict future values.

  Parameters:
  y (array-like): The input time series data.
  num_pred (int): The number of future time steps to predict. Default is the value of `num_pred`.

  Returns:
  tuple: A tuple containing:
      - list: Predicted future values of the time series.
      - float: Mean squared error of the model on the test set.
  
  Raises:
  ValueError: If the input array `y` is empty or if the dataset is too small to split.
  """
  # Convert y to a numpy array and check if it's not empty
  y = np.array(y)
  if y.size == 0:
      return 0,"The input array y is empty."
  
  # Generate x values as a 2D array with one feature (i.e., time steps)
  x = np.arange(len(y)).reshape(-1, 1)
  
  # Reshape y to be a 1D array for the target variable
  y = y.reshape(-1, 1)
  
  # Check if the dataset is large enough for splitting
  if len(y) <= 1:
      return 0,"The dataset is too small to split."
 
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

def arima(y, order=(2, 1, 2), forecast_steps=num_pred):
  """
  Fit an ARIMA model to the time series data and forecast future values.

  Parameters:
  y (array-like): The input time series data.
  order (tuple): The (p, d, q) order of the ARIMA model. Default is (2, 1, 2).
  forecast_steps (int): The number of future time steps to forecast. Default is the value of `num_pred`.

  Returns:
  tuple: A tuple containing:
      - list: Forecasted values of the time series.
      - float: Mean squared error of the model on the test set.
  
  Raises:
  ValueError: If the input data is not a pandas Series.
  """
  data = pd.Series(y)

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

def random_forest(y,num_pred=num_pred):
	"""
	Train a Random Forest regressor on time series data and predict future values.

	Parameters:
	y (array-like): The input time series data.
	num_pred (int): The number of future time steps to predict. Default is the value of `num_pred`.

	Returns:
	tuple: A tuple containing:
	    - list: Predicted future values of the time series.
	    - float: R-squared score of the model on the test set.

	Raises:
	ValueError: If the input array `y` is empty or if the dataset is too small to split.
	"""
	y = np.array(y)
	if y.size == 0:
	  return 0,"Error"
	  
	# Generate x values as a 2D array with one feature (i.e., time steps)
	x = np.arange(len(y)).reshape(-1, 1)

	# Reshape y to be a 1D array for the target variable
	y = y.reshape(-1, 1)

	# Check if the dataset is large enough for splitting
	if len(y) <= 1:
	    return 0,"The dataset is too small to split."

	# Dividir los datos en conjuntos de entrenamiento y prueba
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	# Crear el modelo Random Forest
	rf = RandomForestRegressor(n_estimators=100, random_state=42)
		# Entrenar el modelo
	rf.fit(X_train, y_train)

	# Realizar predicciones
	y_pred = rf.predict(X_test)
	# Evaluar el rendimiento del modelo

	future_time = np.arange(len(y), len(y) + num_pred).reshape(-1, 1)  # Generate future time steps
	predicted_pm25 = rf.predict(future_time)
	#print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
	#print("R^2 Score:", r2_score(y_test, y_pred))

	return list(predicted_pm25.flatten()), float(r2_score(y_test, y_pred))  # Flatten predicted_pm25 for easy reading

def sarima(x, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), train_size=0.8, forecast_steps=num_pred):
    """
    Fit a SARIMA model and forecast future values.
    
    Parameters:
        time_series (pd.Series): The time series data.
        order (tuple): The (p,d,q) order of the ARIMA model.
        seasonal_order (tuple): The (P,D,Q,s) order of the seasonal component.
        train_size (float): The proportion of data to use for training.
        forecast_steps (int): The number of steps to forecast into the future.
    
    Returns:
        pd.Series: The forecasted values.
        pd.Series: The actual test values.
        float: The mean squared error of the forecast.
    """
    time_series=pd.Series(x)
    # Split the data into training and testing sets
    train_size = int(len(time_series) * train_size)
    train, test = time_series[:train_size], time_series[train_size:]

    # Fit the SARIMA model
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast the future values
    forecast = model_fit.forecast(steps=len(test) + forecast_steps)

    # Calculate the mean squared error
    mse = mean_squared_error(test, forecast[:len(test)])

    return forecast, mse

def lasso(y, num_pred=num_pred, alpha=1.0):
    """
    Perform Lasso regression and return predictions and Mean Squared Error (MSE).
    
    Parameters:
    -----------
    X : array-like
        Feature matrix.
        
    y : array-like
        Target variable.
        
    num_pred : int, optional
        Number of predictors (currently unused). Default is num_pred.
        
    alpha : float, optional
        Regularization strength. Default is 1.0.
        
    Returns:
    --------
    list : Predicted values.
    float : Mean Squared Error (MSE).
    """
    y = np.array(y)
    if y.size == 0:
      return 0,"Error"
      
    # Generate x values as a 2D array with one feature (i.e., time steps)
    x = np.arange(len(y)).reshape(-1, 1)

    # Reshape y to be a 1D array for the target variable
    y = y.reshape(-1, 1)

    # Check if the dataset is large enough for splitting
    if len(y) <= 1:
        return 0, "The dataset is too small to split."
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Initialize Lasso model with the specified alpha (regularization strength)
    model = Lasso(alpha=alpha)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict the target variable on the test data
    predicted_pm25 = model.predict(X_test)
    
    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predicted_pm25)
    
    # Return the predictions and the MSE
    return list(predicted_pm25.flatten()), float(mse)