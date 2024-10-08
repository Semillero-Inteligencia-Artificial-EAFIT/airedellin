#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airellin - by MLeafit
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


import warnings

import pandas as pd
num_pred=2000

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
	
def Poly_regresion_with_lag(y, degree=2, n_lags=2, num_pred=num_pred):
    """
    Perform Polynomial regression on a time series data and predict future values using lag features. 

    Parameters:
    y (array-like): The input time series data.
    degree (int): Degree of polynomial features.
    n_lags (int): Number of lag features used for prediction.
    num_pred (int): Number of future time steps to predict.

    Returns:
    tuple: A tuple containing:
        - list: Predicted future values of the time series.
        - float: Mean squared error of the model on the test set.
    
    Raises:
    ValueError: If the input array `y` is empty or if the dataset is too small to split.
    """
    # Function that adds the lag columns to the DataFrame
    def lag_features_df(y, n_lags):
        df = pd.DataFrame(y, columns=["y"])
        for lag in range(1, n_lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df.dropna(inplace=True)  # Drop rows with NaN values
        return df
    
    # Convert y to a numpy array and check if it's not empty
    y = np.array(y)
    if y.size == 0:
        return [], "The input array y is empty."
    
    # Generate lagged features and target variable
    lagged_df = lag_features_df(y, n_lags=n_lags)
    X_lags = lagged_df.drop(columns=["y"]).values
    y_lags = lagged_df["y"].values

    # Generate the time steps for the lagged data
    X_time = np.arange(n_lags, len(y)).reshape(-1, 1)
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_time)
    
    # Combine polynomial features with lag features
    X_combined = np.hstack([X_poly, X_lags])

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X_combined, y_lags, test_size=0.2, random_state=42)
    
    # Initialize and fit the model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    # Predict future values
    future_time = np.arange(len(y), len(y) + num_pred).reshape(-1, 1)  # Generate future time steps
    future_time_poly = poly.transform(future_time)  # Apply polynomial transformation to future time steps
    
    # For future predictions, use the last n_lags values from y as lagged features
    last_known_values = y[-n_lags:]
    
    future_predictions = []
    for i in range(num_pred):
        # Create input for the model by combining future time and lagged values
        X_future = np.hstack([future_time_poly[i].reshape(1, -1), last_known_values.reshape(1, -1)])
        future_value = model.predict(X_future)[0]
        future_predictions.append(future_value)
        
        # Update last_known_values with the predicted value to generate further predictions
        last_known_values = np.roll(last_known_values, -1)
        last_known_values[-1] = future_value

    return list(future_predictions), float(mse)
	
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

def xgboost(y, num_pred=num_pred):
    """
    Train an XGBoost regressor on time series data and predict future values.

    Parameters:
    y (array-like): The input time series data.
    num_pred (int): The number of future time steps to predict. Default is 10.

    Returns:
    tuple: A tuple containing:
        - list: Predicted future values of the time series.
        - float: R-squared score of the model on the test set.

    Raises:
    ValueError: If the input array `y` is empty or if the dataset is too small to split.
    """
    y = np.array(y)
    if y.size == 0:
        return 0, "Error"

    # Generate x values as a 2D array with one feature (i.e., time steps)
    x = np.arange(len(y)).reshape(-1, 1)

    # Reshape y to be a 1D array for the target variable
    y = y.reshape(-1, 1)

    # Check if the dataset is large enough for splitting
    if len(y) <= 1:
        return 0, "The dataset is too small to split."

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost regressor
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train.ravel())

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the R-squared score
    r2 = r2_score(y_test, y_pred)

    # Predict future values
    future_x = np.arange(len(y), len(y) + num_pred).reshape(-1, 1)
    future_pred = model.predict(future_x)

    return future_pred.tolist(), r2

def exponential_smoothing(y, num_pred=num_pred):
    """
    Apply Exponential Smoothing to time series data and predict future values.

    Parameters:
    y (array-like): The input time series data.
    num_pred (int): The number of future time steps to predict. Default is 10.

    Returns:
    tuple: A tuple containing:
        - list: Predicted future values of the time series.
        - float: R-squared score of the model on the training set.

    Raises:
    ValueError: If the input array `y` is empty or if the dataset is too small to model.
    """
    y = np.array(y)
    if y.size == 0:
        return 0, "Error"

    # Check if the dataset is large enough for modeling
    if len(y) <= 1:
        return 0, "The dataset is too small to model."

    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(y, trend=None, seasonal=None, initialization_method="estimated")
    fitted_model = model.fit()

    # Calculate the R-squared score on the training data
    y_pred = fitted_model.fittedvalues
    r2 = r2_score(y, y_pred)

    # Predict future values
    future_pred = fitted_model.forecast(steps=num_pred)

    return future_pred.tolist(), r2


def LSTM(y, num_pred=100):
    """
    Train a RNN with LSTM to time series data and predict future values.

    Parameters:
    y (array-like): The input time series data.
    num_pred (int): The number of future time steps to predict. Default is 10.

    Returns:
    tuple: A tuple containing:
        - list: Predicted future values of the time series.
        - float: R-squared score of the model on the training set.

    Raises:
    ValueError: If the input array `y` is empty or if the dataset is too small to model.
    """


    y = np.array(y)

    if y.size == 0:
        return 0, "Error"

    window_size = 100
    data = y.reshape(-1, 1)
    dataset = TimeseriesGenerator(data, data, length=window_size, batch_size=64)
  

    model = Sequential([

        # First Bidirectional LSTM layer with 42 units and dropout for regularization
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(42, return_sequences=True, dropout=0.2)),

        # Second Bidirectional LSTM layer with 42 units and dropout for regularization
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(42, dropout=0.2)),

        # Dense output layer to predict one value
        Dense(1)
    ])

    # Compile the model using the Adam optimizer and mean squared error loss function
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model using the training data generator for 5 epochs
    history = model.fit(dataset, epochs=5)


    seq = data.copy()[-window_size:]

    for i in range(num_pred):
        
        # Predict the next value using the model
        value = model.predict(seq[-window_size:].reshape(1, window_size, 1), verbose=0)
        seq = np.append(seq,value[0])

    mse = model.evaluate(dataset, verbose=0)

    return seq[-window_size:],mse


def TCN(y, num_pred=num_pred):
    """
    Train a Temporal Convolutional Network (TCN) for time series prediction using TensorFlow.

    Args:
        y (list or np.ndarray): Time series data to train the TCN model.
        num_pred (int, optional): Number of future values to predict. Default is 100.

    Returns:
        tuple: A tuple containing:
            - List of predicted values for the next `num_pred` time steps.
            - Loss value of the trained model.
    """
    y = np.array(y)
    if y.size == 0:
        return 0, "Error"

    # Create input data
    X = np.arange(len(y)).reshape(-1, 1)

    # Define sequence length and create sequences
    seq_len = 100
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_seq, y_seq))
    batch_size = 32
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

    # TCN model
    def create_tcn_model(input_shape, output_size, num_channels, kernel_size, dropout):
        inputs = layers.Input(shape=input_shape)
        x = inputs

        for i in range(len(num_channels)):
            dilation_rate = 2 ** i
            x = layers.Conv1D(filters=num_channels[i], kernel_size=kernel_size,
                              padding='causal', dilation_rate=dilation_rate)(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout)(x)

        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(output_size)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    # Model and training
    input_shape = (seq_len, 1)  # [seq_len, features]
    output_size = 1  # Predicting a single value (PM2.5)
    num_channels = [32, 32, 32, 32]  # Adjust as needed
    kernel_size = 3
    dropout = 0.2

    model = create_tcn_model(input_shape, output_size, num_channels, kernel_size, dropout)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    num_epochs = 2
    history = model.fit(dataset, epochs=num_epochs, verbose=1)

    # Initialize a list to store the predicted values
    predicted_values = []

    # Use the model to predict the next values
    x_last = X_seq[-1].reshape(1, seq_len, 1)
    
    for _ in range(num_pred):
        # Predict the next value
        y_pred = model.predict(x_last)
        
        # Append the predicted value to the list
        predicted_values.append(y_pred[0][0])
        
        # Update the sequence with the predicted value
        x_last = np.roll(x_last, -1, axis=1)
        x_last[0, -1, 0] = y_pred[0][0]

    #print("Predicted PM2.5 values for the next 100 time steps:", predicted_values)
    return predicted_values, model.loss
    
def RNN(y, num_pred=100):
    """
    Train a Recurrent Neural Network (RNN) for time series prediction using TensorFlow.

    Args:
        y (list or np.ndarray): Time series data to train the RNN model.
        num_pred (int, optional): Number of future values to predict. Default is 100.

    Returns:
        tuple: A tuple containing:
            - List of predicted values for the next `num_pred` time steps.
            - Mean squared error (MSE) loss on the training dataset.
    """
    y = np.array(y)
    if y.size == 0:
        return 0, "Error"

    # Create input data
    X = np.arange(len(y)).reshape(-1, 1)

    # Define sequence length and create sequences
    seq_len = 100
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_seq, y_seq))
    batch_size = 32
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

    # RNN model
    def create_rnn_model(input_shape, output_size, units, dropout):
        inputs = layers.Input(shape=input_shape)
        x = layers.SimpleRNN(units, return_sequences=True)(inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.SimpleRNN(units)(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(output_size)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    # Model and training
    input_shape = (seq_len, 1)  # [seq_len, features]
    output_size = 1  # Predicting a single value
    units = 50  # Number of RNN units
    dropout = 0.2

    model = create_rnn_model(input_shape, output_size, units, dropout)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    num_epochs = 2
    history = model.fit(dataset, epochs=num_epochs, verbose=1)

    # Initialize a list to store the predicted values
    predicted_values = []

    # Use the model to predict the next values
    x_last = X_seq[-1].reshape(1, seq_len, 1)
    
    for _ in range(num_pred):
        # Predict the next value
        y_pred = model.predict(x_last)
        
        # Append the predicted value to the list
        predicted_values.append(y_pred[0][0])
        
        # Update the sequence with the predicted value
        x_last = np.roll(x_last, -1, axis=1)
        x_last[0, -1, 0] = y_pred[0][0]

    mse = model.evaluate(dataset, verbose=0)
    
    return predicted_values, mse


def prophet_forecast(y, num_pred=num_pred):
    from prophet import Prophet
    """
    Fit a Prophet model to the time series data and forecast future values.
    
    Parameters:
    y (array-like): The input time series data.
    num_pred (int): The number of future time steps to predict. Default is 100.
    
    Returns:
    tuple: A tuple containing:
        - list: Forecasted values of the time series.
        - float: Mean squared error of the model on the test set.
    
    Raises:
    ValueError: If the input array `y` is empty or too small to split.
    """
    
    # Check if the input is empty
    if len(y) == 0:
        raise ValueError("The input array y is empty.")
    
    # Prepare the dataframe with time and values for Prophet
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),  # Create a date range for `ds`
        'y': y  # Target variable
    })
    
    # Split the data into training and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Initialize the Prophet model
    model = Prophet()
    
    # Fit the model on the training data
    model.fit(train_data)
    
    # Generate a dataframe to hold future dates for prediction
    future = model.make_future_dataframe(periods=num_pred)
    
    # Predict future values
    forecast = model.predict(future)
    
    # Extract forecasted values
    forecast_values = forecast['yhat'].iloc[-num_pred:].values
    
    # Predict on the test set to calculate MSE
    test_predictions = model.predict(test_data[['ds']])
    
    # Compute Mean Squared Error between test set and test predictions
    mse = mean_squared_error(test_data['y'], test_predictions['yhat'])
    
    return list(forecast_values), float(mse)