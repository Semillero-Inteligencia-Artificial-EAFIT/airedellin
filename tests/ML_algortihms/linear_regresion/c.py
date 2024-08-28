import numpy as np
import polars as pl
import pickle
from datetime import datetime

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Check for division by zero scenario
        if np.std(y) == 0:
            # All y values are the same, set bias to the constant y value and weights to zero
            self.weights = np.zeros(n_features)
            self.bias = y[0]
            return
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def save_model(self, file_name):
        model_data = {
            'weights': self.weights,
            'bias': self.bias
        }
        with open(file_name, 'wb') as file:
            pickle.dump(model_data, file)

    def load_model(self, file_name):
        with open(file_name, 'rb') as file:
            model_data = pickle.load(file)
        self.weights = model_data['weights']
        self.bias = model_data['bias']
    def printvalues(self):
        print(self.weights,self.bias)

def read_data(file_name,head=100):
    df = pl.read_csv(file_name).head(head)
    fechas_hora = df['Fecha_Hora'].to_list()
    y = np.array([(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds() for date_str in fechas_hora])
    X = np.arange(len(y)).reshape(-1, 1)
    return X, y

# Example Usage
if __name__ == "__main__":
    X, y = read_data('cleaned_data.csv')

    # Normalize y
    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min)

    model = LinearRegression(learning_rate=0.001, n_iterations=1000)
    model.fit(X, y)
    model.printvalues()

    # Save and load the model as before
    model.save_model("linear_regression_model.pkl")
    loaded_model = LinearRegression()
    loaded_model.load_model("linear_regression_model.pkl")

    # Make predictions
    predictions = loaded_model.predict(X)

    # Denormalize predictions
    predictions = predictions * (y_max - y_min) + y_min

    print("Predicted values from loaded model:", predictions)
