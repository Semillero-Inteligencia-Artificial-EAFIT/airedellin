import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def train_and_save_model(datafile, model_file, plot_file):
    # Load and preprocess the data
    mesures = pd.read_csv(datafile).head(100)
    epoch = mesures["Fecha_Hora"].values
    pm25 = mesures["pm25"].values

    # Convert epoch to numeric values (assuming this is what you want)
    epochs = np.arange(len(epoch))
    pm25s = np.asarray(pm25)

    # Calculate the mean value of pm25
    b = np.mean(pm25s)

    # Calculate w, avoiding division by zero
    w = np.zeros_like(epochs, dtype=float)
    for i in range(len(epochs)):
        denominator = (epochs[i] * epochs[i]) ** 2
        if denominator != 0:
            w[i] = ((epochs[i] * epochs[i]) - (pm25s[i] * pm25s[i])) / denominator

    # Calculate predictions
    predictions = w * epochs + b

    # Fit and evaluate a polynomial (linear fit)
    filtred = np.polyfit(epochs, pm25s, 1)
    Y = np.polyval(filtred, epochs)

    # Plotting
    plt.xlabel("epochs")
    plt.ylabel("pm25")
    plt.title("Unloquer Predict")
    plt.plot(epochs, pm25, 'bo', label='pm25')
    plt.plot(epochs, Y, "go", label='prediction')
    plt.legend()
    plt.savefig(plot_file)  # Save the plot instead of showing it

    # Save the model (fitted polynomial coefficients) to a file
    with open(model_file, 'wb') as f:
        pickle.dump(filtred, f)

    print(f"Model saved to {model_file}")
    print(f"Plot saved to {plot_file}")


def load_model_and_predict(model_file, new_epochs):
    # Load the model from the file
    with open(model_file, 'rb') as f:
        filtred = pickle.load(f)
    
    # Predict new values using the loaded model
    predictions = np.polyval(filtred, new_epochs)
    
    return predictions

# Example usage
new_epochs = np.array([1, 102, 103, 104,3])  # Example of new epochs
predictions = load_model_and_predict("pm25_model.pkl", new_epochs)
print(predictions)
# Example usage
#train_and_save_model("cleaned_data.csv", "pm25_model.pkl", "plot.png")
