import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

name = "cleaned_data.csv"
mesures = pd.read_csv(name).head(100)
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
plt.savefig("plot.png")  # Save the plot instead of showing it

# If running in an interactive environment:
# plt.show()
