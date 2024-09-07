import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import polars as pl

# Read the CSV file
df = pl.read_csv("cleaned_data.csv")
print(df.head())

# Select the 'pm25' column as the target (y) and convert to a tensor
y = torch.tensor(df.select("pm25").to_numpy(), dtype=torch.float32)

# Create a tensor for X (in this case, range values)
X = torch.tensor(np.arange(len(y)), dtype=torch.float32)
X = X.unsqueeze(1)  

print("X shape:", X.shape)
print("y shape:", y.shape)

# Define sequence length and create sequences
seq_len = 100
X_seq = []
y_seq = []

for i in range(len(X) - seq_len):
    X_seq.append(X[i:i+seq_len])
    y_seq.append(y[i+seq_len])

X_seq = torch.stack(X_seq)
y_seq = torch.stack(y_seq)

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)

# Create dataset and dataloader
dataset = TensorDataset(X_seq, y_seq)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# TCN model
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=(kernel_size-1) * dilation_size,
                                 dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape from [batch, seq_len, features] to [batch, features, seq_len]
        x = self.network(x)
        return self.linear(x[:, :, -1])

# Model and training
input_size = 1  # Single feature (time index)
output_size = 1  # Predicting a single value (pm25)
num_channels = [32, 32, 32, 32]  # Adjust as needed
kernel_size = 3
dropout = 0.2

model = TCN(input_size, output_size, num_channels, kernel_size, dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'tcn_model.pth')

# Load the model
model.load_state_dict(torch.load('tcn_model.pth'))
model.eval()

# Predict the next values
with torch.no_grad():
    # Get the last sequence from the data
    x_last = X_seq[-1].unsqueeze(0)
    
    # Predict the next value
    y_pred = model(x_last)
    
    print("Predicted next PM2.5 value:", y_pred.item())

print("PM2.5 prediction complete")