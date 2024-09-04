import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.nn.utils import weight_norm

# Simulated data
batch_size = 32
seq_len = 100
input_size = 10
output_size = 5

X = torch.randn(batch_size, seq_len, input_size)
y = torch.randn(batch_size, seq_len, output_size)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# TCN model
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.ModuleList([
            weight_norm(nn.Conv1d(num_channels[i], num_channels[i+1], kernel_size, padding=(kernel_size-1)//2))
            for i in range(len(num_channels)-1)
        ])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """Assuming input size is (batch, seq_len, input_size)"""
        x = x.transpose(1, 2)  # -> (batch, input_size, seq_len)
        for i, conv in enumerate(self.tcn):
            x = torch.relu(self.dropout(conv(x)))
        x = x.transpose(1, 2)  # -> (batch, seq_len, output_size)
        return self.linear(x)

# Model and training
num_channels = [input_size] + [64] * 3 + [output_size]
kernel_size = 7
dropout = 0.2
model = TCN(input_size, output_size, num_channels, kernel_size, dropout)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for X, y in dataloader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), 'tcn_model.pth')

# Load the model
model.load_state_dict(torch.load('tcn_model.pth'))
model.eval()

# Predict the next values
with torch.no_grad():
    # Get the last batch of data
    X_last, _ = next(iter(dataloader))
    
    # Get the last sequence from the batch
    x_last = X_last[-1:, :, :]
    
    # Predict the next values
    y_pred = model(x_last)
    
    print("Predicted next values:", y_pred.squeeze())