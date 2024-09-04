import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import r2_score

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels[0], kernel_size, stride=1, padding=(kernel_size-1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size, stride=1, padding=(kernel_size-1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[1], output_size, kernel_size, stride=1, padding=(kernel_size-1))
        )
        
    def forward(self, x):
        return self.tcn(x)

def train_tcn(model, train_loader, num_epochs, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def tcn_forecast(y, num_pred=10, num_epochs=50, save_model_path='tcn_model.pth'):
    """
    Apply Temporal Convolutional Network (TCN) to time series data and predict future values.

    Parameters:
    y (array-like): The input time series data.
    num_pred (int): The number of future time steps to predict. Default is 10.
    num_epochs (int): Number of epochs for training. Default is 50.
    save_model_path (str): Path to save the trained model. Default is 'tcn_model.pth'.

    Returns:
    tuple: A tuple containing:
        - list: Predicted future values of the time series.
        - float: R-squared score of the model on the training set.

    Raises:
    ValueError: If the input array `y` is empty or if the dataset is too small to model.
    """
    y = np.array(y, dtype=np.float32)
    if y.size == 0:
        return 0, "Error"

    if len(y) <= 1:
        return 0, "The dataset is too small to model."

    input_size = 1
    output_size = 1
    num_channels = [16, 32]
    kernel_size = 2
    dropout = 0.2
    sequence_length = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = torch.tensor(y).float().unsqueeze(1)
    x = torch.arange(len(y)).float().unsqueeze(1)

    sequences = []
    targets = []
    for i in range(len(y) - sequence_length):
        sequences.append(y[i:i + sequence_length])
        targets.append(y[i + sequence_length])

    sequences = torch.stack(sequences)
    targets = torch.stack(targets)

    dataset = TensorDataset(sequences, targets)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = TCN(input_size, output_size, num_channels, kernel_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_tcn(model, train_loader, num_epochs, criterion, optimizer)

    # Save the model
    torch.save(model.state_dict(), save_model_path)

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            y_pred.append(outputs)
            y_true.append(targets)

    y_pred = torch.cat(y_pred).cpu().numpy()
    y_true = torch.cat(y_true).cpu().numpy()

    r2 = r2_score(y_true, y_pred)

    # Predict future values
    future_pred = []
    current_input = y[-sequence_length:].unsqueeze(0).to(device)
    for _ in range(num_pred):
        future_output = model(current_input)
        future_pred.append(future_output.item())
        current_input = torch.cat((current_input[:, 1:], future_output.unsqueeze(0)), dim=1)

    return future_pred, r2

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

def predict_with_loaded_model(y, num_pred=10, model_path='tcn_model.pth'):
    """
    Load a saved TCN model and use it to predict future values.

    Parameters:
    y (array-like): The input time series data.
    num_pred (int): The number of future time steps to predict. Default is 10.
    model_path (str): Path to the saved model. Default is 'tcn_model.pth'.

    Returns:
    list: Predicted future values of the time series.
    """
    y = np.array(y, dtype=np.float32)
    if y.size == 0:
        return "Error"

    sequence_length = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = torch.tensor(y).float().unsqueeze(1)
    current_input = y[-sequence_length:].unsqueeze(0).to(device)

    # Initialize model
    model = TCN(input_size=1, output_size=1, num_channels=[16, 32]).to(device)
    
    # Load the model
    load_model(model, model_path)

    # Predict future values
    future_pred = []
    with torch.no_grad():
        for _ in range(num_pred):
            future_output = model(current_input)
            future_pred.append(future_output.item())
            current_input = torch.cat((current_input[:, 1:], future_output.unsqueeze(0)), dim=1)

    return future_pred

def main():
    # Generate synthetic time series data
    np.random.seed(0)
    time_series = np.sin(np.linspace(0, 10 * np.pi, 100)) + np.random.normal(scale=0.1, size=100)

    # Train the TCN model
    num_pred = 10
    num_epochs = 50
    save_model_path = 'tcn_model.pth'

    future_pred, r2 = tcn_forecast(time_series, num_pred=num_pred, num_epochs=num_epochs, save_model_path=save_model_path)
    print("R-squared score on the training set:", r2)
    print("Future predictions:", future_pred)

    # Load the model and make predictions
    loaded_future_pred = predict_with_loaded_model(time_series, num_pred=num_pred, model_path=save_model_path)
    print("Future predictions from the loaded model:", loaded_future_pred)

main()