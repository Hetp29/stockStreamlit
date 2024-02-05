import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset

class StockPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(StockPredictor, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
    
def predict_stock_prices(stock_data):
    X = torch.arange(len(stock_data)).float().view(-1, 1)
    y = torch.tensor(stock_data['Close'].values, dtype=torch.float32).view(-1, 1)
    
    model = StockPredictor(input_size=1, output_size=1)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    for epoch in range(1000):
        predictions = model(X)
        loss = criterion(predictions,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        all_predictions = model(X).numpy()
    return all_predictions
