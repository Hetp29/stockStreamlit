import torch 

def predict_stock_prices(stock_data):
    model = torch.nn.Linear(1, 1)
    prices_tensor = torch.tensor(stock_data['Close'].values, dtype=torch.float32).view(-1, 1)
    predictions = model(prices_tensor)
    return predictions 