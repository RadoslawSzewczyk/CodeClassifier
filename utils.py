import torch
import logging
from data import prepare_line_level_data
from config import debugLevel

logging.basicConfig(level=debugLevel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total if total > 0 else float("nan")

def prepare_data():
    return prepare_line_level_data()
