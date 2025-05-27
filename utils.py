import torch
import logging
from data import load_file_paths, build_vocabulary
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
            # logging.debug(f"Inputs: {inputs}")
            # logging.debug(f"Labels: {labels}")
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total if total > 0 else float("nan")

def prepare_data():
    train_files, test_files, label_to_idx = load_file_paths()
    vocab = build_vocabulary(train_files + test_files)
    return train_files, test_files, vocab, label_to_idx

