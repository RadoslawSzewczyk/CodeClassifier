import torch
import torch.optim as optim
import torch.nn as nn
from model import CodeClassifier
from data import create_data_loaders
from config import Config, debugLevel
from utils import evaluate, prepare_data
import logging

logging.basicConfig(level=debugLevel)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}")

def train_model():
    logging.info("Loading data...")
    train_files, test_files, vocab, label_to_idx = prepare_data()
    train_loader, test_loader = create_data_loaders(train_files, test_files, vocab)
    
    model = CodeClassifier(len(vocab), len(label_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    logging.info("Training model...")
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        logging.debug(f"Epoch {epoch+1}/{Config.epochs}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = correct / total
        test_acc = evaluate(model, test_loader)
        
        results.append(f"Epoch {epoch+1}")
        results.append(f"Train Accuracy: {train_acc:.4f}")
        results.append(f"Test Accuracy: {test_acc:.4f}\n")
        results.append(f"Train Loss: {total_loss:.4f}\n")
        results.append(f"Model summary: {Config().prettyPrint()}\n")
        logging.info(f"Epoch {epoch+1}")
        logging.info(f"Train Acc: {train_acc:.4f}")
        logging.info(f"Test Acc: {test_acc:.4f}\n")
    
    with open(Config.results_filename, 'w') as f:
        f.write("\n".join(results))
    
    torch.save(model.state_dict(), 'code_classifier.pth')

def test_model(max_lines):
    _, test_files, vocab, label_to_idx = prepare_data()
    _, test_loader = create_data_loaders([], test_files, vocab)
    model = CodeClassifier(len(vocab), len(label_to_idx)).to(device)
    model.load_state_dict(torch.load('code_classifier.pth', map_location=device))
    acc = evaluate(model, test_loader)
    print(f"Test accuracy on last {max_lines} lines/file: {acc:.4f}")
