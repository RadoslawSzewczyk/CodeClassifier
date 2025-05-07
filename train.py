import torch
import torch.optim as optim
import torch.nn as nn
from model import CodeClassifier
from data import create_data_loaders
from config import Config
from utils import evaluate, prepare_data
import logging

logging.basicConfig(level=logging.DEBUG)
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
            print("a")
            inputs, labels = inputs.to(device), labels.to(device)
            print("b")
            optimizer.zero_grad()
            print("c")
            outputs = model(inputs)
            print("d")
            loss = criterion(outputs, labels)
            print("e")
            loss.backward()
            print("f")
            optimizer.step()
            print("g")
            total_loss += loss.item()
            print("h")
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("i")
        
        train_acc = correct / total
        test_acc = evaluate(model, test_loader)
        
        results.append(f"Epoch {epoch+1}")
        results.append(f"Train Accuracy: {train_acc:.4f}")
        results.append(f"Test Accuracy: {test_acc:.4f}\n")
        results.append(f"Train Loss: {total_loss:.4f}\n")
        results.append(f"Model summary: {Config().prettyPrint()}\n")
        print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        logging.info(f"Epoch {epoch+1}")
        logging.info(f"Train Acc: {train_acc:.4f}")
        logging.info(f"Test Acc: {test_acc:.4f}\n")
    
    with open(Config.results_filename, 'w') as f:
        f.write("\n".join(results))
    logging.info("Training complete. Results saved to:", Config.results_filename)
    
    torch.save(model.state_dict(), 'code_classifier.pth')
