# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import CodeClassifier
from data import create_data_loaders
from config import Config
from utils import evaluate, prepare_data  # Importujemy evaluate i prepare_data z utils.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print("Loading data...")
    train_files, test_files, vocab, label_to_idx = prepare_data()  # Używamy funkcji prepare_data
    train_loader, test_loader = create_data_loaders(train_files, test_files, vocab)
    
    model = CodeClassifier(len(vocab), len(label_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    print("Training model...")
    print(f"Vocabulary size: {len(vocab)}")
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_acc = correct / total
        test_acc = evaluate(model, test_loader)
        
        results.append(f"Epoch {epoch+1}")
        results.append(f"Train Accuracy: {train_acc:.4f}")
        results.append(f"Test Accuracy: {test_acc:.4f}\n")
        
        print(f"Epoch {epoch+1}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}\n")
    
    with open(Config.results_filename, 'w') as f:
        f.write("\n".join(results))
    
    torch.save(model.state_dict(), 'code_classifier.pth')
