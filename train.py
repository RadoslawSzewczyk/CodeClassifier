import torch
import logging
from data import prepare_line_level_data
from config import Config, debugLevel
from model import CodeClassifier
from utils import evaluate

debugLevel = debugLevel
logging.basicConfig(level=debugLevel)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    train_samples, test_samples, vocab, label_to_idx = prepare_line_level_data()
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_samples,
        batch_size=Config.batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_samples,
        batch_size=Config.batch_size,
        shuffle=False,
        pin_memory=True
    )

    model = CodeClassifier(len(vocab), len(label_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    results = []
    logging.info("Training model...")
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
        train_acc = correct / total
        test_acc = evaluate(model, test_loader)
        results.extend([
            f"Epoch {epoch+1}",
            f"Train Accuracy: {train_acc:.4f}",
            f"Test Accuracy: {test_acc:.4f}",
            f"Train Loss: {total_loss:.4f}",
            f"Model summary: {Config().prettyPrint()}"
        ])
        logging.info(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    with open(Config.results_filename, 'w') as f:
        f.write("\n".join(results))
    torch.save(model.state_dict(), "code_classifier.pth")


def test_model(max_lines=None):
    _, _, _, _ = None, None, None, None
    train_samples, test_samples, vocab, label_to_idx = prepare_line_level_data()
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_samples,
        batch_size=Config.batch_size,
        shuffle=False,
        pin_memory=True
    )
    model = CodeClassifier(len(vocab), len(label_to_idx)).to(device)
    model.load_state_dict(torch.load("code_classifier.pth", map_location=device))
    acc = evaluate(model, test_loader)
    print(f"Test accuracy: {acc:.4f}")
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_to_idx.keys()),
                yticklabels=list(label_to_idx.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
