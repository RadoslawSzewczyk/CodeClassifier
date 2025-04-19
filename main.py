import os
import typer
import random
import mmap
import heapq
import re
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

app = typer.Typer()

TOKEN_REGEX = re.compile(r'\w+|[^\w\s]')

class Config:
    data_dir = "trainingData"
    output_dir = "outputs"
    model_name = "code_classifier"
    tokenizer_regex = r'\w+|[^\w\s]'
    max_features = 200000
    sequence_length = 256
    embedding_dim = 256
    batch_size = 32
    epochs = 15
    min_line_length = 40
    file_chunk_size = 1024 * 1024
    max_workers = cpu_count() // 2
    results_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_results.txt"

class CodeClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.embedding_dim)
        self.embed_dropout = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(
            Config.embedding_dim, 
            256, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1, bias=False)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        embedded = self.embed_dropout(self.embedding(x))
        
        lstm_out, _ = self.lstm(embedded)
        
        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), 
            dim=1
        )
        context_vector = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1), 
            dim=1
        )
        
        return self.classifier(context_vector)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(build_vocab=True):
    train_files, test_files, label_to_idx = load_file_paths()
    vocab = build_vocabulary(train_files) if build_vocab else None
    return train_files, test_files, vocab, label_to_idx

def load_file_paths():
    file_paths = []
    label_to_idx = {}
    idx_counter = 0
    
    for filename in os.listdir(Config.data_dir):
        if filename.endswith('.txt'):
            label = filename.split('.')[0]
            if label not in label_to_idx:
                label_to_idx[label] = idx_counter
                idx_counter += 1
            file_paths.append((os.path.join(Config.data_dir, filename), label_to_idx[label]))
    
    random.shuffle(file_paths)
    train, test = train_test_split(file_paths, test_size=0.2, random_state=42)
    return train, test, label_to_idx

def process_file(filepath):
    local_counts = defaultdict(int)
    with open(filepath, 'r') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
            buffer = ''
            while True:
                chunk = mmap_obj.read(Config.file_chunk_size)
                if not chunk:
                    break
                chunk = buffer + chunk.decode('utf-8', errors='ignore')
                lines = chunk.split('\n')
                
                buffer = lines.pop() if not chunk.endswith('\n') else ''
                
                for line in lines:
                    if len(line) >= Config.min_line_length:
                        tokens = TOKEN_REGEX.findall(line)[:Config.sequence_length]
                        for token in tokens:
                            local_counts[token] += 1
    return local_counts

def build_vocabulary(files):
    print("Building vocabulary...")
    vocab = {"<pad>": 0, "<unk>": 1}
    token_counts = defaultdict(int)
    
    file_paths = [fp for fp, _ in files]

    with Pool(processes=max(1, cpu_count() * 3 // 4)) as pool:
        results = pool.imap(process_file, file_paths, chunksize=2)
        
        for i, result in enumerate(results):
            for token, count in result.items():
                token_counts[token] += count
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(file_paths)} files...")

    top_tokens = heapq.nlargest(
        Config.max_features - 2,
        token_counts.items(),
        key=lambda x: x[1]
    )

    for token, _ in top_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    print(f"Vocabulary built with {len(vocab)} tokens")
    return vocab

class CodeDataset(Dataset):
    def __init__(self, file_list, vocab):
        self.file_list = file_list
        self.vocab = vocab

    def __len__(self):
        return len(self.file_list) * 1000

    def __getitem__(self, idx):
        file_idx = idx % len(self.file_list)
        filepath, label = self.file_list[file_idx]
        
        with open(filepath, 'rb') as file:
            memory_map = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            file_size = memory_map.size()
            start_pos = random.randint(0, max(0, file_size - Config.file_chunk_size))
            
            chunk = memory_map[start_pos:start_pos+Config.file_chunk_size]
            lines = chunk.split(b'\n')
            
            valid_lines = []
            for ln in lines:
                if len(ln) >= Config.min_line_length:
                    try:
                        decoded = ln.decode('utf-8', errors='replace').strip()
                        valid_lines.append(decoded)
                    except UnicodeDecodeError:
                        continue
            
            line = random.choice(valid_lines) if valid_lines else ""
            memory_map.close()
        
        tokens = TOKEN_REGEX.findall(line)[:Config.sequence_length]
        indices = [self.vocab.get(t, 1) for t in tokens]
        indices += [0] * (Config.sequence_length - len(indices))
        return torch.LongTensor(indices), label

def create_data_loaders(train_files, test_files, vocab):
    train_dataset = CodeDataset(train_files, vocab)
    test_dataset = CodeDataset(test_files, vocab)
    
    return (
        DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, pin_memory=True),
        DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, pin_memory=True)
    )

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
    
    return correct / total

def train_model():
    print("Loading data...")
    train_files, test_files, vocab, label_to_idx = prepare_data()
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


def load_trained_model(vocab, label_to_idx):
    model = CodeClassifier(len(vocab), len(label_to_idx)).to(device)
    model.load_state_dict(torch.load('code_classifier.pth'))
    return model

def evaluate_model():
    train_files, test_files, vocab, label_to_idx = prepare_data()
    _, test_loader = create_data_loaders(train_files, test_files, vocab)

    model = load_trained_model(vocab, label_to_idx)
    
    acc = evaluate(model, test_loader)
    print(f"Evaluation Accuracy: {acc:.4f}")

def export_vocab():
    train_files, _, vocab, _ = prepare_data()
    
    os.makedirs(Config.output_dir, exist_ok=True)
    vocab_path = os.path.join(Config.output_dir, 'vocab.txt')
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, idx in vocab.items():
            f.write(f"{token}\t{idx}\n")
    print(f"Vocabulary exported to {vocab_path}")

@app.command()
def train(): train_model()

@app.command()
def evaluate(): evaluate_model()

@app.command()
def export_vocabulary(): export_vocab()

if __name__ == "__main__":
    app()
    