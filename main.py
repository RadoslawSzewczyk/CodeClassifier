import os
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

TOKEN_REGEX = re.compile(r'\w+|[^\w\s]')

class Config:
    data_dir = "trainingData"
    max_features = 2000
    sequence_length = 256
    embedding_dim = 128
    batch_size = 8
    epochs = 15
    min_line_length = 40
    file_chunk_size = 1024 * 1024
    max_workers = cpu_count() // 2
    results_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_results.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                        tokens = TOKEN_REGEX.findall(line)
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
        
        tokens = line.split()[:Config.sequence_length]
        indices = [self.vocab.get(t, 1) for t in tokens]
        indices += [0] * (Config.sequence_length - len(indices))
        return torch.LongTensor(indices), label

class CodeClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.embedding_dim)
        self.gru = nn.GRU(Config.embedding_dim, 256, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        concatenated = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(concatenated)

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
    train_files, test_files, label_to_idx = load_file_paths()
    vocab = build_vocabulary(train_files)
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

if __name__ == "__main__":
    train_model()