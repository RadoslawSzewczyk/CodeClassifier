import os
import random
import mmap
import heapq
import re
from collections import defaultdict, Counter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import Config
import torch
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count
import json
from collections import defaultdict
from config import Config
import logging

logging.basicConfig(level=logging.DEBUG)
TOKEN_REGEX = re.compile(r'\w+|[^\w\s]')

def load_file_paths():
    file_paths = []
    label_to_idx = {}
    idx_counter = 0

    for filename in os.listdir(Config.data_dir):
        if filename.endswith('.txt'):
            label = filename.rsplit('.', 1)[0]
            if label not in label_to_idx:
                label_to_idx[label] = idx_counter
                idx_counter += 1
            file_paths.append((os.path.join(Config.data_dir, filename), label_to_idx[label]))

    labels = [lbl for _, lbl in file_paths]
    label_counts = Counter(labels)

    if all(count >= 2 for count in label_counts.values()):
        train, test = train_test_split(
            file_paths,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )
    else:
        train, test = train_test_split(
            file_paths,
            test_size=0.2,
            shuffle=True,
            random_state=42
        )

    return train, test, label_to_idx

def process_file(filepath):
    local_counts = defaultdict(int)
    lines_read = 0

    with open(filepath, 'r') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
            buffer = ''
            while lines_read < Config.train_lines:
                chunk = mmap_obj.read(Config.file_chunk_size)
                if not chunk:
                    break
                chunk = buffer + chunk.decode('utf-8', errors='ignore')
                lines = chunk.split('\n')
                buffer = lines.pop() if not chunk.endswith('\n') else ''
                for line in lines:
                    if lines_read >= Config.train_lines:
                        break
                    if len(line) >= Config.min_line_length:
                        tokens = TOKEN_REGEX.findall(line)[:Config.sequence_length]
                        for token in tokens:
                            local_counts[token] += 1
                        lines_read += 1
    return local_counts

def build_vocabulary(files):
    logging.info("Building vocabulary...")
    vocab = {"<pad>": 0, "<unk>": 1}
    token_counts = defaultdict(int)
    
    file_paths = [fp for fp, _ in files]

    with Pool(processes=max(1, cpu_count() * 3 // 4)) as pool:
        results = pool.imap(process_file, file_paths, chunksize=2)
        
        for i, result in enumerate(results):
            for token, count in result.items():
                token_counts[token] += count
            logging.info(f"Processed {i+1}/{len(file_paths)} files...")

    top_tokens = heapq.nlargest(
        Config.max_features - 2,
        token_counts.items(),
        key=lambda x: x[1]
    )

    for token, _ in top_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    logging.info(f"Vocabulary built with {len(vocab)} tokens")
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

        with open(filepath, 'rb') as f:
            mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            raw_lines = mmap_obj.read().split(b'\n')
            mmap_obj.close()

        valid = []
        for ln in raw_lines:
            if len(valid) >= Config.train_lines:
                break
            if len(ln) >= Config.min_line_length:
                try:
                    valid.append(ln.decode('utf-8', errors='replace').strip())
                except UnicodeDecodeError:
                    continue

        if valid:
            line_idx = (idx // len(self.file_list)) % len(valid)
            line = valid[line_idx]
        else:
            line = ""

        tokens = TOKEN_REGEX.findall(line)[:Config.sequence_length]
        indices = [self.vocab.get(t, self.vocab.get("<unk>")) for t in tokens]
        indices += [self.vocab.get("<pad>")] * (Config.sequence_length - len(indices))
        return torch.LongTensor(indices), label

def create_data_loaders(train_files, test_files, vocab):
    train_dataset = CodeDataset(train_files, vocab)
    test_dataset = CodeDataset(test_files, vocab)
    
    return (
        DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, pin_memory=True),
        DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, pin_memory=True)
    )


def export_vocab():
    _, _, vocab = load_file_paths() 
    
    with open(Config.vocab_filename, 'w') as f:
        json.dump(vocab, f, indent=4)
    
    logging.info(f"Vocabulary exported to {Config.vocab_filename}")