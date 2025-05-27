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
from config import Config, debugLevel
import logging

logging.basicConfig(level=debugLevel)
TOKEN_REGEX = re.compile(r'\w+|[^\w\s]')

def processSpecialCharsAndKeyWords(input_path: str):
    with open('key_words', 'r', encoding='utf-8') as kw:
        keywords = set(line.strip().lower() for line in kw if line.strip())

    specials = [kw for kw in keywords if re.search(r'\W', kw)]
    specials.sort(key=len, reverse=True)
    token_pattern = re.compile(
        r"(?:" + "|".join(re.escape(op) for op in specials) + r")" +
        r"|\w+"
    )

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open('filtered_output.txt', 'w', encoding='utf-8') as fout:

        for line in fin:
            tokens = [m.group(0) for m in token_pattern.finditer(line.lower())]
            kept   = [t for t in tokens if t in keywords]
            fout.write(" ".join(kept) + "\n")

    logging.info(f"processSpecialCharsAndKeyWords: done → filtered_output.txt (from {input_path})")


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
    print(label_to_idx)
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
        self.samples = []
        for filepath, label in file_list:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = []
                for line in f:
                    if len(line) >= Config.min_line_length:
                        tokens = TOKEN_REGEX.findall(line)[:Config.sequence_length]
                        ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
                        ids += [vocab["<pad>"]] * (Config.sequence_length - len(ids))
                        lines.append((torch.LongTensor(ids), label))
                    if len(lines) >= Config.train_lines:
                        break
            self.samples.extend(lines)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_data_loaders(train_files, test_files, vocab):
    train_loader = None
    if train_files:
        train_ds = CodeDataset(train_files, vocab)
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            pin_memory=True
        )

    test_loader = None
    if test_files:
        test_ds = CodeDataset(test_files, vocab)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            pin_memory=True
        )

    return train_loader, test_loader


def export_vocab():
    _, _, vocab = load_file_paths() 
    
    with open(Config.vocab_filename, 'w') as f:
        json.dump(vocab, f, indent=4)
    
    logging.info(f"Vocabulary exported to {Config.vocab_filename}")