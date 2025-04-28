import os
from datetime import datetime
from multiprocessing import cpu_count

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
