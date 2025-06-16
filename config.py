import os
from datetime import datetime
from multiprocessing import cpu_count
import logging

debugLevel = logging.INFO
logging.basicConfig(level=debugLevel)
class Config:
    data_dir = "trainingData"
    output_dir = "outputs"
    model_name = "code_classifier"
    tokenizer_regex = r'\w+|[^\w\s]'
    max_features = 200000
    sequence_length = 256
    embedding_dim = 256
    batch_size = 8
    epochs = 30
    min_line_length = 40
    file_chunk_size = 1024 * 1024
    max_workers = cpu_count() // 2
    train_lines = 10000
    results_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_results.txt"
    
    def prettyPrint(self):
        logging.info("Configuration:")
        logging.info(f"Config: {self.__dict__}")
        logging.info(f"Max workers: {self.max_workers}")
        logging.info(f"Max features: {self.max_features}")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"Epochs: {self.epochs}")
        logging.info(f"Sequence length: {self.sequence_length}")
        logging.info(f"Embedding dim: {self.embedding_dim}")
        logging.info(f"Min line length: {self.min_line_length}")
        logging.info(f"File chunk size: {self.file_chunk_size}")
        logging.info(f"Train lines: {self.train_lines}")
        if(debugLevel != logging.DEBUG):
            logging.info(f"Logging level: {debugLevel}")
