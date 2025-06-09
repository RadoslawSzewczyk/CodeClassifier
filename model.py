import torch
import torch.nn as nn


class CodeClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=False,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)            
        lstm_out, (h_n, _) = self.lstm(embedded) 
        final_hidden = h_n[-1]                   
        return self.classifier(final_hidden)
