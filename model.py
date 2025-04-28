import torch
import torch.nn as nn

class CodeClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.embed_dropout = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(
            256, 
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
