import torch
import torch.nn as nn
import torch.nn.functional as F

class CodeClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=160, conv_channels=160, kernel_sizes=(3, 5, 7), dropout=0.4, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.01)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, conv_channels, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        hidden_dim = conv_channels * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        b, l = x.size()
        pos = self.pos_embed[:, :l, :]
        x = self.embedding(x) + pos
        x = x.transpose(1, 2)
        outs = [F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(-1) for conv in self.conv_layers]
        x = torch.cat(outs, dim=1)
        x = self.dropout(x)
        return self.classifier(x)
