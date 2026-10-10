"""TextCNN and BiLSTM classifiers. Both take (batch, seq_len) token ids with 0 = <pad>."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_ID = 0


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, filter_sizes=(3, 4, 5), num_filters=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, num_filters, k) for k in filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        mask = (x != PAD_ID).unsqueeze(1)                     # (B, 1, T)
        emb = self.embedding(x).transpose(1, 2)               # (B, E, T)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))                             # (B, F, T-k+1)
            # a window is valid if its first token is a real token (padding is only at the end)
            valid = mask[:, :, : h.size(2)]
            h = h.masked_fill(~valid, float("-inf"))
            pooled.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        lengths = (x != PAD_ID).sum(dim=1).clamp(min=1)
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))  # (B, T, 2H)
        mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]).unsqueeze(2)
        out = out.masked_fill(~mask, float("-inf"))
        return self.fc(self.dropout(out.max(dim=1).values))


def build_model(name, vocab_size, num_classes):
    if name == "cnn":
        return TextCNN(vocab_size, num_classes)
    if name == "lstm":
        return BiLSTM(vocab_size, num_classes)
    raise ValueError(f"unknown model {name}")
