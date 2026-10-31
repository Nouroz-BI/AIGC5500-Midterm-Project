'''Text CNN classifier and training loop for the AIGC 5500 midterm project.

The training loop follows the same procedure as src/transformer_model.py, and the
evaluation and cost functions are imported from it, so every model is trained and
measured the same way.

Ablation: kernel_sizes=(1, 1, 1). Width-1 kernels see one word at a time, so the
model can no longer detect multi-word (local) patterns.
'''
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.reproducibility import set_seed
from src.text import PAD_ID, make_data_loader

# Shared with the Transformer, so all models are measured by the same code
from src.transformer_model import count_parameters, evaluate, hardware_description, measure_cost  # noqa: F401


class TextCNN(nn.Module):
    '''Embedding -> parallel 1D convolutions (one per kernel width) -> masked global max-pool -> linear.'''

    def __init__(self, vocab_size, embed_dim=64, num_filters=100, kernel_sizes=(3, 5, 7),
                 dropout=0.1, num_classes=2):
        super().__init__()
        # Odd widths with padding k // 2 keep the output length equal to the input length,
        # so padding positions can be masked before pooling
        for k in kernel_sizes:
            if k % 2 == 0:
                raise ValueError(f'Kernel sizes must be odd; got {k}.')
        self.kernel_sizes = tuple(kernel_sizes)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, tokens):
        pad_mask = (tokens == PAD_ID).unsqueeze(1)          # (B, 1, T)
        x = self.embedding(tokens).transpose(1, 2)          # (B, E, T): Conv1d expects channels before time

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))                             # (B, F, T): pattern score at each position
            h = h.masked_fill(pad_mask, float('-inf'))      # padding can never be the max
            pooled.append(h.max(dim=2).values)              # (B, F): strongest match anywhere in the email

        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


def build_model(vocab_size, model_config, variant_config):
    return TextCNN(vocab_size, **model_config, **variant_config)


def train_one_run(model_config, variant_config, seed, train_dataset, validation_loader, vocab_size,
                  device, train_config, checkpoint_path):
    '''Train one variant for one seed, keeping the epoch with the best validation F1.

    Returns (model with best-validation weights, per-epoch history, best epoch, seconds).
    '''
    set_seed(seed)
    model = build_model(vocab_size, model_config, variant_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'],
                                  weight_decay=train_config['weight_decay'])
    loader = make_data_loader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, seed=seed)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, history = -1.0, 0, []

    start = time.perf_counter()
    for epoch in range(1, train_config['epochs'] + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            tokens, labels = batch['tokens'].to(device), batch['labels'].to(device)
            loss = F.cross_entropy(model(tokens), labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
            optimizer.step()
            running += loss.item() * len(labels)
            seen += len(labels)

        val = evaluate(model, validation_loader, device)
        history.append({'epoch': epoch, 'train_loss': running / seen, 'val_loss': val['loss'],
                        'val_accuracy': val['accuracy'], 'val_f1': val['f1']})
        if val['f1'] > best_f1:                             # model selection on validation only
            best_f1, best_epoch = val['f1'], epoch
            torch.save(model.state_dict(), checkpoint_path)
    seconds = time.perf_counter() - start

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, pd.DataFrame(history), best_epoch, seconds


def load_checkpoint(checkpoint_path, vocab_size, model_config, variant_config, device):
    model = build_model(vocab_size, model_config, variant_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()
