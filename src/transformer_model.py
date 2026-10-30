'''Transformer encoder classifier, training loop, and inference-cost measurement.

Written for the AIGC 5500 midterm project. Trained from scratch; no pretrained weights.
Uses standard PyTorch layers (nn.Embedding, nn.TransformerEncoder) as the brief allows.
'''
import platform
import threading
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.reproducibility import set_seed
from src.text import PAD_ID, make_data_loader


class TransformerClassifier(nn.Module):
    '''Pre-norm Transformer encoder with masked mean pooling over real tokens.

    use_positions=False removes the learned position embeddings (the ablation):
    attention then sees an unordered set of words, like a bag of words.
    '''

    def __init__(self, vocab_size, max_length, embed_dim=64, num_heads=4, num_layers=2,
                 ff_dim=128, dropout=0.1, use_positions=True, num_classes=2):
        super().__init__()
        self.max_length = max_length
        self.use_positions = use_positions
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, embed_dim) if use_positions else None
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                           dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim),
                                             enable_nested_tensor=False)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        pad_mask = tokens == PAD_ID                                   # True where padded
        x = self.token_embedding(tokens)
        if self.use_positions:
            positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
        h = self.encoder(self.embedding_dropout(x), src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)                  # ignore padding when pooling
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


def build_model(vocab_size, max_length, model_config, variant_config):
    return TransformerClassifier(vocab_size, max_length, **model_config, **variant_config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, predictions, targets = 0.0, [], []
    for batch in loader:
        tokens, labels = batch['tokens'].to(device), batch['labels'].to(device)
        logits = model(tokens)
        total_loss += F.cross_entropy(logits, labels, reduction='sum').item()
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    predictions, targets = torch.cat(predictions).numpy(), torch.cat(targets).numpy()
    return {'loss': total_loss / len(targets),
            'accuracy': accuracy_score(targets, predictions),
            'f1': f1_score(targets, predictions)}


def train_one_run(model_config, variant_config, seed, train_dataset, validation_loader, vocab_size,
                  max_length, device, train_config, checkpoint_path):
    '''Train one model for one seed. Model selection uses validation F1 only.

    Returns (model with best-validation weights, per-epoch history, best epoch, seconds).
    '''
    set_seed(seed)
    model = build_model(vocab_size, max_length, model_config, variant_config).to(device)
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
        if val['f1'] > best_f1:
            best_f1, best_epoch = val['f1'], epoch
            torch.save(model.state_dict(), checkpoint_path)
    seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, pd.DataFrame(history), best_epoch, seconds


def load_checkpoint(checkpoint_path, vocab_size, max_length, model_config, variant_config, device):
    model = build_model(vocab_size, max_length, model_config, variant_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


# ---------------------------------------------------------------- inference cost
def hardware_description(device):
    if device.type == 'cuda':
        return f'GPU: {torch.cuda.get_device_name(0)}'
    return f'CPU: {platform.processor() or platform.machine()} ({torch.get_num_threads()} threads)'


class PeakMemory:
    '''Peak memory during a block. CUDA: peak allocated tensor memory. CPU: peak process RSS (needs psutil).'''

    def __init__(self, device, interval=0.002):
        self.device, self.interval, self.peak_mb, self.kind = device, interval, float('nan'), ''
        self._halt, self._thread, self._peak_bytes = threading.Event(), None, 0

    def _sample(self):
        import psutil
        process = psutil.Process()
        while not self._halt.is_set():
            self._peak_bytes = max(self._peak_bytes, process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.kind = 'CUDA peak allocated (includes weights)'
        else:
            try:
                import psutil  # noqa: F401
                self._thread = threading.Thread(target=self._sample, daemon=True)
                self._thread.start()
                self.kind = 'CPU peak process RSS (whole Python process)'
            except ImportError:
                self.kind = 'unavailable (pip install psutil)'
        return self

    def __exit__(self, *exc):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / 1e6
        elif self._thread is not None:
            self._halt.set()
            self._thread.join()
            self.peak_mb = self._peak_bytes / 1e6


@torch.no_grad()
def measure_cost(model, dataset, device, num_samples=300, batch_size=64, warmup=20):
    '''Parameters, time per sample (batch size 1 and batched) and peak inference memory.'''
    model = model.to(device).eval()
    num_samples = min(num_samples, len(dataset))
    sequences = [dataset[i][0] for i in range(num_samples)]
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)

    with PeakMemory(device) as memory:
        for seq in sequences[:warmup]:
            model(seq.unsqueeze(0).to(device))
        sync()
        single = []
        for seq in sequences:
            x = seq.unsqueeze(0).to(device)
            sync(); start = time.perf_counter()
            model(x)
            sync(); single.append(time.perf_counter() - start)

        batches = [nn.utils.rnn.pad_sequence(sequences[i:i + batch_size], batch_first=True, padding_value=PAD_ID).to(device)
                   for i in range(0, num_samples, batch_size)]
        sync(); start = time.perf_counter()
        for x in batches:
            model(x)
        sync(); batched_total = time.perf_counter() - start

    single = pd.Series(single) * 1000.0
    return {'parameters': count_parameters(model),
            'ms_per_sample_batch1_mean': single.mean(),
            'ms_per_sample_batch1_median': single.median(),
            f'ms_per_sample_batch{batch_size}': batched_total / num_samples * 1000.0,
            'peak_memory_mb': memory.peak_mb,
            'memory_measure': memory.kind,
            'samples_timed': num_samples,
            'hardware': hardware_description(device)}
