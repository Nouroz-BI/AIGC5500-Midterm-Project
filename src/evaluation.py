'''Shared classification metrics and evaluation.'''

import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Calculate classification metrics
def classification_metrics(y_true, y_pred) -> dict[str, float]:
    '''Calculate classification metrics; spam is class 1.'''
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels = [0, 1], average='macro',zero_division=0)),
        'spam_f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'spam_precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'spam_recall': float(recall_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> dict[str, float]:
    '''Calculate loss and metrics without updating model weights.'''
    previous_mode = model.training
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='sum')

    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_predictions = []

    try:
        with torch.inference_mode():
            for batch in data_loader:
                tokens = batch['tokens'].to(device)
                labels = batch['labels'].to(device)
                lengths = batch['lengths']

                logits = model(tokens, lengths)

                if logits.shape != (labels.shape[0], 2):
                    raise ValueError(
                        'Expected logits shaped [batch_size, 2].'
                    )

                if not torch.isfinite(logits).all().item():
                    raise ValueError('Model produced non-finite scores.')

                loss = criterion(logits, labels)

                if not torch.isfinite(loss).item():
                    raise ValueError('Evaluation loss is not finite.')

                predictions = logits.argmax(dim=1)

                total_loss += loss.item()
                total_examples += labels.shape[0]

                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())
    finally:
        model.train(previous_mode)

    if total_examples == 0:
        raise ValueError('Cannot evaluate an empty data loader.')

    metrics = classification_metrics(all_labels, all_predictions)
    metrics['loss'] = total_loss / total_examples

    return metrics