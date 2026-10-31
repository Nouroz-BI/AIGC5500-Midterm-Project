'''Shared training procedure for text classifiers.'''

# Import required libraries
import time
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from src.evaluation import evaluate_model
from src.reproducibility import set_seed
from src.text import make_data_loader

# Define the function to train one epoch
def train_one_epoch(model, data_loader, optimizer, device, max_grad_norm = 1.0):
    '''Update model weights for one pass through training data.'''
    # Set the model to training mode
    model.train()
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Initialize the total loss and the total number of examples
    total_loss = 0.0
    total_examples = 0
    
    # Iterate over the data loader
    for batch in data_loader:
        # Get the tokens, labels, and lengths from the batch
        tokens = batch['tokens'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']

        # Zero the gradients
        optimizer.zero_grad(set_to_none = True)

        # Forward pass
        logits = model(tokens, lengths)
        loss = criterion(logits, labels)

        # Check if the loss is finite
        if not torch.isfinite(loss).item():
            raise ValueError('Training loss is not finite.')

        # Backward pass
        loss.backward()

        # Clip the gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm, error_if_nonfinite = True)

        # Update the model parameters
        optimizer.step()

        # Get the batch size
        batch_size = labels.shape[0]
        # Update the total loss and the total number of examples
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    # Check if the total number of examples is zero
    if total_examples == 0:
        # Raise an error if the total number of examples is zero
        raise ValueError('Cannot train on an empty data loader.')

    # Return the average loss
    return total_loss / total_examples


def fit_model(model_factory, train_dataset, validation_loader, device, seed, epochs, learning_rate, weight_decay, max_grad_norm, batch_size, checkpoint_path, history_path, selection_metric='macro_f1'):
    '''Train a fresh model and restore its best validation checkpoint.'''
    # Check if the number of epochs is at least 1
    '''Train a fresh model and restore its best validation checkpoint.'''
    # Check if the number of epochs is at least 1
    if epochs < 1:
        # Raise an error if the number of epochs is less than 1
        raise ValueError('epochs must be at least 1.')

    # Define the allowed metrics
    allowed_metrics = {'accuracy', 'macro_f1', 'spam_f1', 'spam_precision', 'spam_recall'}
    # Check if the selection metric is in the allowed metrics
    if selection_metric not in allowed_metrics:
        # Raise an error if the selection metric is not in the allowed metrics
        raise ValueError(f'selection_metric must be one of {sorted(allowed_metrics)}.')

    # Convert the checkpoint and history paths to Path objects
    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)

    # Avoid silently replacing evidence from an earlier run.
    if checkpoint_path.exists() or history_path.exists():
        # Raise an error if the checkpoint and history paths already exist
        raise FileExistsError('Run outputs already exist. Use a new run name or explicitly archive the previous outputs.')

    # Create the parent directories for the checkpoint and history paths
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset randomness BEFORE constructing the model and loader
    set_seed(seed)
    # Construct the model
    model = model_factory().to(device)

    # Construct the training data loader
    train_loader = make_data_loader(train_dataset, batch_size = batch_size, shuffle = True, seed = seed)

    # Construct the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    # Initialize the best score and the best epoch
    best_score = float('-inf')
    best_epoch = None
    history_rows = []

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    # Iterate over the epochs
    for epoch in range(1, epochs + 1):
        # Train the model for one epoch
        train_loss = train_one_epoch( model, train_loader, optimizer, device, max_grad_norm = max_grad_norm)
        # Evaluate the model on the validation set
        validation = evaluate_model(model, validation_loader, device)
        # Create a row for the history
        row = {'epoch': epoch, 'seed': seed, 'learning_rate': learning_rate, 'train_loss': train_loss, **{f'val_{name}': value for name, value in validation.items()}}
        history_rows.append(row)
        # Get the score for the validation set
        score = validation[selection_metric]

        # Strict improvement preserves the earliest epoch on ties.
        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'seed': seed,
                    'selection_metric': selection_metric,
                    'validation_metrics': validation,
                    'learning_rate': learning_rate,
                    'epochs_budget': epochs,
                    'weight_decay': weight_decay,
                    'max_grad_norm': max_grad_norm,
                    'batch_size': batch_size,
                },
                checkpoint_path,
            )

        # Preserve every completed epoch, even if a later epoch fails.
        pd.DataFrame(history_rows).to_csv(
            history_path,
            index=False,
        )

        # Print the epoch, train loss, validation loss, and validation score
        print(f'Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | val loss {validation['loss']:.4f} | val {selection_metric} {score:.4f}')

    # Synchronize the device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    elapsed_seconds = time.perf_counter() - start
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = True)
    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    # Set the model to evaluation mode
    model.eval()

    return {
        'model': model,
        'history': pd.DataFrame(history_rows),
        'best_epoch': best_epoch,
        'best_validation_metrics': checkpoint['validation_metrics'],
        'training_seconds': elapsed_seconds,
    }