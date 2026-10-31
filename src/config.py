'''Shared configuration for the Enron architecture comparison.'''

# Dataset
DATASET_ID = 'SetFit/enron_spam'
DATASET_REVISION = 'main'  # Pin to an exact commit before final runs.

# Reproducibility
SPLIT_SEED = 42
TRAINING_SEEDS = (42, 123, 2026)

# Data preparation
VALIDATION_FRACTION = 0.2
MAX_VOCAB_SIZE = 10_000
MIN_FREQUENCY = 2
MAX_LENGTH = 256

# Batching
BATCH_SIZE = 64

# Shared training protocol
EPOCHS = 10
LEARNING_RATES = (1e-3,)
TUNING_SEED = 42
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SELECTION_METRIC = "spam_f1"