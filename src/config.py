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

# Validation fraction
VALIDATION_FRACTION = 0.2