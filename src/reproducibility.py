"""Utilities for controlling experiment randomness."""

import random
import numpy as np
import torch

# Set seed for reproducibility
def set_seed(seed: int) -> None:
    """Reset random generators before initializing an experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True