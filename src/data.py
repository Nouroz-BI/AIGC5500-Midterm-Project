'''Dataset loading utilities.'''

from pathlib import Path
from datasets import DatasetDict, load_dataset

# Load the Enron spam dataset without modifying its source splits
def load_enron_dataset(cache_dir: str | Path, revision: str = 'main',) -> DatasetDict:
    '''Load the Enron spam dataset without modifying its source splits.'''
    return load_dataset('SetFit/enron_spam', cache_dir = str(cache_dir), revision = revision)