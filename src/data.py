'''Dataset loading utilities.'''
import json
from pathlib import Path
from datasets import DatasetDict, load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_from_disk

# Load the Enron spam dataset without modifying its source splits
def load_enron_dataset(cache_dir: str | Path, revision: str = 'main') -> DatasetDict:
    '''Load the Enron spam dataset without modifying its source splits.'''
    return load_dataset('SetFit/enron_spam', cache_dir = str(cache_dir), revision = revision)

# Normalize duplicate key
def normalize_duplicate_key(text: str) -> str:
    '''Normalize case and whitespace for duplicate detection only.'''
    return ' '.join(text.lower().split())

# Audit the dataset for quality and integrity
def audit_dataset(raw_dataset: DatasetDict) -> tuple[pd.DataFrame, int]:
    '''Report text quality and train/test overlap without modifying data.'''
    summaries = []
    keys_by_split = {}

    for split_name, dataset in raw_dataset.items():
        # Read only the columns required for these checks.
        frame = dataset.select_columns(['text', 'label']).to_pandas()
        # Check if the text is a string
        valid_text = frame['text'].map(lambda value: isinstance(value, str))
        # Normalize the text for duplicate detection
        keys = frame['text'].map(lambda value: normalize_duplicate_key(value) if isinstance(value, str) else None)
        # Check if the text is not empty
        nonempty = valid_text & keys.ne('')
        valid_keys = keys.loc[nonempty]

        # Create a dataframe with the valid keys and labels
        labelled_keys = pd.DataFrame({
            'key': valid_keys,
            'label': frame.loc[nonempty, 'label']
            })

        # Count the number of unique labels for each key
        label_counts = labelled_keys.groupby('key')['label'].nunique()
        # Check if there are any conflicting label groups
        conflicting_groups = int(label_counts.gt(1).sum())

        # Create a summary of the dataset
        summaries.append({
            'split': split_name,
            'rows': len(frame),
            'missing_or_nonstring_text': int((~valid_text).sum()),
            'empty_text': int((valid_text & keys.eq('')).sum()),
            'invalid_labels': int((~frame['label'].isin([0, 1])).sum()),
            'duplicate_extra_rows': int(valid_keys.duplicated().sum()),
            'conflicting_label_groups': conflicting_groups
            })

        # Add the valid keys to the dictionary of keys by split
        keys_by_split[split_name] = set(valid_keys)

    # Check if there are any shared keys between the train and test splits
    shared_keys = keys_by_split['train'] & keys_by_split['test']
    # Return the summaries and the number of shared keys
    return pd.DataFrame(summaries), len(shared_keys)

# Prepare splits for training and evaluation
def prepare_splits(raw_dataset: DatasetDict, split_seed: int = 42, validation_fraction: float = 0.2) -> tuple[DatasetDict, pd.DataFrame]:
    '''Clean source partitions and create disjoint experimental splits.'''
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1.')

    # Clean the source partitions
    cleaned = {}
    # Record the cleaning process
    cleaning_records = []

    for split_name in ['train', 'test']:
        # Convert the dataset to a pandas dataframe
        frame = raw_dataset[split_name].to_pandas()
        # Get the original number of rows
        original_rows = len(frame)
        # Check if the text is a string and not empty
        valid_text = frame['text'].map(lambda value: isinstance(value, str) and bool(value.strip()))
        # Check if the label is valid
        valid_label = frame['label'].isin([0, 1])

        # Drop rows with invalid text or label
        frame = frame.loc[valid_text & valid_label].copy()
        # Get the number of valid rows
        valid_rows = len(frame)

        # Normalize the text for duplicate detection
        frame['_text_key'] = frame['text'].map(normalize_duplicate_key)

        # Count the number of unique labels for each key
        label_counts = frame.groupby('_text_key')['label'].nunique()
        # Check if there are any conflicting labels
        if label_counts.gt(1).any():
            raise ValueError(f'Conflicting labels found within {split_name}.')

        # Drop duplicate rows
        frame = frame.drop_duplicates('_text_key', keep='first')

        # Record the cleaning process
        cleaning_records.append({'source_split': split_name, 'original_rows': original_rows, 'invalid_or_empty_removed': original_rows - valid_rows, 'duplicate_rows_removed': valid_rows - len(frame)})

        cleaned[split_name] = frame

    # Get the training and test sets
    train_frame = cleaned['train']
    test_frame = cleaned['test']

    # Check if there is any text overlap between the training and test sets
    overlap = train_frame['_text_key'].isin(test_frame['_text_key'])
    # Get the number of rows removed due to overlap
    overlap_removed = int(overlap.sum())
    # Remove the overlapping rows from the training set
    train_frame = train_frame.loc[~overlap].copy()

    # Split the training set into training and validation sets
    train_frame, validation_frame = train_test_split(train_frame, test_size = validation_fraction, random_state = split_seed, stratify = train_frame['label'])

    # Create a dictionary of the splits
    frames = {'train': train_frame, 'validation': validation_frame, 'test': test_frame}

    # Create a dictionary of the keys for each split
    keys = {name: set(frame['_text_key']) for name, frame in frames.items()}

    # Check if there is any text overlap between the splits
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if keys[left] & keys[right]:
            raise ValueError(f'Text overlap between {left} and {right}.')

    # Create a DatasetDict with the cleaned splits
    splits = DatasetDict({name: Dataset.from_pandas(frame.drop(columns = '_text_key').reset_index(drop = True), preserve_index = False) for name, frame in frames.items()})

    # Create a summary of the cleaning process
    cleaning_summary = pd.DataFrame(cleaning_records)
    cleaning_summary['train_test_overlap_removed'] = [overlap_removed, 0]
    
    # Return the splits and the cleaning summary
    return splits, cleaning_summary

# Save the prepared data
def save_prepared_data(splits: DatasetDict, vocabulary: dict[str, int], metadata: dict, output_dir: str | Path) -> None:
    '''Save prepared partitions, vocabulary, and their configuration.'''
    output_dir = Path(output_dir)
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents = True, exist_ok = True)
    # Save the splits
    splits.save_to_disk(str(output_dir / 'splits'))

    # Save the vocabulary
    with (output_dir / 'vocabulary.json').open('w', encoding = 'utf-8') as file:
        json.dump(vocabulary, file, ensure_ascii = False, indent = 2, sort_keys = True)

    # Save the metadata
    with (output_dir / 'metadata.json').open('w', encoding = 'utf-8') as file:
        json.dump(metadata, file, indent = 2, sort_keys = True)

# Load the prepared data
def load_prepared_data(input_dir: str | Path) -> dict:
    '''Load the saved partitions, vocabulary, and configuration.'''
    input_dir = Path(input_dir)

    # Check if the required paths exist
    required_paths = [input_dir / 'splits', input_dir / 'vocabulary.json', input_dir / 'metadata.json']

    # If the required paths do not exist, raise an error
    if not all(path.exists() for path in required_paths):
        raise FileNotFoundError('Prepared data is missing. Run notebooks/data_preparation.ipynb first.')

    # Load the splits
    splits = load_from_disk(str(input_dir / 'splits'))

    # Load the vocabulary
    with (input_dir / 'vocabulary.json').open(encoding = 'utf-8') as file:
        vocabulary = json.load(file)

    # Load the metadata
    with (input_dir / 'metadata.json').open(encoding = 'utf-8') as file:
        metadata = json.load(file)

    # Return the splits, vocabulary, and metadata
    return {'splits': splits, 'vocabulary': vocabulary, 'metadata': metadata}