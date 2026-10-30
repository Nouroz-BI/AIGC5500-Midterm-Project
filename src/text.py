'''Text tokenization and vocabulary utilities.'''

import re
from collections import Counter
from collections.abc import Iterable
import torch
from torch.utils.data import Dataset, DataLoader

# Define the padding and unknown tokens
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
# Define the padding and unknown IDs
PAD_ID = 0
# Define the unknown ID
UNK_ID = 1

# Define the tokenize function
def tokenize(text: str) -> list[str]:
    '''Split lowercase text into words and individual punctuation marks.'''
    return re.findall(r'\w+|[^\w\s]', text.lower())

# Define the build_vocabulary function
def build_vocabulary(texts: Iterable[str], max_vocab_size: int = 10_000, min_frequency: int = 2) -> dict[str, int]:
    '''Build a vocabulary from training text only.'''
    # Check if the maximum vocabulary size is at least 2
    if max_vocab_size < 2:
        raise ValueError('max_vocab_size must include two reserved tokens.')

    # Check if the minimum frequency is at least 1
    if min_frequency < 1:
        raise ValueError('min_frequency must be at least 1.')

    # Create a counter for the text frequencies
    counts = Counter()

    # Update the counter for the text frequencies
    for text in texts:
        counts.update(tokenize(text))

    # Sort the tokens by frequency and alphabetically
    ranked_tokens = sorted(((token, count) for token, count in counts.items() if count >= min_frequency), key = lambda item: (-item[1], item[0]))

    # Create a dictionary for the vocabulary
    vocabulary = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}

    # Add the tokens to the vocabulary
    for token, _ in ranked_tokens:
        if token in vocabulary:
            continue

        # Check if the vocabulary size is at least the maximum vocabulary size
        if len(vocabulary) >= max_vocab_size:
            break

        # Add the token to the vocabulary
        vocabulary[token] = len(vocabulary)

    # Return the vocabulary
    return vocabulary

# Define the encode_text function
def encode_text(text: str, vocabulary: dict[str, int], max_length: int) -> list[int]:
    '''Convert text to token IDs and truncate to the shared limit.'''
    # Check if the maximum length is at least 1
    if max_length < 1:
        raise ValueError('max_length must be at least 1.')

    # Tokenize the text and truncate to the maximum length
    tokens = tokenize(text)[:max_length]
    # Get the token IDs
    token_ids = [vocabulary.get(token, UNK_ID) for token in tokens]

    # Ensure every sequence contains at least one token
    return token_ids or [UNK_ID]

# Define the EncodedTextDataset class
class EncodedTextDataset(Dataset):
    '''Store encoded emails and labels as CPU tensors.'''

    def __init__(self, texts: list[str], labels: list[int], vocabulary: dict[str, int], max_length: int):
        # Convert the texts and labels to lists
        texts = list(texts)
        labels = list(labels)

        # Check if the texts and labels have equal lengths
        if len(texts) != len(labels):
            raise ValueError('Texts and labels must have equal lengths.')

        # Encode the texts
        self.sequences = [torch.tensor(encode_text(text, vocabulary, max_length), dtype = torch.long) for text in texts]

        # Convert the labels to a tensor
        self.labels = torch.tensor(labels, dtype = torch.long)

    def __len__(self):
        # Return the length of the labels
        return len(self.labels)

    def __getitem__(self, index):
        # Return the sequence and label at the given index
        return self.sequences[index], self.labels[index]


# Define the collate_text_batch function
def collate_text_batch(batch: list[tuple[list[int], int]]) -> dict[str, torch.Tensor]:
    '''Right-pad a batch and return tokens, lengths, and labels.'''
    # Unzip the batch into sequences and labels
    sequences, labels = zip(*batch)

    # Create a tensor for the lengths
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype = torch.long)

    # Pad the sequences
    tokens = torch.nn.utils.rnn.pad_sequence(sequences, batch_first = True, padding_value = PAD_ID)

    # Return the tokens, lengths, and labels
    return {'tokens': tokens, 'lengths': lengths, 'labels': torch.stack(labels)}

# Define the make_data_loader function
def make_data_loader(dataset: EncodedTextDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    '''Create a loader with independently seeded batch ordering.'''
    
    # Create a generator with the given seed
    generator = torch.Generator().manual_seed(seed)
    
    # Return the data loader
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_text_batch, generator = generator, num_workers = 0)

# Define the measure_token_lengths function
def measure_token_lengths(texts: Iterable[str]) -> list[int]:
    '''Measure token counts before truncation.'''
    # Return the token lengths
    return [len(tokenize(text)) for text in texts]