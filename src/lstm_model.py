'''
LSTM model for the Enron email spam classification experiment
'''

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.text import PAD_ID

class LSTMClassifier(nn.Module):
    '''Classify emails using an LSTM model.'''
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1, forget_bias: float = 0.0):
        super().__init__()
        
        # Learn one vector per each vocabulary token
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_dim, padding_idx = PAD_ID)
        
        # Read each email in its original token order
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim, num_layers = 1, batch_first = True, bidirectional = False)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(p = dropout)
        
        # Define the classifier layer
        self.classifier = nn.Linear(in_features = hidden_dim, out_features = 2)
        
        # Explicit initialization of the LSTM supports a controlled bias ablation
        with torch.no_grad():
            self.lstm.bias_ih_l0.zero_()
            self.lstm.bias_hh_l0.zero_()
            
            # Gate order: input, forget, candidate, output
            self.lstm.bias_ih_l0[hidden_dim:2 * hidden_dim].fill_(forget_bias)

    # Define the forward pass
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the LSTM model.'''
        
        # Convert the tokens to embeddings [batch, sequence_length] -> [batch, sequence_length, embed_dim]
        embeddings = self.embedding(tokens)

        # Pack the embeddings for efficient processing and exclude padding from recurrent processing
        packed = pack_padded_sequence(embeddings, lengths.detach().cpu(), batch_first = True, enforce_sorted = False)

        # Each email has a final hidden state and a final cell state
        _, (hidden_state, _) = self.lstm(packed)

        # Single layer, single direction: [1, batch, hidden_dim]
        email_representation = hidden_state[-1]

        # Return raw scores shaped [batch, 2] for binary classification
        return self.classifier(self.dropout(email_representation))