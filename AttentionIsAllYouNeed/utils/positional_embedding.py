import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    This module adds positional encodings to input tensors to provide information
    about the position of elements within a sequence. This is crucial for models
    like Transformers that do not inherently process sequence order.

    The positional encoding is based on sinusoidal functions introduced in the
    paper "Attention is All You Need" by Vaswani et al. (2017). The idea is to encode
    the position of each element in a sequence using both sine and cosine functions,
    which allows the model to distinguish between different positions in the sequence.

    Parameters:
    - d_model (int): The dimension of the model (i.e., the size of the input embeddings).
    - seq_len (int): The length of the input sequences (i.e., the maximum sequence length).
    - dropout (float): Dropout rate to apply to the positional encodings to prevent overfitting.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Initialize positional encoding matrix
        self.positional_encodings = self._create_positional_encodings(seq_len, d_model)

        # Register positional encodings as a buffer to ensure they are not updated during training
        self.register_buffer('pe', self.positional_encodings.unsqueeze(0))

    def _create_positional_encodings(self, seq_len: int, d_model: int) -> torch.Tensor:
        """
        Create the positional encoding matrix using sinusoidal functions.

        The positional encoding is defined as follows:
        - Each position is encoded using a combination of sine and cosine functions
          with different frequencies.
        - For position `pos` and dimension `i`, the encoding is computed as:
          PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        where `i` is the dimension index and `d_model` is the dimensionality of the model.

        This approach allows the model to learn relative positions by combining sine and cosine
        functions, ensuring that each position has a unique encoding.

        Parameters:
        - seq_len (int): Length of the sequences.
        - d_model (int): Dimension of the model.

        Returns:
        - torch.Tensor: A tensor of shape (seq_len, d_model) containing the positional encodings.
        """
        # Initialize positional encoding matrix with zeros
        positional_encodings = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the scaling factor for sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        return positional_encodings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input tensor and apply dropout.

        The positional encodings are added to the input tensor to incorporate information
        about the position of each element in the sequence. Dropout is then applied to
        the combined tensor to help with regularization.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        - torch.Tensor: The input tensor with positional encodings added and dropout applied.
        """
        # Add positional encodings to the input tensor
        x = x + self.pe[:, :x.size(1), :].detach()

        # Apply dropout to the tensor
        return self.dropout(x)
