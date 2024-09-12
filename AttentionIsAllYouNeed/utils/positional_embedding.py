import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements positional encodings to add information about sequence order to input tensors.

    In models like Transformers, which lack an inherent sense of order in sequences (unlike RNNs or CNNs),
    positional encodings provide a way to inject information about the relative positions of tokens
    in a sequence. This helps the model differentiate between tokens based on their order, which is
    essential for processing sequential data.

    The encoding is based on sinusoidal functions that vary across dimensions and positions.
    The method is introduced in "Attention is All You Need" by Vaswani et al. (2017), and
    it allows the model to learn the relative positions in a sequence via a fixed encoding.

    Attributes:
    - d_model (int): Dimensionality of the input embeddings (i.e., the size of the feature vector for each token).
    - seq_len (int): Maximum length of input sequences. The positional encodings are precomputed for this length.
    - dropout (nn.Dropout): Dropout layer applied to the encoded input to prevent overfitting during training.

    Methods:
    - _create_positional_encodings: Generates the positional encoding matrix based on sinusoidal functions.
    - forward: Adds positional encodings to the input tensor and applies dropout for regularization.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        """
        Initializes the PositionalEncoding module.

        Args:
        - d_model (int): The dimensionality of the input embeddings.
        - max_seq_len (int): The maximum length of the input sequences.
        - dropout (float): The dropout rate to apply for regularization.

        The positional encodings are precomputed and stored as a buffer that is not updated
        during backpropagation. The dropout layer is applied after adding positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Initialize and store positional encodings
        self.positional_encodings = self._create_positional_encodings(max_seq_len, d_model)

        # Register positional encodings as a buffer so they are not trainable
        self.register_buffer('pe', self.positional_encodings.unsqueeze(0))

    def __create_positional_encodings(self) -> torch.Tensor:
        """
        Generates a matrix of positional encodings using sine and cosine functions.

        The encoding is computed as:
        - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Where `pos` is the position in the sequence and `i` is the dimension index.
        The alternating sine and cosine functions allow each dimension to capture
        different frequencies, which helps the model recognize different relative
        positions between tokens.

        Returns:
        - torch.Tensor: A tensor of shape (max_seq_len, d_model) containing the positional encodings.
        """
        # Initialize positional encoding matrix with zeros
        positional_encodings = torch.zeros(self.max_seq_len, self.d_model)

        # Compute position indices and scaling factors for sinusoidal functions
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        # Apply sine to even indices (2i) and cosine to odd indices (2i+1)
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        return positional_encodings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input tensor and applies dropout.

        The positional encodings are added to the input tensor `x`, which should have the shape
        (batch_size, seq_len, d_model). The positional encodings provide information about the
        relative positions of tokens within the sequence. After adding the positional encodings,
        dropout is applied for regularization.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        - torch.Tensor: The input tensor with positional encodings added, followed by dropout.
        """
        # Add positional encodings to the input
        x = x + self.pe[:, :x.size(1), :].detach()

        # Apply dropout and return the modified tensor
        return self.dropout(x)
