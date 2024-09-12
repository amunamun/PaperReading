import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    A PyTorch module that performs input embeddings for a sequence model.

    This module maps input tokens (usually indices) to dense vectors of a specified dimension.
    The embeddings are scaled by the square root of the model dimension to adjust their magnitude,
    which can be beneficial for training stability and convergence.

    Args:
        d_model (int): The dimension of the embedding vectors.
        vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens that can be embedded.

    Attributes:
        embedding (nn.Embedding): A PyTorch embedding layer that transforms token indices to dense vectors.

    Methods:
        forward(x): Computes the embedded representation of the input tokens, scaled by the square root of d_model.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initializes the InputEmbeddings module.

        Args:
            d_model (int): The dimension of the embedding vectors.
            vocab_size (int): The number of unique tokens in the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Create an embedding layer that maps vocabulary indices to dense vectors of dimension d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        """
        Forward pass through the embedding layer.

        This method takes a batch of token indices and returns the corresponding embeddings.
        Each embedding is scaled by the square root of d_model to ensure that the scale of the embeddings is
        consistent with the model's expected input size.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing token indices.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, d_model) containing the scaled embeddings.
        """
        # Compute the embeddings and scale them
        return self.embedding(x) * math.sqrt(self.d_model)
