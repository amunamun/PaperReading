import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Embeds input tokens into dense vectors for use in a Transformer model, scaling the embeddings
    by the square root of the model dimension.

    In Transformers, the input tokens (words, subwords, etc.) must first be converted to continuous
    vector representations, known as embeddings, before being processed by the model. These embeddings
    are learned during training and capture the semantic relationships between tokens. By scaling the
    embeddings with the square root of the model dimension (`d_model`), the model ensures that the
    variance of these embeddings remains consistent, which can stabilize training.

    This class specifically handles the initial input embedding step in the Transformer, where token
    indices are mapped to dense vectors, which are then scaled before being passed to the rest of the model.

    Args:
        d_model (int): The dimension of the embedding vectors (i.e., the size of the continuous vector
            that each token is mapped to).
        vocab_size (int): The size of the vocabulary (i.e., the number of unique tokens that can be embedded).

    Attributes:
        embedding (nn.Embedding): A PyTorch embedding layer that maps token indices to dense vectors.
        scale (float): A scaling factor equal to the square root of `d_model`, used to adjust the magnitude
            of the embeddings.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initializes the embedding layer and computes the scaling factor for the embeddings.

        The embedding layer maps token indices to dense vectors. The scaling factor ensures that the
        embeddings are appropriately scaled in relation to the model's dimensionality. Without this scaling,
        the magnitude of the embeddings could grow too large and lead to instability during training.

        Args:
            d_model (int): The dimension of the embedding vectors.
            vocab_size (int): The size of the vocabulary (number of unique tokens).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """
        Performs the forward pass, computing the embeddings for the input tokens and scaling them.

        Each token index in the input tensor is mapped to a dense vector via the embedding layer.
        The resulting embeddings are then scaled by the square root of `d_model` to prevent the
        embeddings' magnitudes from growing too large. This is critical because without scaling, the
        variance of the embeddings would increase with `d_model`, which could destabilize the model
        and make training slower or more difficult.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing token indices.
                Each index represents a token from the vocabulary.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, d_model) containing the scaled embeddings,
                which are ready to be input into the subsequent layers of the Transformer model.
        """
        return self.embedding(x) * self.scale
