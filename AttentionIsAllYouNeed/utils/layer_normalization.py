import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    A custom implementation of Layer Normalization.
    This normalizes the input data along the last dimension (usually features)
    by subtracting the mean and dividing by the standard deviation for each sample.

    Attributes:
    ----------
    features : int
        The number of features in the input data. Used to define the size of the learnable parameters.
    epsilon : float
        A small constant to avoid division by zero during standard deviation calculation.
    alpha : torch.nn.Parameter
        A learnable scaling factor (initialized as ones) applied after normalization.
    bias : torch.nn.Parameter
        A learnable shift factor (initialized as zeros) added after normalization.
    """

    def __init__(self, features: int, epsilon: float = 1e-6):
        """
        Initializes the LayerNormalization class with the number of features and an optional epsilon value.

        Parameters:
        ----------
        features : int
            Number of features in the input data, used to define the size of `alpha` and `bias` parameters.
        epsilon : float, optional
            A small value to ensure numerical stability, default is 1e-6.
        """
        super(LayerNormalization, self).__init__()
        # Learnable scale parameter (alpha), initialized to ones, shape: (features,)
        self.alpha = nn.Parameter(torch.ones(features))
        # Learnable bias parameter, initialized to zeros, shape: (features,)
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Layer Normalization.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., num_features),
            where normalization is applied across the last dimension.

        Returns:
        -------
        torch.Tensor
            The normalized tensor with the same shape as the input, scaled by `alpha` and shifted by `bias`.
        """
        # Compute the mean across the last dimension (features) and keep the dimensions
        mean = x.mean(dim=-1, keepdim=True)
        # Compute the standard deviation across the last dimension (features) and keep the dimensions
        std = x.std(dim=-1, keepdim=True)
        # Normalize the input, then scale by alpha and shift by bias
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias
