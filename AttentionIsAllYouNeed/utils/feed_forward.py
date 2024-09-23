import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN) used in
    Transformer models. This consists of two linear transformations
    with a ReLU activation function applied in between.

    Args:
        d_model (int): The dimensionality of the input and output (e.g., 512).
        d_ff (int): The dimensionality of the inner layer (e.g., 2048).
        dropout (float): Dropout rate to prevent overfitting. Default is 0.1.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # First linear layer: transforms input from d_model to d_ff dimensions
        self.fc1 = nn.Linear(d_model, d_ff)

        # Second linear layer: transforms back from d_ff to d_model dimensions
        self.fc2 = nn.Linear(d_ff, d_model)

        # Dropout layer to randomly drop elements during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Defines the forward pass of the Position-wise FFN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            Tensor: Output tensor of the same shape as input (batch_size, sequence_length, d_model).
        """
        # Apply the first linear transformation, followed by ReLU activation
        x = self.fc1(x).relu()

        # Apply dropout and the second linear transformation
        x = self.fc2(self.dropout(x))

        return x

