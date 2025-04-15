import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encoding information.
    The input tensor is expected to be of shape (SeqLen, Batch, Dim).
    From "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: The embedding dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length supported.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape of div_term: (d_model / 2)

        pe = torch.zeros(max_len, 1, d_model) # (max_len, 1, d_model)
        # Apply sin to even indices in the array; 2i
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices in the array; 2i+1
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer that should not be considered a model parameter.
        # Buffers are tensors saved and restored in state_dict, but not trained by the optimizer.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [SeqLen, Batch, Dim]

        Returns:
            Tensor with positional encoding added, shape [SeqLen, Batch, Dim]
        """
        # x.size(0) is the sequence length (number of features/genes in our case)
        # self.pe is (max_len, 1, d_model). We select up to the input sequence length.
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 