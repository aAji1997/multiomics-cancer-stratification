import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """
    A standard Transformer encoder block.
    Includes Multi-Head Self-Attention, Add & Norm, Feed-Forward, Add & Norm.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimensionality of the input and output embeddings.
            num_heads: Number of attention heads.
            ff_dim: Hidden dimension of the feed-forward network.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        # Multi-Head Self-Attention
        # (seqlen, batch, dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)

        # Feed-Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x: Input tensor of shape (SeqLen, Batch, Dim).
            mask: Optional mask for the attention mechanism.

        Returns:
            Output tensor of shape (SeqLen, Batch, Dim).
        """
        # 1. Multi-Head Self-Attention
        # Attention expects query, key, value
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask, need_weights=False)
        # Add & Norm after attention
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Feed-Forward Network
        ff_output = self.feed_forward(x)
        # Add & Norm after feed-forward
        x = self.norm2(x + self.dropout(ff_output))

        return x 