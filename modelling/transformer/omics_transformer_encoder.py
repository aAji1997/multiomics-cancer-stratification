# modelling/transformer/omics_transformer_encoder.py
import torch
import torch.nn as nn
from typing import Dict

# Make sure these imports point to the correct files relative to this one
from .transformer_block import TransformerBlock
# Removing positional encoding import as features are no longer treated as sequence
# from .positional_encoding import PositionalEncoding

class OmicsTransformerEncoder(nn.Module):
    """
    Encodes multiple omics types using per-omics projection followed by
    Transformer blocks for inter-omics integration.

    1. Projects each omics type's feature vector to a common dimension (`embed_dim`).
    2. Stacks these projections, creating a sequence of omics embeddings per patient.
    3. Adds learned omics-type embeddings (acting like positional encoding).
    4. Processes this sequence through Transformer blocks to learn inter-omics relationships.
    5. Aggregates the output sequence (e.g., mean pooling) per patient.
    6. Passes the aggregated vector through a final MLP for the output patient embedding.
    """
    def __init__(self,
                 omics_input_dims: Dict[str, int],
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 num_transformer_layers: int,
                 output_dim: int,
                 dropout: float = 0.1
                 # max_len is removed as it was for feature positional encoding
                 ):
        """
        Args:
            omics_input_dims: Dictionary mapping omics type name (str) to its number of input features (int).
            embed_dim: The core embedding dimension for projections and transformer.
            num_heads: Number of attention heads in each TransformerBlock.
            ff_dim: Hidden dimension of the feed-forward networks within TransformerBlocks.
            num_transformer_layers: Number of TransformerBlocks to stack.
            output_dim: The final output dimension for the patient embeddings.
            dropout: Dropout rate used throughout the model.
        """
        super().__init__()
        self.omics_input_dims = omics_input_dims
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.omics_types = sorted(omics_input_dims.keys()) # Consistent order
        self.num_omics_types = len(self.omics_types)

        # --- Per-Omics Input Projection ---
        self.projection_layers = nn.ModuleDict()
        for omics_type, input_dim in omics_input_dims.items():
            self.projection_layers[omics_type] = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # --- Omics Type Embeddings (like positional encoding for omics types) ---
        self.omics_type_embeddings = nn.Embedding(self.num_omics_types, embed_dim)
        self.layer_norm_input = nn.LayerNorm(embed_dim) # Normalize after adding type embedding
        self.dropout_input = nn.Dropout(dropout)

        # --- Transformer Blocks ---
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_transformer_layers)]
        )

        # --- Output Aggregation & MLP ---
        self.final_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim), # Input is the aggregated embedding
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, output_dim)
        )

    def forward(self, raw_omics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            raw_omics_data: Dictionary mapping omics type name (str) to
                             input feature tensor (torch.Tensor) of shape
                             (Batch, NumFeatures_omic).

        Returns:
            Integrated patient embeddings tensor of shape (Batch, output_dim).
        """
        batch_size = -1
        projected_embeddings = []

        for i, omics_type in enumerate(self.omics_types):
            if omics_type not in raw_omics_data:
                raise ValueError(f"Missing omics type '{omics_type}' in input data.")

            x = raw_omics_data[omics_type] # Shape: (Batch, NumFeatures_omic)
            current_batch_size = x.shape[0]

            if batch_size == -1:
                batch_size = current_batch_size
            elif batch_size != current_batch_size:
                raise ValueError(f"Batch size mismatch ({omics_type} has {current_batch_size}, expected {batch_size}).")

            if x.shape[1] != self.omics_input_dims[omics_type]:
                 raise ValueError(f"Input dimension mismatch for {omics_type}. "
                                  f"Expected {self.omics_input_dims[omics_type]}, got {x.shape[1]}.")

            # 1. Project each omics feature vector
            projected = self.projection_layers[omics_type](x) # Shape: (Batch, embed_dim)
            projected_embeddings.append(projected)

        # 2. Stack projections to create sequence: (Batch, NumOmicsTypes, embed_dim)
        stacked_embeddings = torch.stack(projected_embeddings, dim=1)

        # 3. Add omics type embeddings
        # Create tensor of type indices: (NumOmicsTypes) -> (1, NumOmicsTypes)
        type_indices = torch.arange(self.num_omics_types, device=stacked_embeddings.device).unsqueeze(0)
        # Expand to match batch size: (Batch, NumOmicsTypes)
        type_indices = type_indices.expand(batch_size, -1)
        # Get embeddings: (Batch, NumOmicsTypes, embed_dim)
        type_embeds = self.omics_type_embeddings(type_indices)

        # Add type embeddings to projected embeddings
        x_combined = stacked_embeddings + type_embeds
        x_combined = self.layer_norm_input(x_combined)
        x_combined = self.dropout_input(x_combined)

        # Permute for Transformer: (NumOmicsTypes, Batch, embed_dim)
        transformer_input = x_combined.permute(1, 0, 2)

        # 4. Pass through Transformer layers
        transformer_output = transformer_input
        for layer in self.transformer_layers:
            # No mask needed here as all patients have all omics types in the sequence
            transformer_output = layer(transformer_output)
        # Shape: (NumOmicsTypes, Batch, embed_dim)

        # 5. Aggregate output tokens (mean pooling over omics types/sequence dim)
        # Shape: (Batch, embed_dim)
        aggregated_embedding = transformer_output.mean(dim=0)

        # 6. Final MLP projection
        # Shape: (Batch, output_dim)
        final_patient_embeddings = self.final_mlp(aggregated_embedding)

        return final_patient_embeddings