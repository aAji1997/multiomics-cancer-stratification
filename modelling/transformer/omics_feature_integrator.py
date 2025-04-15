import torch
import torch.nn as nn
from typing import Dict, List

class OmicsFeatureIntegrator(nn.Module):
    """
    Integrates features from multiple omics types for each patient.

    Projects each omics type's feature vector to a common dimension,
    concatenates them, and optionally passes through a final MLP.

    This serves as a basic integration module before potentially adding
    more complex Transformer-based interactions.
    """
    def __init__(self,
                 omics_input_dims: Dict[str, int],
                 common_embed_dim: int,
                 final_output_dim: int | None = None, # If None, output is concatenated common embeddings
                 dropout: float = 0.1):
        """
        Args:
            omics_input_dims: Dictionary mapping omics type name (str) to
                              its number of input features (int).
            common_embed_dim: The dimension to project each omics type into.
            final_output_dim: Optional dimension for a final MLP layer applied
                               to the concatenated embeddings. If None, the
                               output is just the concatenation.
            dropout: Dropout rate for the optional final MLP.
        """
        super().__init__()
        self.omics_input_dims = omics_input_dims
        self.common_embed_dim = common_embed_dim
        self.final_output_dim = final_output_dim
        self.dropout_rate = dropout
        self.omics_types = sorted(omics_input_dims.keys()) # Ensure consistent order

        # Input projection layer for each omics type
        self.projection_layers = nn.ModuleDict()
        for omics_type, input_dim in omics_input_dims.items():
            # Simple linear projection per omics type
            self.projection_layers[omics_type] = nn.Linear(input_dim, common_embed_dim)

        # Calculate concatenated dimension
        concatenated_dim = len(self.omics_types) * common_embed_dim

        # Optional final MLP layer
        if final_output_dim is not None:
            self.final_mlp = nn.Sequential(
                nn.LayerNorm(concatenated_dim), # Normalize before MLP
                nn.Linear(concatenated_dim, final_output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
                # No final projection back needed if this is the output embedding dim
            )
        else:
            self.final_mlp = None

    def forward(self, raw_omics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for feature integration.

        Args:
            raw_omics_data: Dictionary mapping omics type name (str) to
                             input feature tensor (torch.Tensor) of shape
                             (Batch, NumFeatures_omic).

        Returns:
            Integrated patient embeddings tensor of shape (Batch, OutputDim),
            where OutputDim is final_output_dim if specified, otherwise
            len(omics_types) * common_embed_dim.
        """
        projected_embeddings = []
        for omics_type in self.omics_types:
            if omics_type not in raw_omics_data:
                raise ValueError(f"Missing omics type '{omics_type}' in input data.")
            if raw_omics_data[omics_type].shape[1] != self.omics_input_dims[omics_type]:
                 raise ValueError(f"Input dimension mismatch for omics type '{omics_type}'. "
                                  f"Expected {self.omics_input_dims[omics_type]}, "
                                  f"got {raw_omics_data[omics_type].shape[1]}.")

            x = raw_omics_data[omics_type]
            projected = self.projection_layers[omics_type](x)
            projected_embeddings.append(projected)

        # Concatenate the projected embeddings from all omics types
        concatenated_embeddings = torch.cat(projected_embeddings, dim=1)

        # Apply final MLP if specified
        if self.final_mlp is not None:
            output_embeddings = self.final_mlp(concatenated_embeddings)
        else:
            output_embeddings = concatenated_embeddings

        return output_embeddings 