import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, LayerNorm
from torch_geometric.data import HeteroData

# Define a simple residual block if needed, or handle inline
# class ResHeteroBlock(nn.Module): ... # Might be overkill

# Define a MaskedHeteroConv layer that can respect gene masks in message passing
class MaskedHeteroConv(nn.Module):
    """
    Extension of HeteroConv that respects gene masks during message passing.
    Uses masks to control influence of imputed (non-original) genes.
    """
    def __init__(self, convs, aggr="sum", gene_masks=None):
        super().__init__()
        self.convs = nn.ModuleDict()
        for edge_type, conv_module in convs.items():
            self.convs[str(edge_type)] = conv_module
        self.aggr = aggr
        self.gene_masks = gene_masks  # Dictionary mapping modality to gene mask

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with support for gene masking.
        
        Args:
            x_dict (dict): Input node features by type
            edge_index_dict (dict): Edge indices by type
            
        Returns:
            dict: Output node features
        """
        # Initialize output dictionary
        out_dict = {}
        
        # Process each edge type
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # Skip if source or target node type not in x_dict
            if src_type not in x_dict or dst_type not in x_dict:
                continue
                
            # Apply the convolution for this edge type
            src, dst = x_dict[src_type], x_dict[dst_type]
            conv = self.convs[str(edge_type)]
            
            # Check if the source or destination is 'gene' and apply masking if needed
            if (src_type == 'gene' or dst_type == 'gene') and self.gene_masks is not None:
                # Get the edge-specific result
                out = conv((src, dst), edge_index)
                
                # Apply mask if this is updating gene nodes
                if dst_type == 'gene':
                    # Create a combined gene mask (logical OR across all modalities)
                    combined_mask = None
                    for modality, mask in self.gene_masks.items():
                        mod_mask = torch.tensor(mask, dtype=torch.float, device=out.device)
                        if combined_mask is None:
                            combined_mask = mod_mask
                        else:
                            combined_mask = torch.maximum(combined_mask, mod_mask)
                    
                    # Apply the mask if available
                    if combined_mask is not None:
                        # Reshape the mask for broadcasting
                        reshaped_mask = combined_mask.view(-1, 1)  # [num_genes, 1]
                        
                        # Apply mask to message - originally present genes (mask=1) get full weight
                        # imputed genes (mask=0) get reduced influence in message passing
                        out = out * reshaped_mask + out * (1 - reshaped_mask) * 0.1  # Scale down imputed genes to 10%
                        
                # For any edge type, aggregate to destination
                if dst_type in out_dict:
                    out_dict[dst_type] += out
                else:
                    out_dict[dst_type] = out
            else:
                # Standard processing for edges not involving genes
                out = conv((src, dst), edge_index)
                if dst_type in out_dict:
                    out_dict[dst_type] += out
                else:
                    out_dict[dst_type] = out
        
        return out_dict

class HeteroGCN(nn.Module):
    def __init__(self, metadata, node_feature_dims, hidden_channels, out_channels,
                 num_layers=12, conv_type='sage', num_heads=16, dropout_rate=0.5,
                 use_layer_norm=True, activation_fn='leaky_relu', gene_masks=None):
        """
        Enhanced Heterogeneous Graph Convolutional Network with Residual Connections.

        Applies an initial linear projection, followed by multiple GNN layers with
        residual connections, normalization, and activation. Designed to work with
        precomputed embeddings or raw features.

        Args:
            metadata (tuple): Metadata tuple (node_types, edge_types) from HeteroData.
            node_feature_dims (dict): Dictionary mapping node type (str) to input feature dimension (int).
            hidden_channels (int): Number of hidden units in GNN layers.
            out_channels (int): Number of output units (final embedding dimension).
            num_layers (int): Number of GNN layers (must be >= 1).
            conv_type (str): Type of convolution ('gcn', 'sage', 'gat').
            num_heads (int): Number of attention heads for GATConv.
            dropout_rate (float): Dropout probability.
            use_layer_norm (bool): Whether to use LayerNorm.
            activation_fn (str): Activation function ('relu', 'leaky_relu').
            gene_masks (dict, optional): Dictionary mapping modality names to binary masks (1 for originally present, 0 for added).
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        node_types, edge_types = metadata
        self.node_types = node_types
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.gene_masks = gene_masks  # Store gene masks

        # Activation function
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # --- Input Projection ---
        self.input_lins = nn.ModuleDict()
        for node_type, in_dim in node_feature_dims.items():
            # Project input features to the hidden dimension
            self.input_lins[node_type] = Linear(in_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleDict() if use_layer_norm else None

        # --- GNN Layers ---
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            # Input to GNN layers is always hidden_channels (after projection/previous layer)
            current_in_channels = hidden_channels
            current_out_channels = out_channels if is_last_layer else hidden_channels

            conv_dict = {}
            for edge_type in edge_types:
                # Use (-1, -1) to let HeteroConv handle input dimensions dynamically based on graph structure
                src, _, dst = edge_type # Not directly needed for setting dims with (-1, -1)

                if conv_type.lower() == 'gcn':
                    # GCNConv handles bipartite message passing with (-1, -1)
                    conv_dict[edge_type] = GCNConv((-1, -1), current_out_channels)
                elif conv_type.lower() == 'sage':
                    # SAGEConv also handles bipartite message passing with (-1, -1)
                    conv_dict[edge_type] = SAGEConv((-1, -1), current_out_channels)
                elif conv_type.lower() == 'gat':
                    if current_out_channels % num_heads != 0:
                         raise ValueError(f"GAT output channels ({current_out_channels}) in layer {i} must be divisible by num_heads ({num_heads})")
                    gat_out_channels_per_head = current_out_channels // num_heads
                    # add_self_loops=False: Let HeteroConv manage self-loops if needed, based on edge types
                    conv_dict[edge_type] = GATConv((-1, -1), gat_out_channels_per_head, heads=num_heads, dropout=dropout_rate, add_self_loops=False)
                else:
                    raise ValueError(f"Unsupported conv_type: {conv_type}")

            # Use our MaskedHeteroConv instead of regular HeteroConv if gene masks are available
            if gene_masks is not None:
                conv = MaskedHeteroConv(conv_dict, aggr='sum', gene_masks=gene_masks)
            else:
                # Fallback to regular HeteroConv if no masks
                conv = HeteroConv(conv_dict, aggr='sum')
                
            self.convs.append(conv)

            # Add LayerNorm module for each node type for this layer
            if use_layer_norm:
                # Ensure self.norms is initialized
                if self.norms is None: self.norms = nn.ModuleDict()
                norm_dict = nn.ModuleDict()
                for node_type in node_types:
                    # Norm dimension matches the output dimension of the current convolution layer
                    norm_channels = current_out_channels
                    norm_dict[node_type] = LayerNorm(norm_channels)
                # Store the normalization modules for this layer
                self.norms[f'norm_{i}'] = norm_dict

        # Removed the final self.lin layer from the previous version.
        # The output embeddings are now directly the result of the final GNN layer block.


    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with initial projection and residual connections.

        Args:
            x_dict (dict): Input node features {node_type: tensor}.
                           Typically precomputed embeddings from an autoencoder.
            edge_index_dict (dict): Edge indices {edge_type: tensor}.

        Returns:
            dict: Output node embeddings {node_type: tensor}.
        """
        # 1. Input Projection
        # Project initial features (e.g., AE embeddings) into hidden dimension space
        h_dict = {
            node_type: self.input_lins[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Apply gene masking to initial gene embeddings if masks are available
        if 'gene' in h_dict and self.gene_masks is not None:
            # Create a combined mask across all modalities
            combined_mask = None
            for modality, mask in self.gene_masks.items():
                mod_mask = torch.tensor(mask, dtype=torch.float, device=h_dict['gene'].device)
                if combined_mask is None:
                    combined_mask = mod_mask
                else:
                    combined_mask = torch.maximum(combined_mask, mod_mask)
                    
            # Apply the mask to initial gene embeddings
            if combined_mask is not None:
                # Reshape for broadcasting (genes, 1)
                mask_reshaped = combined_mask.view(-1, 1)
                
                # Scale down imputed genes (mask=0) to have less influence 
                # Original genes (mask=1) keep full influence
                h_dict['gene'] = h_dict['gene'] * mask_reshaped + h_dict['gene'] * (1 - mask_reshaped) * 0.1
        
        # Apply activation after initial projection
        h_dict = {ntype: self.activation(h) for ntype, h in h_dict.items()}


        # 2. GNN Layers with Residuals
        for i, conv in enumerate(self.convs):
            # Store the input to this layer block for the potential residual connection
            h_prev_dict = h_dict

            # Apply GNN convolution
            h_conv_dict = conv(h_dict, edge_index_dict) # Result after convolution ONLY

            # --- Residual Connection ---
            # Check if residual connection is applicable (input and output dims match)
            # This is true for intermediate layers (hidden->hidden) and potentially
            # the last layer if hidden_channels == out_channels.
            can_add_residual = True
            # Check dimensions for all node types involved in the output of the convolution
            for node_type in h_conv_dict.keys():
                # Ensure the node type existed in the input dict as well
                if node_type not in h_prev_dict:
                    can_add_residual = False # Cannot add residual if node type is new
                    break
                if h_conv_dict[node_type].shape[-1] != h_prev_dict[node_type].shape[-1]:
                    can_add_residual = False # Dimension mismatch
                    break

            # Add residual connection *before* normalization and activation
            if can_add_residual:
                 h_dict = {
                     node_type: h_prev_dict.get(node_type, 0) + h_conv # Add previous state
                     for node_type, h_conv in h_conv_dict.items()
                 }
            else:
                # If residual cannot be added, use the convolution output directly
                h_dict = h_conv_dict

            # --- Normalization, Activation, Dropout ---
            # Apply per node type to the result (potentially including residual)
            temp_processed_h_dict = {}
            for node_type, h in h_dict.items():
                # Apply Norm (if enabled and exists for this layer/node_type)
                if self.use_layer_norm and f'norm_{i}' in self.norms and node_type in self.norms[f'norm_{i}']:
                    h = self.norms[f'norm_{i}'][node_type](h)
                # else: # Optional warning if norm is expected but missing
                #     if self.use_layer_norm:
                #         print(f"Warning: LayerNorm module missing for type '{node_type}' layer {i}.")


                # Apply Activation (applied to all layers in this setup)
                h = self.activation(h)

                # Apply Dropout (applied after activation, only during training)
                h = F.dropout(h, p=self.dropout_rate, training=self.training)

                temp_processed_h_dict[node_type] = h
            # Update h_dict with features processed by Norm/Act/Dropout
            h_dict = temp_processed_h_dict

        # The final h_dict contains the output embeddings after the last GNN layer block
        return h_dict

