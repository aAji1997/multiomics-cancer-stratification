# modelling/gcn/integrated_transformer_gcn_model.py
import torch
import torch.nn as nn
from typing import Dict, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Use absolute import path
from modelling.transformer.omics_transformer_encoder import OmicsTransformerEncoder
# Use explicit relative import for file in the same directory
from hetero_gcn_model import HeteroGCN




# --- Omics Decoder Module (Adapted from JointAE OmicsProcessor) ---
class OmicsDecoder(nn.Module):
    def __init__(self, gcn_patient_out_dim, gcn_gene_out_dim, omics_input_dims: Dict[str, int], num_genes, 
                 use_modality_specific_decoders=False, activation='sigmoid', patient_batch_size=32, reduce_memory=False):
        """
        Enhanced OmicsDecoder with support for modality-specific decoding.

        Args:
            gcn_patient_out_dim (int): Dimension of patient embeddings from GCN.
            gcn_gene_out_dim (int): Dimension of gene embeddings from GCN.
            omics_input_dims (Dict[str, int]): Dictionary mapping modality name to its input dimension.
            num_genes (int): Number of genes.
            use_modality_specific_decoders (bool): If True, uses separate decoder for each modality.
                                                  If False, decodes to a single concatenated tensor (original behavior).
            activation (str): Activation function to use ('sigmoid', 'relu', or 'none').
            patient_batch_size (int): Number of patients to process at once to save memory.
            reduce_memory (bool): If True, uses smaller intermediate dimensions in the decoder
                                  to reduce memory usage.
        """
        super().__init__()
        self.gcn_patient_out_dim = gcn_patient_out_dim
        self.gcn_gene_out_dim = gcn_gene_out_dim
        self.omics_input_dims = omics_input_dims # Dict mapping modality name to ORIGINAL feature dim
        self.modality_order = sorted(omics_input_dims.keys()) # Ensure consistent order
        self.num_modalities_out = len(self.modality_order)
        self.num_genes = num_genes
        self.use_modality_specific_decoders = use_modality_specific_decoders
        self.patient_batch_size = patient_batch_size
        self.reduce_memory = reduce_memory
        
        # Calculate total original feature dimension across modalities
        self.total_original_dim = sum(omics_input_dims.values())

        # 1. Decode patient embedding z_p to a patient-level context
        # Output dimension can be tuned, maybe related to gene dim?
        patient_decoder_intermediate = gcn_patient_out_dim * 2
        if reduce_memory:
            # Use smaller dimensions when reducing memory usage
            patient_decoder_intermediate = min(128, patient_decoder_intermediate)
            
        # Target dim: make it large enough to potentially hold combined info, maybe related to gene dim?
        patient_context_dim = gcn_gene_out_dim * 2
        if reduce_memory:
            # Use smaller dimensions when reducing memory usage
            patient_context_dim = min(128, patient_context_dim)
            
        self.patient_decoder = nn.Sequential(
            nn.Linear(gcn_patient_out_dim, patient_decoder_intermediate),
            nn.ReLU(),
            nn.LayerNorm(patient_decoder_intermediate),
            nn.Linear(patient_decoder_intermediate, patient_context_dim)
        )

        # 2. Combine patient context and GCN gene embedding per gene
        combined_decoder_input_dim = patient_context_dim + gcn_gene_out_dim
        
        # For the reconstruction path, we either:
        # a) Decode to a single concatenated tensor (original approach)
        # b) Decode to a latent representation that will feed into modality-specific decoders
        
        reconstruction_intermediate_dim = (combined_decoder_input_dim + self.total_original_dim) // 2
        if reduce_memory:
            # Use a much smaller intermediate dimension when reducing memory
            reconstruction_intermediate_dim = min(128, combined_decoder_input_dim // 2)
        
        if self.use_modality_specific_decoders:
            # Aim to reconstruct a latent representation, similar to OmicsProcessor's reconstruction_mlp
            # We'll use total_latent_dim as the dimensionality for the latent space
            # This is a design choice - we could make it a parameter if needed
            total_latent_dim = self.total_original_dim // 2  # Simplification - could be parameterized
            if reduce_memory:
                # Use a much smaller latent dimension when reducing memory
                total_latent_dim = min(64, total_latent_dim // 2)
                
            self.total_latent_dim = total_latent_dim
            
            # Calculate latent dimensions for each modality based on input proportions
            self.modality_latent_dims = {}
            for mod_name, in_dim in omics_input_dims.items():
                # Proportional allocation: modality gets latent dim proportional to its input dim
                latent_dim = max(
                    16,  # Minimum latent dim
                    int(total_latent_dim * (in_dim / self.total_original_dim))
                )
                if reduce_memory:
                    # Cap the maximum latent dimension per modality when reducing memory
                    latent_dim = min(32, latent_dim)
                self.modality_latent_dims[mod_name] = latent_dim
                
            # Adjust reconstruction_mlp to output latent representation
            self.reconstruction_mlp = nn.Sequential(
                nn.Linear(combined_decoder_input_dim, reconstruction_intermediate_dim),
                nn.ReLU(),
                nn.LayerNorm(reconstruction_intermediate_dim),
                nn.Linear(reconstruction_intermediate_dim, total_latent_dim)  # Output latent space
            )
            
            # 3. Modality-specific decoders (similar to OmicsProcessor)
            self.modality_decoders = nn.ModuleDict()
            for mod_name, in_dim in omics_input_dims.items():
                latent_dim = self.modality_latent_dims[mod_name]
                intermediate_dim = max(16, latent_dim * 2)  # Tunable
                if reduce_memory:
                    # Cap the intermediate dimension when reducing memory
                    intermediate_dim = min(32, intermediate_dim)
                
                # Input: latent representation -> Output: reconstructed modality values
                self.modality_decoders[mod_name] = nn.Sequential(
                    nn.Linear(latent_dim, intermediate_dim),
                    nn.ReLU(),
                    nn.LayerNorm(intermediate_dim),
                    nn.Linear(intermediate_dim, 1)  # Each gene gets 1 value per modality
                )
                
        else:
            # Original approach: directly reconstruct concatenated tensor
            self.reconstruction_mlp = nn.Sequential(
                nn.Linear(combined_decoder_input_dim, reconstruction_intermediate_dim),
                nn.ReLU(),
                nn.LayerNorm(reconstruction_intermediate_dim),
                # Output should match the total original dimensionality across all modalities
                nn.Linear(reconstruction_intermediate_dim, self.total_original_dim)
            )
        
        # Final activation - Sigmoid if data was normalized [0,1], otherwise maybe None or ReLU
        if activation.lower() == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif activation.lower() == 'relu':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()  # No activation

    def forward(self, z_p, z_gene):
        """
        Decodes patient (z_p) and gene (z_gene) embeddings from GCN
        back to the omics feature space, using batching to reduce memory usage.

        Args:
            z_p (Tensor): Patient embeddings (batch_size, gcn_patient_out_dim).
            z_gene (Tensor): Gene embeddings (num_genes, gcn_gene_out_dim).

        Returns:
            If use_modality_specific_decoders=False:
                Tensor: Reconstructed concatenated omics (batch_size, num_genes, total_original_dim).
            If use_modality_specific_decoders=True:
                Dict[str, Tensor]: Reconstructed omics per modality {modality: tensor(batch_size, num_genes, 1)}.
                AND concatenated tensor (batch_size, num_genes, num_modalities).
        """
        batch_size = z_p.shape[0]
        
        # Process patients in smaller batches to reduce memory usage
        num_batches = (batch_size + self.patient_batch_size - 1) // self.patient_batch_size
        
        if not self.use_modality_specific_decoders:
            # Initialize output tensor for concatenated approach
            result = torch.zeros(batch_size, self.num_genes, self.total_original_dim, 
                               device=z_p.device, dtype=z_p.dtype)
            
            # Process in batches
            for i in range(num_batches):
                start_idx = i * self.patient_batch_size
                end_idx = min(start_idx + self.patient_batch_size, batch_size)
                
                # Get current batch of patient embeddings
                z_p_batch = z_p[start_idx:end_idx]
                batch_size_current = z_p_batch.shape[0]
                
                # 1. Decode z_p to patient context
                patient_context = self.patient_decoder(z_p_batch)
                
                # 2. Prepare for combination - only expand to necessary dimensions
                patient_context_expanded = patient_context.unsqueeze(1).expand(-1, self.num_genes, -1)
                z_gene_expanded = z_gene.unsqueeze(0).expand(batch_size_current, -1, -1)
                
                # 3. Combine patient context and gene embedding
                combined_decoder_input = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)
                
                # 4. Apply reconstruction MLP
                rec_omics_cat = self.reconstruction_mlp(combined_decoder_input)
                
                # 5. Apply final activation
                batch_result = self.final_activation(rec_omics_cat)
                
                # Store result for this batch
                result[start_idx:end_idx] = batch_result
                
                # Explicitly delete temporary tensors to free memory
                del patient_context, patient_context_expanded, z_gene_expanded, combined_decoder_input, rec_omics_cat, batch_result
                torch.cuda.empty_cache() if z_p.is_cuda else None
                
            return result
        else:
            # Modality-specific approach - initialize dictionaries
            reconstructed_modalities = {mod_name: torch.zeros(batch_size, self.num_genes, 1,
                                                           device=z_p.device, dtype=z_p.dtype)
                                     for mod_name in self.modality_order}
            
            # Process in batches
            for i in range(num_batches):
                start_idx = i * self.patient_batch_size
                end_idx = min(start_idx + self.patient_batch_size, batch_size)
                
                # Get current batch of patient embeddings
                z_p_batch = z_p[start_idx:end_idx]
                batch_size_current = z_p_batch.shape[0]
                
                # 1. Decode z_p to patient context
                patient_context = self.patient_decoder(z_p_batch)
                
                # 2. Prepare for combination - only expand to necessary dimensions
                patient_context_expanded = patient_context.unsqueeze(1).expand(-1, self.num_genes, -1)
                z_gene_expanded = z_gene.unsqueeze(0).expand(batch_size_current, -1, -1)
                
                # 3. Combine patient context and gene embedding
                combined_decoder_input = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)
                
                # 4. Reconstruct latent representation
                rec_latent = self.reconstruction_mlp(combined_decoder_input)
                
                # 5. Apply modality-specific decoders
                batch_reconstructed_tensors = []
                
                current_dim = 0
                for mod_name in self.modality_order:
                    latent_dim = self.modality_latent_dims[mod_name]
                    
                    # Extract modality-specific latent representation
                    if current_dim + latent_dim <= rec_latent.shape[-1]:
                        mod_latent = rec_latent[:, :, current_dim:current_dim + latent_dim]
                        current_dim += latent_dim
                    else:
                        # Handle potential dimension mismatch gracefully
                        mod_latent = rec_latent[:, :, current_dim:]
                        current_dim = rec_latent.shape[-1]  # Set to end
                    
                    # Apply modality-specific decoder
                    mod_decoder = self.modality_decoders[mod_name]
                    rec_mod = mod_decoder(mod_latent)  # Shape (batch_size_current, num_genes, 1)
                    rec_mod = self.final_activation(rec_mod)  # Apply activation to each modality
                    
                    # Store in result dictionary
                    reconstructed_modalities[mod_name][start_idx:end_idx] = rec_mod
                    batch_reconstructed_tensors.append(rec_mod)
                
                # Free memory
                del patient_context, patient_context_expanded, z_gene_expanded
                del combined_decoder_input, rec_latent, mod_latent, rec_mod
                torch.cuda.empty_cache() if z_p.is_cuda else None
                
            # Create concatenated tensor from individual modalities
            concatenated = torch.cat([reconstructed_modalities[mod_name] for mod_name in self.modality_order], dim=-1)
            
            # Return both dictionary and concatenated format
            reconstructed_modalities['concatenated'] = concatenated
            return reconstructed_modalities
            
    def decode_single_modality(self, z_p, z_gene, modality):
        """
        Decode only a specific modality (for efficiency when only one modality is needed).
        Only available when use_modality_specific_decoders=True.
        Uses batched processing to save memory.
        
        Args:
            z_p (Tensor): Patient embeddings (batch_size, gcn_patient_out_dim).
            z_gene (Tensor): Gene embeddings (num_genes, gcn_gene_out_dim).
            modality (str): Name of the modality to decode.
            
        Returns:
            Tensor: Reconstructed modality (batch_size, num_genes, 1)
        """
        if not self.use_modality_specific_decoders:
            raise ValueError("decode_single_modality is only available when use_modality_specific_decoders=True")
            
        if modality not in self.modality_order:
            raise ValueError(f"Unknown modality: {modality}. Available modalities: {self.modality_order}")
            
        batch_size = z_p.shape[0]
        
        # Find position of requested modality in latent space
        current_dim = 0
        target_latent_dim = None
        
        for mod_name in self.modality_order:
            latent_dim = self.modality_latent_dims[mod_name]
            if mod_name == modality:
                target_latent_dim = latent_dim
                break
            current_dim += latent_dim
            
        if target_latent_dim is None:
            raise ValueError(f"Could not find latent dimensions for modality: {modality}")
            
        # Initialize output tensor
        result = torch.zeros(batch_size, self.num_genes, 1, device=z_p.device, dtype=z_p.dtype)
        
        # Process in batches
        num_batches = (batch_size + self.patient_batch_size - 1) // self.patient_batch_size
        
        for i in range(num_batches):
            start_idx = i * self.patient_batch_size
            end_idx = min(start_idx + self.patient_batch_size, batch_size)
            
            # Get current batch of patient embeddings
            z_p_batch = z_p[start_idx:end_idx]
            batch_size_current = z_p_batch.shape[0]
            
            # 1. Decode z_p to patient context
            patient_context = self.patient_decoder(z_p_batch)
            
            # 2. Prepare for combination
            patient_context_expanded = patient_context.unsqueeze(1).expand(-1, self.num_genes, -1)
            z_gene_expanded = z_gene.unsqueeze(0).expand(batch_size_current, -1, -1)
            
            # 3. Combine patient context and gene embedding
            combined_decoder_input = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)
            
            # 4. Reconstruct latent representation
            rec_latent = self.reconstruction_mlp(combined_decoder_input)
            
            # 5. Extract and decode only the requested modality's latent space
            mod_latent = rec_latent[:, :, current_dim:current_dim + target_latent_dim]
            mod_decoder = self.modality_decoders[modality]
            rec_mod = mod_decoder(mod_latent)
            batch_result = self.final_activation(rec_mod)
            
            # Store result for this batch
            result[start_idx:end_idx] = batch_result
            
            # Free memory
            del patient_context, patient_context_expanded, z_gene_expanded
            del combined_decoder_input, rec_latent, mod_latent, rec_mod, batch_result
            torch.cuda.empty_cache() if z_p.is_cuda else None
            
        return result
# -----------------------------------------------------------------

class IntegratedTransformerGCN(nn.Module):
    """
    Integrates an OmicsTransformerEncoder with a HeteroGCN.
    The Transformer processes raw omics to get patient embeddings.
    The GCN then combines patient and gene embeddings using the graph structure.
    Includes an OmicsDecoder for reconstruction loss.
    """
    def __init__(self,
                 # --- Required Args --- #
                 omics_input_dims: Dict[str, int],
                 transformer_embed_dim: int,
                 transformer_num_heads: int,
                 transformer_ff_dim: int,
                 num_transformer_layers: int,
                 transformer_output_dim: int, # Matches GCN 'patient' input dim
                 gcn_metadata: Tuple[list, list],
                 gene_feature_dim: int,
                 gcn_hidden_channels: int,
                 gcn_out_channels: int, # GCN Output channels for patient AND gene
                 gcn_num_layers: int,
                 # --- Optional Args --- #
                 transformer_dropout: float = 0.1,
                 gcn_conv_type: str = 'sage',
                 gcn_num_heads: int = 4,
                 gcn_dropout_rate: float = 0.5,
                 gcn_use_layer_norm: bool = True,
                 gene_masks: Dict | None = None,
                 add_omics_decoder: bool = False, # Flag to add the decoder
                 use_modality_specific_decoders: bool = False,  # New argument
                 decoder_activation: str = 'sigmoid',  # New argument
                 decoder_patient_batch_size: int = 32,  # Batch size for memory efficiency
                 num_genes: int | None = None, # Required if add_omics_decoder is True
                 use_gradient_checkpointing: bool = False, # Enable gradient checkpointing for memory efficiency
                 reduce_decoder_memory: bool = False, # Use smaller dimensions in decoder for memory efficiency
                 use_mixed_precision: bool = False # Enable automatic mixed precision
                 ):
        """
        Args:
            omics_input_dims: For OmicsTransformerEncoder.
            transformer_embed_dim: For OmicsTransformerEncoder.
            transformer_num_heads: For OmicsTransformerEncoder.
            transformer_ff_dim: For OmicsTransformerEncoder.
            num_transformer_layers: For OmicsTransformerEncoder.
            transformer_output_dim: Output dim of Transformer, input dim for GCN 'patient' nodes.
            gcn_metadata: Metadata for HeteroGCN.
            gene_feature_dim: Explicitly required gene feature dimension.
            gcn_hidden_channels: Hidden channels for HeteroGCN.
            gcn_out_channels: Final output channels for HeteroGCN.
            gcn_num_layers: Number of layers for HeteroGCN.
            transformer_dropout: For OmicsTransformerEncoder.
            gcn_conv_type: Convolution type for HeteroGCN.
            gcn_num_heads: Attention heads for GAT in HeteroGCN.
            gcn_dropout_rate: Dropout for HeteroGCN.
            gcn_use_layer_norm: Whether HeteroGCN uses LayerNorm.
            gene_masks: Optional dictionary of gene masks passed to HeteroGCN.
            add_omics_decoder (bool): If True, adds the omics decoder for reconstruction loss.
            use_modality_specific_decoders (bool): If True, the omics decoder will use separate 
                                                 decoders for each modality. Default: False.
            decoder_activation (str): Activation function for the decoder ('sigmoid', 'relu', 'none').
                                      Default: 'sigmoid'.
            decoder_patient_batch_size (int): Number of patients to process at once during decoding.
                                             Lower values use less memory but may be slower.
            num_genes (int, optional): Number of genes, required if add_omics_decoder is True.
            use_gradient_checkpointing (bool): If True, uses gradient checkpointing to reduce memory usage
                                              at the cost of some additional computation time.
            reduce_decoder_memory (bool): If True, uses smaller intermediate dimensions in the decoder
                                         to reduce memory usage.
            use_mixed_precision (bool): If True, uses automatic mixed precision for forward pass
                                        to reduce memory usage and potentially speed up computation.
        """
        super().__init__()

        # Storing num_genes if provided, needed for decoder
        self.num_genes = num_genes
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        
        self.transformer_encoder = OmicsTransformerEncoder(
            omics_input_dims=omics_input_dims,
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim,
            num_transformer_layers=num_transformer_layers,
            output_dim=transformer_output_dim,
            dropout=transformer_dropout
        )

        # Define node feature dimensions for GCN
        gcn_node_feature_dims = {
            'patient': transformer_output_dim,
            'gene': gene_feature_dim
        }

        self.gcn = HeteroGCN(
            metadata=gcn_metadata,
            node_feature_dims=gcn_node_feature_dims,
            hidden_channels=gcn_hidden_channels,
            out_channels=gcn_out_channels, # Ensure GCN outputs consistent dim for patient/gene
            num_layers=gcn_num_layers,
            conv_type=gcn_conv_type,
            num_heads=gcn_num_heads,
            dropout_rate=gcn_dropout_rate,
            use_layer_norm=gcn_use_layer_norm,
            gene_masks=gene_masks
        )
        
        # Instantiate Omics Decoder if requested
        self.omics_decoder = None
        if add_omics_decoder:
            if self.num_genes is None:
                raise ValueError("num_genes must be provided to IntegratedTransformerGCN when add_omics_decoder is True")
                
            self.omics_decoder = OmicsDecoder(
                gcn_patient_out_dim=gcn_out_channels, # From GCN output
                gcn_gene_out_dim=gcn_out_channels,    # From GCN output
                omics_input_dims=omics_input_dims,    # Original dims
                num_genes=self.num_genes,             # Number of genes
                use_modality_specific_decoders=use_modality_specific_decoders,  # New param
                activation=decoder_activation,        # New param
                patient_batch_size=decoder_patient_batch_size,  # For memory efficiency
                reduce_memory=reduce_decoder_memory   # Use smaller dimensions for memory efficiency
            )
            
            # Store these for reference
            self.use_modality_specific_decoders = use_modality_specific_decoders
            self.decoder_activation = decoder_activation
            self.decoder_patient_batch_size = decoder_patient_batch_size

    def forward(self,
                raw_omics_data: Dict[str, torch.Tensor],
                gene_embeddings: torch.Tensor,
                edge_index_dict: Dict[Tuple, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        Returns dict containing final 'patient' and 'gene' embeddings.
        """
        # Use automatic mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available() and self.training:
            with torch.cuda.amp.autocast():
                return self._forward_impl(raw_omics_data, gene_embeddings, edge_index_dict)
        else:
            return self._forward_impl(raw_omics_data, gene_embeddings, edge_index_dict)
    
    def _forward_impl(self,
                     raw_omics_data: Dict[str, torch.Tensor],
                     gene_embeddings: torch.Tensor,
                     edge_index_dict: Dict[Tuple, torch.Tensor]
                     ) -> Dict[str, torch.Tensor]:
        """
        Implementation of the forward pass, to allow for mixed precision wrapping.
        """
        # Use gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            # 1. Define custom functions for checkpointing
            def create_custom_transformer_forward(module, data):
                def custom_forward():
                    return module(data)
                return custom_forward
                
            def create_custom_gcn_forward(module, x_dict, edge_dict):
                def custom_forward():
                    return module(x_dict, edge_dict)
                return custom_forward
            
            # 2. Get initial patient embeddings from Transformer using checkpointing
            initial_patient_embeddings = torch.utils.checkpoint.checkpoint(
                create_custom_transformer_forward(self.transformer_encoder, raw_omics_data),
                use_reentrant=False  # Non-reentrant checkpointing is more memory efficient
            )

            # 3. Prepare GCN input dict
            x_dict = {
                'patient': initial_patient_embeddings,
                'gene': gene_embeddings
            }

            # 4. Pass through GCN using checkpointing
            final_embeddings_dict = torch.utils.checkpoint.checkpoint(
                create_custom_gcn_forward(self.gcn, x_dict, edge_index_dict),
                use_reentrant=False
            )
        else:
            # Standard forward pass without checkpointing
            # 1. Get initial patient embeddings from Transformer
            initial_patient_embeddings = self.transformer_encoder(raw_omics_data)

            # 2. Prepare GCN input dict
            x_dict = {
                'patient': initial_patient_embeddings,
                'gene': gene_embeddings
            }

            # 3. Pass through GCN
            final_embeddings_dict = self.gcn(x_dict, edge_index_dict)

        return final_embeddings_dict
    
    def decode_omics(self, final_embeddings_dict: Dict[str, torch.Tensor]):
        """
        Decodes the final GCN embeddings back to omics space if decoder exists.
        
        Returns:
            When use_modality_specific_decoders=False:
                Tensor: Reconstructed concatenated omics tensor.
            When use_modality_specific_decoders=True:
                Dict[str, Tensor]: Reconstructed modalities with additional 'concatenated' key.
        """
        if self.omics_decoder is None:
            raise RuntimeError("Omics decoder was not added to the model.")
            
        z_p = final_embeddings_dict.get('patient')
        z_gene = final_embeddings_dict.get('gene')
        
        if z_p is None or z_gene is None:
            raise ValueError("Missing 'patient' or 'gene' embeddings in input dict for decoding.")
        
        # Use mixed precision if enabled and in training mode
        if self.use_mixed_precision and torch.cuda.is_available() and self.training:
            with torch.cuda.amp.autocast():
                return self._decode_omics_impl(z_p, z_gene)
        else:
            return self._decode_omics_impl(z_p, z_gene)
    
    def _decode_omics_impl(self, z_p, z_gene):
        """
        Implementation of omics decoding to allow for mixed precision wrapping.
        """
        # Use gradient checkpointing for decoder if enabled and in training mode
        if hasattr(self, 'use_gradient_checkpointing') and self.use_gradient_checkpointing and self.training:
            def create_custom_decoder_forward(module, p_embeds, g_embeds):
                def custom_forward():
                    return module(p_embeds, g_embeds)
                return custom_forward
                
            return torch.utils.checkpoint.checkpoint(
                create_custom_decoder_forward(self.omics_decoder, z_p, z_gene),
                use_reentrant=False
            )
        else:
            return self.omics_decoder(z_p, z_gene)
        
    def decode_single_modality(self, final_embeddings_dict: Dict[str, torch.Tensor], modality: str):
        """
        Decodes a single modality for efficiency. Only available with modality-specific decoders.
        
        Args:
            final_embeddings_dict: Dictionary with 'patient' and 'gene' embeddings
            modality: Name of the modality to decode
            
        Returns:
            Tensor: Reconstructed modality tensor
        """
        if self.omics_decoder is None:
            raise RuntimeError("Omics decoder was not added to the model.")
            
        if not hasattr(self, 'use_modality_specific_decoders') or not self.use_modality_specific_decoders:
            raise ValueError("decode_single_modality requires use_modality_specific_decoders=True")
            
        z_p = final_embeddings_dict.get('patient')
        z_gene = final_embeddings_dict.get('gene')
        
        if z_p is None or z_gene is None:
            raise ValueError("Missing 'patient' or 'gene' embeddings in input dict for decoding.")
        
        # Use mixed precision if enabled and in training mode
        if self.use_mixed_precision and torch.cuda.is_available() and self.training:
            with torch.cuda.amp.autocast():
                return self.omics_decoder.decode_single_modality(z_p, z_gene, modality)
        else:
            return self.omics_decoder.decode_single_modality(z_p, z_gene, modality)

    def forward_gcn_only(self, x_dict, edge_index_dict):
        """
        Forwards only through the GCN part of the model, for inference when using pre-computed embeddings.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            
        Returns:
            Dictionary of node embeddings for each node type
        """
        return self.gcn(x_dict, edge_index_dict)

    def get_available_modalities(self):
        """
        Returns a list of modalities available for decoding with modality-specific decoder.
        
        Returns:
            list: List of modality names, or None if modality-specific decoders aren't used
        """
        if self.omics_decoder is None:
            return None
            
        if not hasattr(self, 'use_modality_specific_decoders') or not self.use_modality_specific_decoders:
            return None
            
        return self.omics_decoder.modality_order

# Example usage (conceptual - requires actual data loading and setup)
if __name__ == '__main__':
    # --- Dummy Data & Config --- #
    num_patients = 50
    num_genes = 500
    # Omics features (dummy)
    omics_data = {
        'rnaseq': torch.randn(num_patients, 1000),
        'methylation': torch.randn(num_patients, 2000)
    }
    omics_dims = {'rnaseq': 1000, 'methylation': 2000}
    
    # Pre-computed gene embeddings (dummy)
    gene_embed_dim = 128
    precomputed_gene_embeddings = torch.randn(num_genes, gene_embed_dim)

    # Dummy gene masks (optional)
    dummy_masks = {
        'rnaseq': [1] * 400 + [0] * (num_genes - 400),
        'methylation': [1] * 350 + [0] * (num_genes - 350)
    }

    # Graph structure (dummy)
    edge_index_gg = torch.randint(0, num_genes, (2, 1500)) # gene-gene
    edge_index_pg_indices_0 = torch.randint(0, num_patients, (1, 2000))
    edge_index_pg_indices_1 = torch.randint(0, num_genes, (1, 2000))
    edge_index_pg = torch.cat([edge_index_pg_indices_0, edge_index_pg_indices_1], dim=0)
    edge_index_gp = edge_index_pg[[1, 0], :] # gene-patient
    
    edges = {
        ('gene', 'interacts', 'gene'): edge_index_gg,
        ('patient', 'expresses', 'gene'): edge_index_pg,
        ('gene', 'rev_expresses', 'patient'): edge_index_gp
    }
    metadata = (['patient', 'gene'], list(edges.keys()))
    
    # --- Model Instantiation --- #
    transformer_out_dim = 64 # Example: Output dim of transformer, input dim for GCN patient nodes
    gcn_out_dim = 32 # Final output dim
    
    integrated_model = IntegratedTransformerGCN(
        # --- Required Args --- #
        omics_input_dims=omics_dims,
        transformer_embed_dim=128,
        transformer_num_heads=4,
        transformer_ff_dim=256,
        num_transformer_layers=2,
        transformer_output_dim=transformer_out_dim, # Must match GCN patient input
        gcn_metadata=metadata,
        gene_feature_dim=gene_embed_dim, # Pass the known gene embedding dimension
        gcn_hidden_channels=128,
        gcn_out_channels=gcn_out_dim, 
        gcn_num_layers=2,
        # --- Optional Args --- #
        transformer_dropout=0.1,
        gcn_conv_type='sage',
        gcn_dropout_rate=0.5,
        gcn_use_layer_norm=True,
        gene_masks=dummy_masks,
        add_omics_decoder=True,
        use_modality_specific_decoders=False,
        decoder_activation='sigmoid',
        decoder_patient_batch_size=16,  # Process patients in small batches to save memory
        num_genes=num_genes,
        use_gradient_checkpointing=False,
        reduce_decoder_memory=False,
        use_mixed_precision=False
    )

    print("Integrated Model:", integrated_model)

    # --- Forward Pass --- #
    try:
        final_embeddings = integrated_model(
            raw_omics_data=omics_data,
            gene_embeddings=precomputed_gene_embeddings,
            edge_index_dict=edges
        )
        print("\nOutput Embeddings Dictionary:")
        for node_type, embeds in final_embeddings.items():
            print(f"  {node_type}: {embeds.shape}")
            
        # Check output shapes
        assert final_embeddings['patient'].shape == (num_patients, gcn_out_dim)
        assert final_embeddings['gene'].shape == (num_genes, gcn_out_dim)
        print("\nForward pass successful with expected output shapes!")
        
        # --- Omics Decoding Pass --- #
        reconstructed_omics = integrated_model.decode_omics(final_embeddings)
        print(f"\nReconstructed omics shape: {reconstructed_omics.shape}")
        total_original_dim = sum(omics_dims.values())
        assert reconstructed_omics.shape == (num_patients, num_genes, total_original_dim)
        print("Omics decoding pass successful!")

    except Exception as e:
        print(f"\nError during forward/decode pass: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck model initialization, input dimensions, and HeteroGCN implementation.") 