import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric as tg




class InteractionGraphAutoencoder(nn.Module):
    def __init__(self, feature_dim, gene_embedding_dim, dropout=0.5):
        """
        Args:
            feature_dim (int): Dimension of initial node features.
            gene_embedding_dim (int): Dimension of the latent gene embeddings (Z_gene).
            dropout (float): Dropout probability.
        """
        super(InteractionGraphAutoencoder, self).__init__()
        hidden_channels_intermediate1 = gene_embedding_dim * 4 # Increased capacity
        hidden_channels_intermediate2 = gene_embedding_dim * 2

        self.conv1 = tg.nn.GCNConv(feature_dim, hidden_channels_intermediate1)
        self.norm1 = nn.LayerNorm(hidden_channels_intermediate1)
        self.conv2 = tg.nn.GCNConv(hidden_channels_intermediate1, hidden_channels_intermediate2) # Added layer
        self.norm2 = nn.LayerNorm(hidden_channels_intermediate2) # Added norm

        # Output layers for mean (mu) and log variance (log_var)
        # GCNConv outputting 2*embedding_dim, split later
        self.conv_out = tg.nn.GCNConv(hidden_channels_intermediate2, gene_embedding_dim * 2) # Input from conv2

        self.dropout_layer = nn.Dropout(dropout) 

    def encode(self, x, edge_index, edge_weight=None):
        """Encodes the graph into latent distribution parameters (mu, log_var)
           and samples gene embeddings (z_gene).

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Graph connectivity.
            edge_weight (Tensor, optional): Edge weights. Defaults to None.

        Returns:
            tuple: Contains mu (mean), log_var (log variance), and z_gene (sampled embedding).
        """
        # Layer 1
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.dropout_layer(x)

        # Layer 2 (Added)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.norm2(x)
        x = self.dropout_layer(x) # Apply dropout again

        # Get the combined output for mu and log_var
        out_features = self.conv_out(x, edge_index, edge_weight=edge_weight)

        # Split into mu and log_var
        mu = out_features[:, :out_features.shape[1] // 2]
        log_var = out_features[:, out_features.shape[1] // 2:]

        z_gene = self.reparameterize(mu, log_var)
        return mu, log_var, z_gene

    def reparameterize(self, mu, log_var):
        """Performs reparameterization trick to sample from Gaussian distribution."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            # During evaluation, use the mean directly
            return mu


    def decode(self, z_gene):
        """Reconstructs the adjacency matrix logits from latent gene embeddings."""
        # Remove sigmoid: BCEWithLogitsLoss expects raw logits
        adj_rec_logits = z_gene @ z_gene.t()
        return adj_rec_logits

    def forward(self, x, edge_index, edge_weight=None):
        mu, log_var, z_gene = self.encode(x, edge_index, edge_weight)
        adj_reconstructed = self.decode(z_gene)
        return adj_reconstructed, mu, log_var, z_gene


# Modified Omics Processor (integrates Z_gene, encodes/decodes patient omics)
class OmicsProcessor(nn.Module):
    def __init__(self, modality_latent_dims: dict, modality_order: list,
                 gene_embedding_dim, patient_embedding_dim, num_genes, gene_masks=None):
        """
        Args:
            modality_latent_dims (dict): Dictionary mapping modality name (str) to its desired latent dimension (int).
            modality_order (list): List of modality names in the order they appear in the input tensor's last dim.
            gene_embedding_dim (int): Dimension of Z_gene from the graph AE.
            patient_embedding_dim (int): Desired dimension for the final patient embedding (z_p).
            num_genes (int): Total number of genes.
            gene_masks (dict, optional): Dictionary mapping modality names to binary masks (1 for originally present, 0 for added during harmonization).
        """
        super(OmicsProcessor, self).__init__()
        self.num_genes = num_genes
        self.modality_order = modality_order
        self.modality_latent_dims = modality_latent_dims
        self.num_modalities_in = len(modality_order) # Number of input modalities
        self.gene_embedding_dim = gene_embedding_dim
        self.patient_embedding_dim = patient_embedding_dim
        self.total_latent_dim = sum(modality_latent_dims.values())
        self.gene_masks = gene_masks

        # Encoder Part
        # 1. Process per-gene omics modalities individually
        self.modality_encoders = nn.ModuleDict()
        for mod_name in self.modality_order:
            latent_dim = self.modality_latent_dims[mod_name]
            intermediate_dim = max(16, latent_dim // 2) 
            # Input shape (B, N, 1) -> Output shape (B, N, latent_dim)
            self.modality_encoders[mod_name] = nn.Sequential(
                nn.Linear(1, intermediate_dim), 
                nn.ReLU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, latent_dim), 
                nn.ReLU() 
            )

        # 2. Combine concatenated latent modalities with gene embedding (Z_gene)
        combined_feature_dim = self.total_latent_dim + gene_embedding_dim
        gene_combiner_intermediate_dim1 = (combined_feature_dim + patient_embedding_dim) // 2 # Tunable intermediate dimension
        gene_combiner_intermediate_dim2 = patient_embedding_dim * 2 # Tunable intermediate dimension
        gene_combiner_out_dim = patient_embedding_dim # Match patient_aggregator input

        self.gene_combiner = nn.Sequential(
            nn.Linear(combined_feature_dim, gene_combiner_intermediate_dim1),
            nn.ReLU(),
            nn.LayerNorm(gene_combiner_intermediate_dim1), 
            nn.Linear(gene_combiner_intermediate_dim1, gene_combiner_intermediate_dim2), 
            nn.ReLU(),
            nn.LayerNorm(gene_combiner_intermediate_dim2), 
            nn.Linear(gene_combiner_intermediate_dim2, gene_combiner_out_dim), # Final projection
            nn.ReLU()
        )

        # 3. Aggregate gene representations into a patient embedding (z_p)
        aggregator_intermediate_dim = (gene_combiner_out_dim + patient_embedding_dim) // 2
        self.patient_aggregator = nn.Sequential(
            nn.Linear(gene_combiner_out_dim, aggregator_intermediate_dim), 
            nn.ReLU(),
            nn.Linear(aggregator_intermediate_dim, patient_embedding_dim),
        )


        # Decoder Part - REVISED ARCHITECTURE
        # 1. Decode patient embedding z_p to a patient-level context
        # Output dimension can be tuned, matching total_latent_dim seems reasonable
        patient_decoder_intermediate = patient_embedding_dim * 2
        self.patient_decoder = nn.Sequential(
            nn.Linear(patient_embedding_dim, patient_decoder_intermediate),
            nn.ReLU(),
            nn.LayerNorm(patient_decoder_intermediate),
            nn.Linear(patient_decoder_intermediate, self.total_latent_dim) # Output patient context
        )

        # 2. Combine patient context and gene embedding z_gene per gene
        combined_decoder_input_dim = self.total_latent_dim + self.gene_embedding_dim
        reconstruction_intermediate_dim = (combined_decoder_input_dim + self.total_latent_dim) // 2 # Tunable

        # This MLP reconstructs the concatenated latent omics representation per gene
        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(combined_decoder_input_dim, reconstruction_intermediate_dim),
            nn.ReLU(),
            nn.LayerNorm(reconstruction_intermediate_dim),
            nn.Linear(reconstruction_intermediate_dim, self.total_latent_dim) # Output matches latent omics target
        )

        # 3. Modality-specific decoders (unchanged from original)
        #    These take slices of the output from reconstruction_mlp
        #    and reconstruct the original data per modality.
        self.modality_decoders = nn.ModuleDict()
        for mod_name, latent_dim in self.modality_latent_dims.items():
            intermediate_dim = max(16, latent_dim // 2) # Match encoder structure loosely
            # Input shape (B, N, latent_dim) -> Output shape (B, N, 1)
            self.modality_decoders[mod_name] = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim), 
                nn.ReLU(),
                nn.LayerNorm(intermediate_dim), 
                nn.Linear(intermediate_dim, 1)
            )

        # Final activation for reconstructed omics data
        self.final_activation = nn.Sigmoid() 


    def encode(self, x_patient_structured, z_gene):
        """
        Encodes structured patient omics data, integrating graph-based gene embeddings.
        Processes each modality separately before combining.

        Args:
            x_patient_structured (Tensor): Patient's omics data (batch_size, num_genes, num_modalities_in).
            z_gene (Tensor): Gene embeddings from graph AE (num_genes, gene_embedding_dim).

        Returns:
            Tensor: Patient latent embedding (z_p) (batch_size, patient_embedding_dim).
        """
        batch_size = x_patient_structured.shape[0]

        # 1. Split input and apply modality-specific encoders
        latent_modality_tensors = []
        split_data = torch.split(x_patient_structured, 1, dim=-1)

        if len(split_data) != len(self.modality_order):
             raise ValueError(f"Number of modalities in input ({len(split_data)}) does not match expected modality_order ({len(self.modality_order)})")

        for i, mod_name in enumerate(self.modality_order):
            mod_tensor = split_data[i] # Shape (B, N, 1)
            mod_encoder = self.modality_encoders[mod_name]
            latent_mod = mod_encoder(mod_tensor) # Shape (B, N, latent_dim)
            
            # Apply mask if available for this modality
            if self.gene_masks is not None and mod_name in self.gene_masks:
                # Create mask tensor and expand to match batch dimension
                mask = torch.tensor(self.gene_masks[mod_name], 
                                   dtype=latent_mod.dtype, 
                                   device=latent_mod.device)
                # Reshape mask to [1, num_genes, 1] for broadcasting
                mask = mask.view(1, -1, 1)
                # Apply mask - this focuses the model on originally present genes
                # Values from added genes (mask=0) will not contribute to the latent representation
                latent_mod = latent_mod * mask
            
            latent_modality_tensors.append(latent_mod)

        # Concatenate latent modality tensors: -> (B, N, total_latent_dim)
        latent_omics_cat = torch.cat(latent_modality_tensors, dim=-1)

        # 2. Combine with Z_gene
        # Expand z_gene: (N, G_Emb) -> (B, N, G_Emb)
        z_gene_expanded = z_gene.unsqueeze(0).expand(batch_size, -1, -1)
        # Concatenate: (B, N, total_latent_dim + gene_embedding_dim)
        combined_features = torch.cat([latent_omics_cat, z_gene_expanded], dim=-1)

        # Process combined features per gene: -> (B, N, gene_combiner_out_dim)
        combined_gene_reps = self.gene_combiner(combined_features)

        # 3. Aggregate across genes with masked weighted pooling if masks are available
        if self.gene_masks is not None:
            # Create a combined mask across all modalities
            # A gene is considered valid if it appears in ANY modality (logical OR)
            combined_mask = None
            for mod_name in self.modality_order:
                if mod_name in self.gene_masks:
                    mod_mask = torch.tensor(self.gene_masks[mod_name], 
                                          dtype=torch.float, 
                                          device=combined_gene_reps.device)
                    if combined_mask is None:
                        combined_mask = mod_mask
                    else:
                        combined_mask = torch.maximum(combined_mask, mod_mask)
            
            # If we have any valid masks, use weighted pooling
            if combined_mask is not None:
                # Reshape mask for broadcasting: [num_genes] -> [1, num_genes, 1]
                combined_mask = combined_mask.view(1, -1, 1)
                # Apply mask and compute weighted mean
                masked_sum = (combined_gene_reps * combined_mask).sum(dim=1)
                mask_sum = combined_mask.sum(dim=1) + 1e-8  # Avoid division by zero
                aggregated_rep = masked_sum / mask_sum
            else:
                # Fallback to simple mean if no masks
                aggregated_rep = torch.mean(combined_gene_reps, dim=1)
        else:
            # Original simple mean pooling if no masks
            aggregated_rep = torch.mean(combined_gene_reps, dim=1)

        # 4. Final projection to patient embedding z_p
        # (B, gene_combiner_out_dim) -> (B, patient_embedding_dim)
        z_p = self.patient_aggregator(aggregated_rep)

        return z_p

    def decode(self, z_p, z_gene):
        """
        Decodes patient embedding z_p back to structured omics data, using gene embeddings.
        Reconstructs latent modalities first, then applies modality-specific decoders.

        Args:
            z_p (Tensor): Patient latent embedding (batch_size, patient_embedding_dim).
            z_gene (Tensor): Gene embeddings (num_genes, gene_embedding_dim).

        Returns:
            Tensor: Reconstructed omics data (batch_size, num_genes, num_modalities_in).
        """
        batch_size = z_p.shape[0]

        # 1. Decode z_p to patient context
        # (B, patient_embedding_dim) -> (B, total_latent_dim)
        patient_context = self.patient_decoder(z_p)

        # 2. Prepare for combination with z_gene
        # Expand patient_context: (B, total_latent_dim) -> (B, 1, total_latent_dim) -> (B, N, total_latent_dim)
        patient_context_expanded = patient_context.unsqueeze(1).expand(-1, self.num_genes, -1)
        # Expand z_gene: (N, gene_embedding_dim) -> (1, N, gene_embedding_dim) -> (B, N, gene_embedding_dim)
        z_gene_expanded = z_gene.unsqueeze(0).expand(batch_size, -1, -1)

        # 3. Combine patient context and gene embedding
        # (B, N, total_latent_dim + gene_embedding_dim)
        combined_decoder_input = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)

        # 4. Reconstruct concatenated latent omics representation per gene
        # (B, N, total_latent_dim + gene_embedding_dim) -> (B, N, total_latent_dim)
        rec_latent_omics_cat = self.reconstruction_mlp(combined_decoder_input)

        # 5. Split concatenated latent and apply modality-specific decoders
        reconstructed_modality_tensors = []
        current_dim = 0
        # Important: iterate through modality_order to maintain correct output order
        for mod_name in self.modality_order:
            latent_dim = self.modality_latent_dims[mod_name]
            # Extract the latent slice for this modality
            rec_latent_mod = rec_latent_omics_cat[:, :, current_dim : current_dim + latent_dim] # Shape (B, N, latent_dim)
            current_dim += latent_dim

            # Apply the specific decoder
            mod_decoder = self.modality_decoders[mod_name]
            rec_mod_data = mod_decoder(rec_latent_mod) # Shape (B, N, 1)
            reconstructed_modality_tensors.append(rec_mod_data)

        # 6. Concatenate reconstructed modalities
        # List of tensors (each B, N, 1) -> Tensor (B, N, M_in)
        reconstructed_structured = torch.cat(reconstructed_modality_tensors, dim=-1)

        # 7. Apply final activation
        omics_reconstructed = self.final_activation(reconstructed_structured)

        return omics_reconstructed


# Joint Autoencoder Wrapper (Now incorporating VGAE for graph part)
class JointAutoencoder(nn.Module):
    def __init__(self, num_nodes, modality_latent_dims: dict, modality_order: list,
                 graph_feature_dim, gene_embedding_dim, patient_embedding_dim, graph_dropout=0.5, gene_masks=None):
        """
        Args:
            num_nodes (int): Number of genes.
            modality_latent_dims (dict): Dict mapping modality name to latent dim. Passed to OmicsProcessor.
            modality_order (list): Order of modalities. Passed to OmicsProcessor.
            graph_feature_dim (int): Input feature dimension for the graph AE's initial nodes.
            gene_embedding_dim (int): Latent dimension for gene embeddings (Z_gene).
            patient_embedding_dim (int): Latent dimension for patient embeddings (z_p).
            graph_dropout (float): Dropout for the graph AE.
            gene_masks (dict, optional): Dictionary mapping modality names to binary masks (1 for originally present, 0 for added).
        """
        super(JointAutoencoder, self).__init__()
        self.num_nodes = num_nodes # Store num_nodes for KL loss calculation later
        self.graph_autoencoder = InteractionGraphAutoencoder(
            feature_dim=graph_feature_dim,
            gene_embedding_dim=gene_embedding_dim,
            dropout=graph_dropout
        )
        self.omics_processor = OmicsProcessor(
            modality_latent_dims=modality_latent_dims,
            modality_order=modality_order,
            gene_embedding_dim=gene_embedding_dim,
            patient_embedding_dim=patient_embedding_dim,
            num_genes=num_nodes,
            gene_masks=gene_masks
        )
        
    def forward(self, graph_x, graph_edge_index, omics_x_structured, graph_edge_weight=None):
        """
        Performs the full joint forward pass with VGAE.

        Args:
            graph_x (Tensor): Initial node features for the graph AE.
            graph_edge_index (LongTensor): Graph connectivity.
            omics_x_structured (Tensor): Batch of patient omics data.
            graph_edge_weight (Tensor, optional): Edge weights for the graph AE.

        Returns:
            tuple: Contains:
                - omics_reconstructed (Tensor)
                - adj_reconstructed (Tensor)
                - z_patient (Tensor)
                - z_gene (Tensor): Sampled gene embeddings.
                - mu (Tensor): Mean of the gene embedding distribution.
                - log_var (Tensor): Log variance of the gene embedding distribution.
        """
        # 1. Get gene embeddings (mu, log_var, sampled z_gene) from graph VGAE
        mu, log_var, z_gene = self.graph_autoencoder.encode(graph_x, graph_edge_index, edge_weight=graph_edge_weight)

        # 2. Encode patient omics using OmicsProcessor and the *sampled* z_gene
        z_patient = self.omics_processor.encode(omics_x_structured, z_gene)

        # 3. Decode patient embedding back to omics, *using sampled z_gene*
        omics_reconstructed = self.omics_processor.decode(z_patient, z_gene)

        # 4. Decode *sampled* gene embeddings back to adjacency matrix
        adj_reconstructed = self.graph_autoencoder.decode(z_gene)

        return omics_reconstructed, adj_reconstructed, z_patient, z_gene, mu, log_var



