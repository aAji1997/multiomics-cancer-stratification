import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F # Import functional
import torch.optim as optim
import pandas as pd
import torch_geometric as tg

# Graph Autoencoder (responsible for gene embeddings Z_gene)
# Modified to be a Variational Graph Autoencoder (VGAE)
class InteractionGraphAutoencoder(nn.Module):
    def __init__(self, feature_dim, gene_embedding_dim, dropout=0.5):
        """
        Args:
            feature_dim (int): Dimension of initial node features.
            gene_embedding_dim (int): Dimension of the latent gene embeddings (Z_gene).
            dropout (float): Dropout probability.
        """
        super(InteractionGraphAutoencoder, self).__init__()
        hidden_channels_intermediate = gene_embedding_dim * 2
        self.conv1 = tg.nn.GCNConv(feature_dim, hidden_channels_intermediate)
        self.norm1 = nn.LayerNorm(hidden_channels_intermediate)
        
        # Output layers for mean (mu) and log variance (log_var)
        # GCNConv outputting 2*embedding_dim, split later
        self.conv_out = tg.nn.GCNConv(hidden_channels_intermediate, gene_embedding_dim * 2)
        
        self.dropout_layer = nn.Dropout(dropout) # Renamed from self.dropout to avoid conflict

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
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.dropout_layer(x)
        
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

    # Standalone forward might not be needed if only using encode/decode in JointAE
    def forward(self, x, edge_index, edge_weight=None):
        mu, log_var, z_gene = self.encode(x, edge_index, edge_weight)
        adj_reconstructed = self.decode(z_gene)
        return adj_reconstructed, mu, log_var, z_gene


# Modified Omics Processor (integrates Z_gene, encodes/decodes patient omics)
class OmicsProcessor(nn.Module):
    def __init__(self, num_modalities, gene_embedding_dim, patient_embedding_dim, num_genes):
        """
        Args:
            num_modalities (int): Number of omics types per gene (e.g., 4).
            gene_embedding_dim (int): Dimension of Z_gene from the graph AE.
            patient_embedding_dim (int): Desired dimension for the final patient embedding (z_p).
            num_genes (int): Total number of genes.
        """
        super(OmicsProcessor, self).__init__()
        self.num_genes = num_genes
        self.num_modalities = num_modalities
        self.gene_embedding_dim = gene_embedding_dim
        self.patient_embedding_dim = patient_embedding_dim

        # Encoder Part
        # 1. Process per-gene omics modalities
        modality_processor_out_dim = 32 # Tunable intermediate dimension
        self.modality_processor = nn.Sequential(
            nn.Linear(num_modalities, modality_processor_out_dim),
            nn.ReLU()
            # Potentially add more layers here
        )

        # 2. Combine processed modalities with gene embedding (Z_gene)
        combined_feature_dim = modality_processor_out_dim + gene_embedding_dim
        gene_combiner_out_dim = 64 # Tunable intermediate dimension
        self.gene_combiner = nn.Sequential(
            nn.Linear(combined_feature_dim, gene_combiner_out_dim),
            nn.ReLU()
            # Potentially add more layers here
        )

        # 3. Aggregate gene representations into a patient embedding (z_p)
        # Using simple mean pooling for now. Attention could be an upgrade.
        # self.attention_pool = ... # e.g., torch.nn.MultiheadAttention + pooling
        self.patient_aggregator = nn.Sequential(
            # Optional: Add layers operating on aggregated representation
            nn.Linear(gene_combiner_out_dim, patient_embedding_dim),
            # Consider activation (e.g., ReLU) if needed before final z_p
        )


        # Decoder Part
        # Reconstruct the structured omics tensor (num_genes, num_modalities) from z_p
        # Simple MLP decoder for now
        self.decoder = nn.Sequential(
            nn.Linear(patient_embedding_dim, patient_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(patient_embedding_dim * 2, num_genes * num_modalities),
            nn.Sigmoid() # Assuming omics data is scaled [0, 1]
        )

    def encode(self, x_patient_structured, z_gene):
        """
        Encodes structured patient omics data, integrating graph-based gene embeddings.

        Args:
            x_patient_structured (Tensor): Patient's omics data (batch_size, num_genes, num_modalities).
            z_gene (Tensor): Gene embeddings from graph AE (num_genes, gene_embedding_dim).

        Returns:
            Tensor: Patient latent embedding (z_p) (batch_size, patient_embedding_dim).
        """
        batch_size = x_patient_structured.shape[0]

        # 1. Process modalities per gene: (B, N, M) -> (B, N, modality_processor_out_dim)
        processed_modalities = self.modality_processor(x_patient_structured)

        # 2. Combine with Z_gene
        # Expand z_gene to match batch size: (N, G_Emb) -> (B, N, G_Emb)
        z_gene_expanded = z_gene.unsqueeze(0).expand(batch_size, -1, -1)
        # Concatenate: (B, N, modality_processor_out_dim + gene_embedding_dim)
        combined_features = torch.cat([processed_modalities, z_gene_expanded], dim=-1)

        # Process combined features per gene: -> (B, N, gene_combiner_out_dim)
        combined_gene_reps = self.gene_combiner(combined_features)

        # 3. Aggregate across genes (simple mean pooling)
        # (B, N, gene_combiner_out_dim) -> (B, gene_combiner_out_dim)
        aggregated_rep = torch.mean(combined_gene_reps, dim=1)

        # 4. Final projection to patient embedding z_p
        # (B, gene_combiner_out_dim) -> (B, patient_embedding_dim)
        z_p = self.patient_aggregator(aggregated_rep)

        return z_p

    def decode(self, z_p):
        """
        Decodes patient embedding z_p back to structured omics data.

        Args:
            z_p (Tensor): Patient latent embedding (batch_size, patient_embedding_dim).

        Returns:
            Tensor: Reconstructed omics data (batch_size, num_genes, num_modalities).
        """
        batch_size = z_p.shape[0]
        # (B, patient_embedding_dim) -> (B, N * M)
        reconstructed_flat = self.decoder(z_p)
        # Reshape: (B, N * M) -> (B, N, M)
        reconstructed_structured = reconstructed_flat.view(batch_size, self.num_genes, self.num_modalities)
        return reconstructed_structured


# Joint Autoencoder Wrapper (Now incorporating VGAE for graph part)
class JointAutoencoder(nn.Module):
    def __init__(self, num_nodes, num_modalities, graph_feature_dim,
                 gene_embedding_dim, patient_embedding_dim, graph_dropout=0.5):
        """
        Args:
            num_nodes (int): Number of genes.
            num_modalities (int): Number of omics modalities per gene.
            graph_feature_dim (int): Input feature dimension for the graph AE's initial nodes.
            gene_embedding_dim (int): Latent dimension for gene embeddings (Z_gene).
            patient_embedding_dim (int): Latent dimension for patient embeddings (z_p).
            graph_dropout (float): Dropout for the graph AE.
        """
        super(JointAutoencoder, self).__init__()
        self.num_nodes = num_nodes # Store num_nodes for KL loss calculation later
        self.graph_autoencoder = InteractionGraphAutoencoder(
            feature_dim=graph_feature_dim,
            gene_embedding_dim=gene_embedding_dim,
            dropout=graph_dropout
        )
        self.omics_processor = OmicsProcessor(
            num_modalities=num_modalities,
            gene_embedding_dim=gene_embedding_dim,
            patient_embedding_dim=patient_embedding_dim,
            num_genes=num_nodes
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

        # 3. Decode patient embedding back to omics
        omics_reconstructed = self.omics_processor.decode(z_patient)

        # 4. Decode *sampled* gene embeddings back to adjacency matrix
        adj_reconstructed = self.graph_autoencoder.decode(z_gene)

        return omics_reconstructed, adj_reconstructed, z_patient, z_gene, mu, log_var



