import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F # Import functional
import torch.optim as optim
import pandas as pd
import torch_geometric as tg

# Graph Autoencoder (responsible for gene embeddings Z_gene)
# Remains largely the same, but ensure hidden_dim matches the desired gene_embedding_dim
class InteractionGraphAutoencoder(nn.Module):
    def __init__(self, feature_dim, gene_embedding_dim, dropout=0.5):
        """
        Args:
            feature_dim (int): Dimension of initial node features (e.g., num_genes for identity).
            gene_embedding_dim (int): Dimension of the latent gene embeddings (Z_gene).
            dropout (float): Dropout probability.
        """
        super(InteractionGraphAutoencoder, self).__init__()
        # Using two GCN layers. Adjust complexity if needed.
        # Intermediate dimension can be tuned.
        hidden_channels_intermediate = gene_embedding_dim * 2
        self.conv1 = tg.nn.GCNConv(feature_dim, hidden_channels_intermediate)
        # Output dimension is the final gene embedding size
        self.conv2 = tg.nn.GCNConv(hidden_channels_intermediate, gene_embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        """Encodes the graph into latent gene embeddings (Z_gene)."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # No activation on the final embedding layer generally
        z_gene = self.conv2(x, edge_index)
        return z_gene

    def decode(self, z_gene):
        """Reconstructs the adjacency matrix from latent gene embeddings."""
        adj_rec = torch.sigmoid(z_gene @ z_gene.t())
        return adj_rec

    def forward(self, x, edge_index):
        """Full forward pass for standalone graph AE (if needed) or just encoding"""
        z_gene = self.encode(x, edge_index)
        # Decoding might happen separately in the joint model using z_gene
        # adj_reconstructed = self.decode(z_gene)
        # return adj_reconstructed, z_gene
        return z_gene # Primarily used for encoding in the joint model


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

# Joint Autoencoder Wrapper
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

    def forward(self, graph_x, graph_edge_index, omics_x_structured):
        """
        Performs the full joint forward pass.

        Args:
            graph_x (Tensor): Initial node features for the graph AE (e.g., identity matrix) (num_nodes, graph_feature_dim).
            graph_edge_index (LongTensor): Graph connectivity (2, num_edges).
            omics_x_structured (Tensor): Batch of patient omics data (batch_size, num_nodes, num_modalities).

        Returns:
            tuple: Contains:
                - omics_reconstructed (Tensor): Reconstructed omics data (batch_size, num_nodes, num_modalities).
                - adj_reconstructed (Tensor): Reconstructed adjacency matrix (num_nodes, num_nodes).
                - z_patient (Tensor): Patient latent embeddings (batch_size, patient_embedding_dim).
                - z_gene (Tensor): Gene latent embeddings (num_nodes, gene_embedding_dim).
        """
        # 1. Get gene embeddings from graph AE
        z_gene = self.graph_autoencoder.encode(graph_x, graph_edge_index) # (N, gene_emb_dim)

        # 2. Encode patient omics using OmicsProcessor and z_gene
        z_patient = self.omics_processor.encode(omics_x_structured, z_gene) # (B, patient_emb_dim)

        # 3. Decode patient embedding back to omics
        omics_reconstructed = self.omics_processor.decode(z_patient) # (B, N, M)

        # 4. Decode gene embeddings back to adjacency matrix
        adj_reconstructed = self.graph_autoencoder.decode(z_gene) # (N, N)

        return omics_reconstructed, adj_reconstructed, z_patient, z_gene

# Remove the old OmicsAutoencoder class
# class OmicsAutoencoder(nn.Module):
#    ... (definition removed) ...

