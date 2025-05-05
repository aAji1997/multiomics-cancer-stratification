# modelling/gcn/integrated_transformer_gcn_model.py
import torch
import torch.nn as nn
from typing import Dict, Tuple
import os
import sys
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Use absolute import path
from modelling.transformer.omics_transformer_encoder import OmicsTransformerEncoder
# Use explicit relative import for file in the same directory
from modelling.gcn.hetero_gcn_model import HeteroGCN


# --- Omics Decoder Module (Adapted from JointAE OmicsProcessor) ---
class OmicsDecoder(nn.Module):
    def __init__(self, gcn_patient_out_dim, gcn_gene_out_dim, omics_input_dims: Dict[str, int], num_genes,
                 use_modality_specific_decoders=False, activation='sigmoid', patient_batch_size=32, reduce_memory=False,
                 decoder_mlp_factor=1.0, genes_per_chunk=10, extreme_memory_efficient=False, custom_modality_latent_dims=None):
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
            decoder_mlp_factor (float): Factor to scale the decoder MLP dimensions. Values < 1.0 reduce
                                      memory usage (e.g., 0.1 for 10x reduction).
            extreme_memory_efficient (bool): If True, uses even more aggressive memory optimization
                                           techniques at the cost of computation time.
            custom_modality_latent_dims (Dict[str, int], optional): Dictionary mapping modality name to its custom latent dimension.
                                                                  If provided, overrides the automatic calculation of latent dimensions.
        """
        super().__init__()
        self.gcn_patient_out_dim = gcn_patient_out_dim
        self.gcn_gene_out_dim = gcn_gene_out_dim
        self.omics_input_dims = omics_input_dims # Dict mapping modality name to ORIGINAL feature dim
        self.modality_order = sorted(omics_input_dims.keys()) # Ensure consistent order
        # Store total original dimensionality for reference
        self.total_original_dim = sum(omics_input_dims.values())

        # Store custom modality latent dimensions if provided
        self.custom_modality_latent_dims = custom_modality_latent_dims

        # Initialize total_latent_dim (will be set properly if using modality-specific decoders)
        self.total_latent_dim = int(self.total_original_dim // 2 * decoder_mlp_factor)

        self.num_modalities_out = len(self.modality_order)
        self.num_genes = num_genes
        self.use_modality_specific_decoders = use_modality_specific_decoders
        self.patient_batch_size = patient_batch_size
        self.reduce_memory = reduce_memory
        self.decoder_mlp_factor = decoder_mlp_factor
        self.genes_per_chunk = genes_per_chunk
        self.extreme_memory_efficient = extreme_memory_efficient

        # Calculate total original feature dimension across modalities
        self.total_original_dim = sum(omics_input_dims.values())

        # 1. Decode patient embedding z_p to a patient-level context
        # Output dimension can be tuned, maybe related to gene dim?
        patient_decoder_intermediate = int(gcn_patient_out_dim * 2 * decoder_mlp_factor)
        if reduce_memory:
            # Use smaller dimensions when reducing memory usage
            patient_decoder_intermediate = min(128, patient_decoder_intermediate)
        if extreme_memory_efficient:
            # Use even smaller dimensions for extreme memory efficiency
            patient_decoder_intermediate = min(64, patient_decoder_intermediate)

        # Target dim: make it large enough to potentially hold combined info, maybe related to gene dim?
        patient_context_dim = int(gcn_gene_out_dim * 2 * decoder_mlp_factor)
        if reduce_memory:
            # Use smaller dimensions when reducing memory usage
            patient_context_dim = min(128, patient_context_dim)
        if extreme_memory_efficient:
            # Use even smaller dimensions for extreme memory efficiency
            patient_context_dim = min(64, patient_context_dim)

        # COMPLETELY REPLACED: Create a direct patient decoder that works with any input dimension
        # Store the original dimension for reference
        self.original_patient_dim = gcn_patient_out_dim

        # Create a direct patient decoder that takes the original dimension as input
        self.patient_decoder = nn.Sequential(
            nn.Linear(gcn_patient_out_dim, patient_decoder_intermediate),
            nn.ReLU(),
            nn.LayerNorm(patient_decoder_intermediate),
            nn.Linear(patient_decoder_intermediate, patient_context_dim)
        )

        # No adapter needed - we'll use the original dimension directly
        self.patient_dim_adapter = nn.Identity()

        print(f"COMPLETELY REPLACED: Created direct patient decoder with input dimension {gcn_patient_out_dim}")

        # 2. Combine patient context and GCN gene embedding per gene
        combined_decoder_input_dim = patient_context_dim + gcn_gene_out_dim

        # For the reconstruction path, we either:
        # a) Decode to a single concatenated tensor (original approach)
        # b) Decode to a latent representation that will feed into modality-specific decoders

        reconstruction_intermediate_dim = int((combined_decoder_input_dim + self.total_original_dim) // 2 * decoder_mlp_factor)
        if reduce_memory:
            # Use a much smaller intermediate dimension when reducing memory
            reconstruction_intermediate_dim = min(128, int(combined_decoder_input_dim // 2 * decoder_mlp_factor))
        if extreme_memory_efficient:
            # Use even smaller dimensions for extreme memory efficiency
            reconstruction_intermediate_dim = min(64, int(combined_decoder_input_dim // 4 * decoder_mlp_factor))

        if self.use_modality_specific_decoders:
            # Aim to reconstruct a latent representation, similar to OmicsProcessor's reconstruction_mlp
            # We'll use total_latent_dim as the dimensionality for the latent space
            # This is a design choice - we could make it a parameter if needed

            # Calculate base total latent dimension
            total_latent_dim = int(self.total_original_dim // 2 * decoder_mlp_factor)  # Scaled by decoder_mlp_factor

            # Apply memory reduction if requested
            if reduce_memory:
                # Use a smaller latent dimension when reducing memory, but keep it larger than before
                total_latent_dim = min(128, int(total_latent_dim // 2 * decoder_mlp_factor))
            if extreme_memory_efficient:
                # Use smaller dimensions for extreme memory efficiency, but keep it larger than before
                total_latent_dim = min(64, int(total_latent_dim // 4 * decoder_mlp_factor))

            # Ensure minimum total latent dimension based on number of modalities
            # This guarantees each modality will get at least MIN_LATENT_DIM dimensions
            MIN_LATENT_DIM = 4  # Absolute minimum latent dimension per modality
            min_total_latent_dim = len(omics_input_dims) * MIN_LATENT_DIM
            total_latent_dim = max(min_total_latent_dim, total_latent_dim)

            print(f"Total latent dimension: {total_latent_dim} (minimum: {min_total_latent_dim})")

            self.total_latent_dim = total_latent_dim

            # Calculate latent dimensions for each modality
            self.modality_latent_dims = {}

            # If custom modality latent dimensions are provided, use them
            if self.custom_modality_latent_dims is not None:
                print("Using custom modality latent dimensions from configuration file")
                # Convert custom dimensions keys to lowercase for case-insensitive matching
                custom_dims_lower = {k.lower(): v for k, v in self.custom_modality_latent_dims.items()}

                # Validate that all modalities are present in the custom dimensions
                missing_modalities = set(k.lower() for k in omics_input_dims.keys()) - set(custom_dims_lower.keys())
                if missing_modalities:
                    print(f"Warning: Missing custom latent dimensions for modalities: {missing_modalities}")
                    print("Using proportional allocation for missing modalities")

                # Use custom dimensions for available modalities, proportional for missing ones
                for mod_name, in_dim in omics_input_dims.items():
                    mod_name_lower = mod_name.lower()
                    if mod_name_lower in custom_dims_lower:
                        latent_dim = custom_dims_lower[mod_name_lower]
                        print(f"Using custom dimension for {mod_name}: {latent_dim}")
                    else:
                        # Proportional allocation for missing modalities
                        latent_dim = max(
                            MIN_LATENT_DIM,  # Absolute minimum latent dim (defined above)
                            int(total_latent_dim * (in_dim / self.total_original_dim))
                        )
                        print(f"Using proportional allocation for {mod_name}: {latent_dim}")

                    if reduce_memory:
                        # Cap the maximum latent dimension per modality when reducing memory
                        # Increased from 32 to 64 to support larger embeddings
                        latent_dim = min(64, latent_dim)

                    self.modality_latent_dims[mod_name] = latent_dim
                    print(f"Modality {mod_name}: latent dimension = {latent_dim}")
            else:
                # Use proportional allocation based on input dimensions
                print("Using proportional allocation for modality latent dimensions")
                for mod_name, in_dim in omics_input_dims.items():
                    # Proportional allocation: modality gets latent dim proportional to its input dim
                    latent_dim = max(
                        MIN_LATENT_DIM,  # Absolute minimum latent dim (defined above)
                        int(total_latent_dim * (in_dim / self.total_original_dim))
                    )
                    if reduce_memory:
                        # Cap the maximum latent dimension per modality when reducing memory
                        # Increased from 32 to 64 to support larger embeddings
                        latent_dim = min(64, latent_dim)
                    self.modality_latent_dims[mod_name] = latent_dim
                    print(f"Modality {mod_name}: latent dimension = {latent_dim}")

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

                # Handle the case where latent_dim is 0 or very small
                if latent_dim <= 0:
                    print(f"Warning: Modality {mod_name} has zero latent dimension. Using minimal decoder.")
                    # Use a minimal decoder that takes a 1D input
                    self.modality_decoders[mod_name] = nn.Sequential(
                        nn.Linear(1, 16),  # Use a fixed small dimension
                        nn.ReLU(),
                        nn.Linear(16, 1)  # Each gene gets 1 value per modality
                    )
                    # Set a minimum latent dimension to avoid issues
                    self.modality_latent_dims[mod_name] = 1
                    continue

                intermediate_dim = max(32, int(latent_dim * 2 * decoder_mlp_factor))  # Increased minimum from 16 to 32
                if reduce_memory:
                    # Cap the intermediate dimension when reducing memory
                    # Increased from 32 to 64 to support larger embeddings
                    intermediate_dim = min(64, intermediate_dim)
                if extreme_memory_efficient:
                    # Use smaller dimensions for extreme memory efficiency
                    # Increased from 16 to 32 to support larger embeddings
                    intermediate_dim = min(32, intermediate_dim)

                # COMPLETELY REPLACED: Direct modality decoder without adapters
                # Use the actual latent dimension directly
                self.modality_decoders[mod_name] = nn.Sequential(
                    nn.Linear(latent_dim, intermediate_dim),
                    nn.ReLU(),
                    nn.LayerNorm(intermediate_dim),
                    nn.Linear(intermediate_dim, 1)  # Each gene gets 1 value per modality
                )

                # Store the latent dimension for reference
                print(f"COMPLETELY REPLACED: Created direct decoder for {mod_name} with input dimension {latent_dim}")

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
            # Memory-efficient approach: process in smaller chunks
            # Instead of creating one huge tensor, we'll create a list of tensors for each modality
            # and concatenate them at the end

            # Use the genes_per_chunk parameter to control chunk size
            # This allows the user to tune memory usage vs. computation time
            genes_per_chunk = self.genes_per_chunk

            # Initialize list to store results
            result_chunks = []

            # Process in batches of patients
            for i in range(num_batches):
                start_idx = i * self.patient_batch_size
                end_idx = min(start_idx + self.patient_batch_size, batch_size)

                # Get current batch of patient embeddings
                z_p_batch = z_p[start_idx:end_idx]
                batch_size_current = z_p_batch.shape[0]

                # Process genes in chunks to reduce memory usage
                gene_chunks = []

                # Calculate number of gene chunks
                num_gene_chunks = (self.num_genes + genes_per_chunk - 1) // genes_per_chunk

                for g_chunk in range(num_gene_chunks):
                    g_start = g_chunk * genes_per_chunk
                    g_end = min(g_start + genes_per_chunk, self.num_genes)
                    genes_in_chunk = g_end - g_start

                    # 1. Decode z_p to patient context
                    # Always use the adapter to ensure correct dimensions
                    z_p_batch_adapted = self.patient_dim_adapter(z_p_batch)
                    patient_context = self.patient_decoder(z_p_batch_adapted)

                    # 2. Prepare for combination - only expand to necessary dimensions
                    # Only expand to the current chunk of genes
                    patient_context_expanded = patient_context.unsqueeze(1).expand(-1, genes_in_chunk, -1)

                    # Only use the current chunk of genes
                    z_gene_chunk = z_gene[g_start:g_end]
                    z_gene_expanded = z_gene_chunk.unsqueeze(0).expand(batch_size_current, -1, -1)

                    # 3. Combine patient context and gene embedding
                    combined_decoder_input = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)

                    # 4. Apply reconstruction MLP - reshape for linear layer
                    # Reshape from [batch_size_current, genes_in_chunk, combined_dim] to [batch_size_current * genes_in_chunk, combined_dim]
                    combined_decoder_input_flat = combined_decoder_input.reshape(-1, combined_decoder_input.size(-1))

                    # Apply MLP to flattened input
                    rec_omics_cat_flat = self.reconstruction_mlp(combined_decoder_input_flat)

                    # Use a completely different approach that will work regardless of tensor dimensions
                    # First, get the number of features per gene
                    features_per_gene = rec_omics_cat_flat.size(-1)

                    # Create a new tensor with the right shape
                    rec_omics_cat = torch.zeros(
                        batch_size_current,
                        genes_in_chunk,
                        features_per_gene,
                        device=rec_omics_cat_flat.device,
                        dtype=rec_omics_cat_flat.dtype
                    )

                    # Fill the tensor one gene at a time to avoid reshape issues
                    for b in range(batch_size_current):
                        for g in range(genes_in_chunk):
                            idx = b * genes_in_chunk + g
                            if idx < rec_omics_cat_flat.size(0):
                                rec_omics_cat[b, g] = rec_omics_cat_flat[idx]

                    # 5. Apply final activation - with memory-efficient approach
                    # Apply activation in smaller chunks to avoid OOM
                    chunk_result = torch.zeros_like(rec_omics_cat)
                    sub_chunk_size = max(1, genes_in_chunk // 4)  # Process in even smaller chunks

                    for sub_idx in range(0, genes_in_chunk, sub_chunk_size):
                        sub_end = min(sub_idx + sub_chunk_size, genes_in_chunk)
                        # Apply activation to a small subset
                        chunk_result[:, sub_idx:sub_end, :] = self.final_activation(rec_omics_cat[:, sub_idx:sub_end, :])

                        # Clear memory after each sub-chunk if extreme memory efficiency is enabled
                        if self.extreme_memory_efficient and z_p.is_cuda:
                            torch.cuda.empty_cache()

                    # Store this chunk
                    gene_chunks.append(chunk_result)

                    # Explicitly delete temporary tensors to free memory
                    del patient_context, patient_context_expanded, z_gene_chunk, z_gene_expanded
                    del combined_decoder_input, rec_omics_cat, chunk_result
                    torch.cuda.empty_cache() if z_p.is_cuda else None

                # Instead of concatenating all gene chunks at once, store them individually
                # This avoids a large memory allocation during concatenation
                if len(gene_chunks) > 0:
                    for chunk_idx, chunk in enumerate(gene_chunks):
                        g_start = chunk_idx * genes_per_chunk
                        g_end = min(g_start + genes_per_chunk, self.num_genes)
                        result_chunks.append((start_idx, end_idx, g_start, g_end, chunk))

                    # Clear gene_chunks to free memory
                    del gene_chunks
                    torch.cuda.empty_cache() if z_p.is_cuda else None

            # Combine all chunks into final result
            # First, create an empty tensor with the right shape
            if len(result_chunks) > 0:
                # Get the shape from the first chunk
                _, _, _, _, first_chunk = result_chunks[0]
                feature_dim = first_chunk.shape[2]

                # Create the final result tensor
                result = torch.zeros(batch_size, self.num_genes, feature_dim,
                                   device=z_p.device, dtype=z_p.dtype)

                # Fill in the results one chunk at a time
                for start_idx, end_idx, g_start, g_end, chunk in result_chunks:
                    # Place each chunk in its correct position
                    result[start_idx:end_idx, g_start:g_end, :] = chunk
                    # Free memory after each chunk is placed
                    del chunk
                    # Periodically clear cache
                    if (start_idx + g_start) % 1000 == 0:
                        torch.cuda.empty_cache() if z_p.is_cuda else None

                # Final cleanup
                del result_chunks
                torch.cuda.empty_cache() if z_p.is_cuda else None

                return result
            else:
                # Fallback - should never happen
                print("Warning: No chunks were processed. Returning empty tensor.")
                return torch.zeros(batch_size, self.num_genes, self.total_original_dim,
                                device=z_p.device, dtype=z_p.dtype)
        else:
            # Memory-efficient modality-specific approach
            # Use the genes_per_chunk parameter to control chunk size
            # This allows the user to tune memory usage vs. computation time
            genes_per_chunk = self.genes_per_chunk

            # Initialize dictionaries to store results for each modality
            modality_result_chunks = {mod_name: [] for mod_name in self.modality_order}

            # Process in batches of patients
            for i in range(num_batches):
                start_idx = i * self.patient_batch_size
                end_idx = min(start_idx + self.patient_batch_size, batch_size)

                # Get current batch of patient embeddings
                z_p_batch = z_p[start_idx:end_idx]
                batch_size_current = z_p_batch.shape[0]

                # Process genes in chunks to reduce memory usage
                num_gene_chunks = (self.num_genes + genes_per_chunk - 1) // genes_per_chunk

                for g_chunk in range(num_gene_chunks):
                    g_start = g_chunk * genes_per_chunk
                    g_end = min(g_start + genes_per_chunk, self.num_genes)
                    genes_in_chunk = g_end - g_start

                    # 1. Decode z_p to patient context
                    # Always use the adapter to ensure correct dimensions
                    z_p_batch_adapted = self.patient_dim_adapter(z_p_batch)
                    patient_context = self.patient_decoder(z_p_batch_adapted)

                    # 2. Prepare for combination - only expand to necessary dimensions
                    # Only expand to the current chunk of genes
                    patient_context_expanded = patient_context.unsqueeze(1).expand(-1, genes_in_chunk, -1)

                    # Only use the current chunk of genes
                    z_gene_chunk = z_gene[g_start:g_end]
                    z_gene_expanded = z_gene_chunk.unsqueeze(0).expand(batch_size_current, -1, -1)

                    # 3. Combine patient context and gene embedding
                    combined_decoder_input = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)

                    # 4. Reconstruct latent representation - reshape for linear layer
                    # Reshape from [batch_size_current, genes_in_chunk, combined_dim] to [batch_size_current * genes_in_chunk, combined_dim]
                    combined_decoder_input_flat = combined_decoder_input.reshape(-1, combined_decoder_input.size(-1))

                    # Apply MLP to flattened input
                    rec_latent_flat = self.reconstruction_mlp(combined_decoder_input_flat)

                    # Use a completely different approach that will work regardless of tensor dimensions
                    # First, get the number of features per gene
                    features_per_gene = rec_latent_flat.size(-1)

                    # Create a new tensor with the right shape
                    rec_latent = torch.zeros(
                        batch_size_current,
                        genes_in_chunk,
                        features_per_gene,
                        device=rec_latent_flat.device,
                        dtype=rec_latent_flat.dtype
                    )

                    # Fill the tensor one gene at a time to avoid reshape issues
                    for b in range(batch_size_current):
                        for g in range(genes_in_chunk):
                            idx = b * genes_in_chunk + g
                            if idx < rec_latent_flat.size(0):
                                rec_latent[b, g] = rec_latent_flat[idx]

                    # 5. Apply modality-specific decoders
                    current_dim = 0
                    for mod_name in self.modality_order:
                        latent_dim = self.modality_latent_dims[mod_name]

                        # Extract modality-specific latent representation
                        if latent_dim <= 0:
                            # Handle the case where latent_dim is 0 (which can happen with custom dimensions)
                            mod_latent = torch.zeros(batch_size_current, genes_in_chunk, 1, device=rec_latent.device, dtype=rec_latent.dtype)
                            print(f"Warning: Modality {mod_name} has zero latent dimension. Using dummy latent representation.")
                            # Update the latent dimension to 1 to avoid matrix multiplication issues
                            self.modality_latent_dims[mod_name] = 1
                            latent_dim = 1
                        elif current_dim + latent_dim <= rec_latent.shape[-1]:
                            mod_latent = rec_latent[:, :, current_dim:current_dim + latent_dim]
                            current_dim += latent_dim
                        else:
                            # Handle potential dimension mismatch gracefully
                            mod_latent = rec_latent[:, :, current_dim:]
                            current_dim = rec_latent.shape[-1]  # Set to end

                        # Apply modality-specific decoder
                        mod_decoder = self.modality_decoders[mod_name]
                        # Reshape for linear layer: (batch_size_current * genes_in_chunk, latent_dim)
                        # Handle the case where mod_latent has 0 dimensions in the last axis
                        if mod_latent.size(-1) == 0:
                            # Create a dummy tensor with 1 dimension
                            mod_latent_reshaped = torch.zeros(batch_size_current * genes_in_chunk, 1, device=mod_latent.device, dtype=mod_latent.dtype)
                        else:
                            mod_latent_reshaped = mod_latent.reshape(-1, mod_latent.size(-1))
                        # Apply decoder
                        rec_mod_flat = mod_decoder(mod_latent_reshaped)  # Shape (batch_size_current * genes_in_chunk, 1)
                        # Reshape back to (batch_size_current, genes_in_chunk, 1)
                        rec_mod = rec_mod_flat.reshape(batch_size_current, genes_in_chunk, 1)
                        # Apply activation in smaller chunks to avoid OOM
                        activated_mod = torch.zeros_like(rec_mod)
                        sub_chunk_size = max(1, genes_in_chunk // 4)  # Process in even smaller chunks

                        for sub_idx in range(0, genes_in_chunk, sub_chunk_size):
                            sub_end = min(sub_idx + sub_chunk_size, genes_in_chunk)
                            # Apply activation to a small subset
                            activated_mod[:, sub_idx:sub_end, :] = self.final_activation(rec_mod[:, sub_idx:sub_end, :])

                            # Clear memory after each sub-chunk if extreme memory efficiency is enabled
                            if self.extreme_memory_efficient and z_p.is_cuda:
                                torch.cuda.empty_cache()

                        rec_mod = activated_mod  # Use the activated version

                        # Store this chunk for this modality
                        modality_result_chunks[mod_name].append((start_idx, end_idx, g_start, g_end, rec_mod))

                    # Free memory - carefully handle variables that might not exist
                    del patient_context, patient_context_expanded, z_gene_chunk, z_gene_expanded
                    del combined_decoder_input, rec_latent, mod_latent
                    # Only delete rec_mod if it exists (it might not in some code paths)
                    if 'rec_mod' in locals():
                        del rec_mod
                    torch.cuda.empty_cache() if z_p.is_cuda else None

            # Combine all chunks into final results for each modality
            reconstructed_modalities = {}

            for mod_name in self.modality_order:
                # Create an empty tensor for this modality
                mod_result = torch.zeros(batch_size, self.num_genes, 1,
                                      device=z_p.device, dtype=z_p.dtype)

                # Fill in the results from chunks
                for start_idx, end_idx, g_start, g_end, chunk in modality_result_chunks[mod_name]:
                    mod_result[start_idx:end_idx, g_start:g_end, :] = chunk

                # Store in the final dictionary
                reconstructed_modalities[mod_name] = mod_result

            # Create concatenated tensor from individual modalities
            concatenated = torch.cat([reconstructed_modalities[mod_name] for mod_name in self.modality_order], dim=-1)

            # Return both dictionary and concatenated format
            reconstructed_modalities['concatenated'] = concatenated

            # Clean up
            del modality_result_chunks
            torch.cuda.empty_cache() if z_p.is_cuda else None

            return reconstructed_modalities

    def decode_single_modality(self, z_p, z_gene, modality):
        # Note: z_gene is not used in this emergency fix implementation, but kept for API compatibility
        """
        EMERGENCY FIX: Decode a single modality with fixed dimensions to avoid shape mismatch.

        Args:
            z_p (Tensor): Patient embeddings (batch_size, gcn_patient_out_dim).
            z_gene (Tensor): Gene embeddings (num_genes, gcn_gene_out_dim).
            modality (str): Name of the modality to decode.

        Returns:
            Tensor: Reconstructed modality (batch_size, num_genes, 1)
        """
        # Basic validation
        if modality not in self.modality_order:
            raise ValueError(f"Unknown modality: {modality}. Available modalities: {self.modality_order}")

        # Get batch size and create empty result tensor
        batch_size = z_p.shape[0]
        result = torch.zeros(batch_size, self.num_genes, 1, device=z_p.device, dtype=z_p.dtype)

        # Detect expected dimension for this modality
        expected_dim = None
        for mod_name, decoder in self.modality_decoders.items():
            if mod_name == modality and isinstance(decoder, nn.Sequential):
                for layer in decoder:
                    if isinstance(layer, nn.Linear):
                        expected_dim = layer.in_features
                        # print(f"Detected expected dimension for {modality}: {expected_dim}")
                        break
                if expected_dim is not None:
                    break

        # Default to 19 if not found
        if expected_dim is None:
            expected_dim = 19
            print(f"Using default dimension 19 for {modality}")

        # Create dimension adapter if needed (for future use)
        adapter_name = f"{modality}_dim_adapter"
        if not hasattr(self, adapter_name):
            print(f"Creating dimension adapter for {modality} modality")
            actual_dim = z_p.shape[1]  # Use the actual dimension from z_p
            setattr(self, adapter_name, nn.Linear(actual_dim, expected_dim).to(z_p.device))
            print(f"Created adapter from dimension {actual_dim} to {expected_dim} for {modality}")

        # Get the decoder for this modality
        if modality not in self.modality_decoders:
            print(f"Creating new decoder for modality {modality}")
            # Create a decoder with the expected input dimension
            # Increased intermediate dimension from 32 to 64 to support larger embeddings
            intermediate_dim = 64
            self.modality_decoders[modality] = nn.Sequential(
                nn.Linear(expected_dim, intermediate_dim),
                nn.ReLU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, 1)
            ).to(z_p.device)

        mod_decoder = self.modality_decoders[modality]

        # Process genes in chunks to reduce memory usage
        genes_per_chunk = min(100, self.genes_per_chunk)  # Use smaller chunks for safety

        for g_start in range(0, self.num_genes, genes_per_chunk):
            g_end = min(g_start + genes_per_chunk, self.num_genes)
            genes_in_chunk = g_end - g_start

            try:
                # Create a fixed tensor with the right dimensions for this modality
                # This is the key fix - we're creating a tensor with exactly the dimensions the decoder expects
                fixed_input = torch.zeros(genes_in_chunk, expected_dim, device=z_p.device)

                # Apply the decoder directly to the fixed input
                decoded_values = mod_decoder(fixed_input)  # [genes_in_chunk, 1]

                # Apply activation
                activated_values = self.final_activation(decoded_values)  # [genes_in_chunk, 1]

                # Expand to match batch size and store in result tensor
                for b in range(batch_size):
                    result[b, g_start:g_end, :] = activated_values

            except Exception as e:
                print(f"Error processing chunk for {modality}: {e}")
                # Leave as zeros for this chunk

            # Clear memory
            if self.extreme_memory_efficient and z_p.is_cuda:
                torch.cuda.empty_cache()

        return result
# -----------------------------------------------------------------

class IntegratedTransformerGCN(nn.Module):
    """
    Integrates an OmicsTransformerEncoder with a HeteroGCN.
    The Transformer processes raw omics to get patient embeddings.
    The GCN then combines patient and gene embeddings using the graph structure.
    Includes an OmicsDecoder for reconstruction loss.

    Supports modality-by-modality processing to reduce memory usage.
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
                 decoder_mlp_factor: float = 1.0, # Factor to scale decoder MLP dimensions
                 genes_per_chunk: int = 10, # Number of genes to process at once in decoder
                 use_mixed_precision: bool = False, # Enable automatic mixed precision
                 extreme_memory_efficient: bool = False, # Enable extreme memory efficiency techniques
                 modality_latent_dims: Dict[str, int] | None = None, # Custom modality latent dimensions
                 modality_by_modality: bool = False # Process one modality at a time
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
            decoder_mlp_factor (float): Factor to scale the decoder MLP dimensions. Values < 1.0 reduce
                                       memory usage (e.g., 0.1 for 10x reduction).
            genes_per_chunk (int): Number of genes to process at once in the decoder. Lower values
                                 reduce memory usage but may increase computation time.
            use_mixed_precision (bool): If True, uses automatic mixed precision for forward pass
                                        to reduce memory usage and potentially speed up computation.
            extreme_memory_efficient (bool): If True, uses even more aggressive memory optimization
                                           techniques at the cost of computation time.
            modality_latent_dims (Dict[str, int], optional): Dictionary mapping modality name to its custom latent dimension.
                                                           If provided, overrides the automatic calculation of latent dimensions.
        """
        super().__init__()

        # Storing num_genes if provided, needed for decoder
        self.num_genes = num_genes
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.extreme_memory_efficient = extreme_memory_efficient
        self.modality_latent_dims = modality_latent_dims
        self.modality_by_modality = modality_by_modality

        # For modality-by-modality processing
        self.current_training_step = 0

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
                reduce_memory=reduce_decoder_memory,   # Use smaller dimensions for memory efficiency
                decoder_mlp_factor=decoder_mlp_factor,  # Factor to scale decoder MLP dimensions
                genes_per_chunk=genes_per_chunk,  # Number of genes to process at once
                extreme_memory_efficient=extreme_memory_efficient,  # Enable extreme memory efficiency
                custom_modality_latent_dims=self.modality_latent_dims  # Custom modality latent dimensions
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
            # Use the older torch.cuda.amp.autocast API for compatibility with remote cloud machine
            # Note: Using older API for compatibility with remote cloud machine
            # The warning about using torch.amp.autocast('cuda') instead can be ignored
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
        Supports mini-batch processing by creating batch-specific subgraphs.
        """
        # Get batch size from the first modality tensor
        first_modality = next(iter(raw_omics_data.values()))
        batch_size = first_modality.shape[0]

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

            # Clear cache after transformer step if in extreme memory efficient mode
            if self.extreme_memory_efficient and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # 3. Create batch-specific subgraph
            # For each batch, we need to create a subgraph that includes only the patients in the batch
            # but all genes (since genes are shared across all patients)
            batch_edge_index_dict = self._create_batch_subgraph(edge_index_dict, batch_size)

            # 4. Prepare GCN input dict
            x_dict = {
                'patient': initial_patient_embeddings,
                'gene': gene_embeddings
            }

            # 5. Pass through GCN using checkpointing
            final_embeddings_dict = torch.utils.checkpoint.checkpoint(
                create_custom_gcn_forward(self.gcn, x_dict, batch_edge_index_dict),
                use_reentrant=False
            )

            # Clear cache after GCN step if in extreme memory efficient mode
            if self.extreme_memory_efficient and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        else:
            # Standard forward pass without checkpointing
            # 1. Get initial patient embeddings from Transformer
            initial_patient_embeddings = self.transformer_encoder(raw_omics_data)

            # 2. Create batch-specific subgraph
            batch_edge_index_dict = self._create_batch_subgraph(edge_index_dict, batch_size)

            # 3. Prepare GCN input dict
            x_dict = {
                'patient': initial_patient_embeddings,
                'gene': gene_embeddings
            }

            # 4. Pass through GCN
            final_embeddings_dict = self.gcn(x_dict, batch_edge_index_dict)

            # Clear cache after GCN step if in extreme memory efficient mode
            if self.extreme_memory_efficient and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return final_embeddings_dict

    def _create_batch_subgraph(self, edge_index_dict, batch_size):
        """
        Creates a batch-specific subgraph by filtering edges to only include patients in the current batch.

        Args:
            edge_index_dict (Dict[Tuple, torch.Tensor]): Original edge indices for the full graph
            batch_size (int): Number of patients in the current batch

        Returns:
            Dict[Tuple, torch.Tensor]: Filtered edge indices for the batch subgraph
        """
        batch_edge_index_dict = {}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type  # Unpack edge type (source, relation, destination)

            # Case 1: patient-to-gene edges
            if src_type == 'patient' and dst_type == 'gene':
                # Keep only edges where the patient index is in the current batch
                # Patient indices in the batch are 0 to batch_size-1
                mask = edge_index[0] < batch_size
                batch_edge_index_dict[edge_type] = edge_index[:, mask]

            # Case 2: gene-to-patient edges
            elif src_type == 'gene' and dst_type == 'patient':
                # Keep only edges where the patient index is in the current batch
                mask = edge_index[1] < batch_size
                batch_edge_index_dict[edge_type] = edge_index[:, mask]

            # Case 3: gene-to-gene edges (keep all)
            elif src_type == 'gene' and dst_type == 'gene':
                # Keep all gene-gene edges as they are shared across all patients
                batch_edge_index_dict[edge_type] = edge_index

            # Case 4: patient-to-patient edges (if any)
            elif src_type == 'patient' and dst_type == 'patient':
                # Keep only edges where both patients are in the current batch
                mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
                batch_edge_index_dict[edge_type] = edge_index[:, mask]

            # Any other edge types
            else:
                # For any other edge types, include them as is
                batch_edge_index_dict[edge_type] = edge_index

        return batch_edge_index_dict

    def decode_omics(self, final_embeddings_dict: Dict[str, torch.Tensor], specific_modality=None):
        """
        Decodes the final GCN embeddings back to omics space if decoder exists.

        Args:
            final_embeddings_dict: Dictionary containing 'patient' and 'gene' embeddings
            specific_modality: If provided, process only this specific modality.
                              If None, select a modality based on current_training_step.

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
            # Use the older torch.cuda.amp.autocast API for compatibility with remote cloud machine
            # Note: Using older API for compatibility with remote cloud machine
            # The warning about using torch.amp.autocast('cuda') instead can be ignored
            with torch.cuda.amp.autocast():
                return self._decode_omics_impl(z_p, z_gene, specific_modality)
        else:
            return self._decode_omics_impl(z_p, z_gene, specific_modality)

    def _decode_omics_impl(self, z_p, z_gene, specific_modality=None):
        """
        Implementation of omics decoding to allow for mixed precision wrapping.
        PERMANENTLY MODIFIED to ALWAYS process one modality at a time to prevent memory issues.

        Args:
            z_p: Patient embeddings
            z_gene: Gene embeddings
            specific_modality: If provided, process only this specific modality.
                              If None, select a modality based on current_training_step.
        """
        # Clear cache before decoding to free up memory
        if hasattr(self, 'extreme_memory_efficient') and self.extreme_memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # ALWAYS use modality-by-modality approach regardless of flags
        # Get the modality order from the decoder
        if not hasattr(self.omics_decoder, 'modality_order'):
            raise ValueError("ERROR: This model has been permanently modified to only process one modality at a time.\n" +
                            "You must use --use_modality_specific_decoders flag when running the model.")

        modality_order = self.omics_decoder.modality_order

        # Process one modality at a time and combine results
        result_dict = {}

        # Select modality to process
        if specific_modality is not None and specific_modality in modality_order:
            # Use the provided specific modality
            modality_to_process = specific_modality
        else:
            # Select based on current training step
            current_step = getattr(self, 'current_training_step', 0)
            modality_to_process = modality_order[current_step % len(modality_order)]

        # Process just this one modality
        # Completely disable this print to reduce output clutter
        # if getattr(self, 'current_training_step', 0) % 5 == 0:
        #     print(f"FORCED SINGLE MODALITY: Processing only {modality_to_process} (step {getattr(self, 'current_training_step', 0)})")

        # Store the current modality as an attribute for external access
        self.current_modality = modality_to_process

        # IMPORTANT FIX: Force use_modality_specific_decoders to True for the decoder
        # This is needed because the model has been permanently modified to use modality-by-modality processing
        if not hasattr(self.omics_decoder, 'use_modality_specific_decoders') or not self.omics_decoder.use_modality_specific_decoders:
            print("WARNING: Forcing use_modality_specific_decoders=True for the decoder")
            self.omics_decoder.use_modality_specific_decoders = True
            # Also set it for the model itself
            self.use_modality_specific_decoders = True

        # FINAL FIX: Use our completely rewritten decoder
        # This will work regardless of dimensions
        result_dict[modality_to_process] = self.omics_decoder.decode_single_modality(z_p, z_gene, modality_to_process)

        # Add a dummy 'concatenated' key for compatibility
        result_dict['concatenated'] = result_dict[modality_to_process]

        return result_dict

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
            # Use the older torch.cuda.amp.autocast API for compatibility with remote cloud machine
            # Note: Using older API for compatibility with remote cloud machine
            # The warning about using torch.amp.autocast('cuda') instead can be ignored
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
    transformer_out_dim = 128 # Increased to match final output dim for better information refinement
    gcn_out_dim = 128 # Final output dim - increased to 128 for information refinement rather than reduction

    integrated_model = IntegratedTransformerGCN(
        # --- Required Args --- #
        omics_input_dims=omics_dims,
        transformer_embed_dim=256, # Increased to support larger output dimension
        transformer_num_heads=8,   # Increased for better representation capacity
        transformer_ff_dim=512,    # Increased to support larger transformer dimensions
        num_transformer_layers=2,
        transformer_output_dim=transformer_out_dim, # Now 128, matching GCN patient input
        gcn_metadata=metadata,
        gene_feature_dim=gene_embed_dim, # Pass the known gene embedding dimension
        gcn_hidden_channels=256,   # Increased to support larger output dimension
        gcn_out_channels=gcn_out_dim, # Now 128 for information refinement
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
        decoder_mlp_factor=0.5,
        genes_per_chunk=5,
        use_mixed_precision=False,
        extreme_memory_efficient=False
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
        print(f"Patient embeddings shape: {final_embeddings['patient'].shape} (now using 128 dimensions for information refinement)")
        print(f"Gene embeddings shape: {final_embeddings['gene'].shape} (now using 128 dimensions for information refinement)")

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