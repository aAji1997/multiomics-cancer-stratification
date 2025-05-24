"""
This script patches the IntegratedTransformerGCN model to force modality-by-modality processing.
It completely disables the decode_omics method and replaces it with a version that always
processes one modality at a time.

Usage:
1. Import this file before creating the model
2. The patch will be automatically applied

Example:
```python
import modelling.gcn.force_modality_patch  # This will apply the patch
from modelling.gcn.integrated_transformer_gcn_model import IntegratedTransformerGCN
```
"""

import sys
import os
import torch
from typing import Dict, Tuple

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the model class to patch
# We need to import this in a way that avoids circular imports
import modelling.gcn.integrated_transformer_gcn_model
IntegratedTransformerGCN = modelling.gcn.integrated_transformer_gcn_model.IntegratedTransformerGCN

# Store the original decode_omics method
original_decode_omics = IntegratedTransformerGCN.decode_omics

# Define the replacement method that forces modality-by-modality processing
def force_modality_by_modality_decode_omics(self, final_embeddings_dict: Dict[str, torch.Tensor]):
    """
    Forced modality-by-modality version of decode_omics.
    This method always processes one modality at a time, regardless of model settings.

    Args:
        final_embeddings_dict: Dictionary with 'patient' and 'gene' embeddings

    Returns:
        Dictionary of reconstructed modality tensors
    """
    if self.omics_decoder is None:
        raise RuntimeError("Omics decoder was not added to the model.")

    z_p = final_embeddings_dict.get('patient')
    z_gene = final_embeddings_dict.get('gene')

    if z_p is None or z_gene is None:
        raise ValueError("Missing 'patient' or 'gene' embeddings in input dict for decoding.")

    # Get available modalities
    if not hasattr(self.omics_decoder, 'modality_order'):
        raise ValueError("Modality order not found in decoder. Make sure use_modality_specific_decoders=True.")

    modality_order = self.omics_decoder.modality_order

    # Select one modality to process based on current training step
    current_step = getattr(self, 'current_training_step', 0)
    modality_to_process = modality_order[current_step % len(modality_order)]

    print(f"FORCED modality-by-modality: Processing only {modality_to_process} (step {current_step})")

    # Process just this one modality
    result_dict = {}
    result_dict[modality_to_process] = self.decode_single_modality(final_embeddings_dict, modality_to_process)

    # Add a dummy 'concatenated' key for compatibility
    result_dict['concatenated'] = result_dict[modality_to_process]

    return result_dict

# Replace the original method with our forced version
IntegratedTransformerGCN.decode_omics = force_modality_by_modality_decode_omics

print("✅ Applied force_modality_patch: IntegratedTransformerGCN.decode_omics has been patched to force modality-by-modality processing.")
