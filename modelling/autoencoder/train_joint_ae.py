import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
import argparse
import os
import numpy as np
import time # Import time
from datetime import datetime
from tqdm.auto import tqdm  # Add tqdm import

# Local imports
from model import JointAutoencoder 
from data_utils import load_prepared_data, JointOmicsDataset, prepare_graph_data

def train_joint_autoencoder(args):
    """Trains the Joint Autoencoder with TensorBoard logging."""

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TensorBoard Setup ---
    # Create a unique log directory for this run
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"{args.cancer_type}_{run_timestamp}")
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logs will be saved to: {log_path}")

    # --- Data Loading and Preparation ---
    print("Loading data...")
    prepared_data = load_prepared_data(args.data_path)
    if not prepared_data or args.cancer_type not in prepared_data:
        print(f"Error: Could not load or find data for {args.cancer_type} in {args.data_path}")
        writer.close()
        return

    cancer_data = prepared_data[args.cancer_type]
    omics_data_dict = cancer_data['omics_data']
    adj_matrix = cancer_data['adj_matrix']
    gene_list = cancer_data['gene_list']
    num_genes = len(gene_list)

    print("Preparing graph data...")
    # Use identity matrix as initial features for the graph AE part
    graph_node_features, graph_edge_index, graph_adj_tensor = prepare_graph_data(adj_matrix, use_identity_features=True)
    graph_feature_dim = graph_node_features.shape[1]

    # Move static graph data to device
    graph_node_features = graph_node_features.to(device)
    graph_edge_index = graph_edge_index.to(device)
    graph_adj_tensor = graph_adj_tensor.to(device) # Target for graph reconstruction loss

    print("Creating JointOmicsDataset...")
    modalities_to_use = args.modalities.split(',') if args.modalities else ['rnaseq', 'methylation', 'scnv', 'miRNA']
    try:
        joint_omics_dataset = JointOmicsDataset(omics_data_dict, gene_list, modalities=modalities_to_use)
    except (ValueError, KeyError) as e:
        print(f"Error creating JointOmicsDataset: {e}")
        writer.close()
        return

    if len(joint_omics_dataset) == 0:
        print("Error: JointOmicsDataset is empty.")
        writer.close()
        return

    num_modalities = joint_omics_dataset.num_modalities
    dataloader = DataLoader(joint_omics_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device == 'cuda' else False)
    print(f"Created DataLoader with {len(dataloader)} batches.")

    # --- Model Instantiation ---
    model = JointAutoencoder(
        num_nodes=num_genes,
        num_modalities=num_modalities,
        graph_feature_dim=graph_feature_dim,
        gene_embedding_dim=args.gene_embedding_dim,
        patient_embedding_dim=args.patient_embedding_dim,
        graph_dropout=args.graph_dropout
    ).to(device)

    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # --- Loss and Optimizer ---
    # Omics Reconstruction Loss 
    omics_loss_fn = F.mse_loss # nn.MSELoss()
    # Graph Reconstruction Loss (BCE for adjacency matrix probabilities)
    graph_loss_fn = F.binary_cross_entropy

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # --- Training Loop ---
    print("\nStarting joint training...")
    global_step = 0
    start_time_total = time.time()
    model.train()
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(args.epochs), desc='Training Epochs', position=0)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        total_loss = 0.0
        total_omics_loss = 0.0
        total_graph_loss = 0.0

        # Create batch progress bar
        batch_pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                         desc=f'Epoch {epoch+1}/{args.epochs}', 
                         leave=False, position=1)

        for batch_idx, omics_batch_structured in batch_pbar:
            omics_batch_structured = omics_batch_structured.to(device)
            omics_reconstructed, adj_reconstructed, z_patient, z_gene = model(
                graph_node_features, graph_edge_index, omics_batch_structured
            )

            loss_o = omics_loss_fn(omics_reconstructed, omics_batch_structured)
            loss_g = graph_loss_fn(adj_reconstructed, graph_adj_tensor)
            combined_loss = args.omics_loss_weight * loss_o + args.graph_loss_weight * loss_g

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            # Update batch progress bar with current loss
            batch_pbar.set_postfix({'loss': f'{combined_loss.item():.4f}'})

            # Log batch losses to TensorBoard
            writer.add_scalar('Loss/Batch/Total', combined_loss.item(), global_step)
            writer.add_scalar('Loss/Batch/Omics', loss_o.item(), global_step)
            writer.add_scalar('Loss/Batch/Graph', loss_g.item(), global_step)

            total_loss += combined_loss.item()
            total_omics_loss += loss_o.item()
            total_graph_loss += loss_g.item()
            global_step += 1

        # Log epoch results
        avg_loss = total_loss / len(dataloader)
        avg_omics_loss = total_omics_loss / len(dataloader)
        avg_graph_loss = total_graph_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time

        # Update epoch progress bar with average losses
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'omics_loss': f'{avg_omics_loss:.4f}',
            'graph_loss': f'{avg_graph_loss:.4f}'
        })

        # Log epoch losses to TensorBoard
        writer.add_scalar('Loss/Epoch/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Epoch/Omics', avg_omics_loss, epoch)
        writer.add_scalar('Loss/Epoch/Graph', avg_graph_loss, epoch)
        writer.add_scalar('Timing/Epoch_Duration_sec', epoch_duration, epoch)
        writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % args.log_interval == 0:
            print(f"\nEpoch [{epoch+1}/{args.epochs}], Avg Total Loss: {avg_loss:.6f}, "
                  f"Avg Omics Loss: {avg_omics_loss:.6f}, Avg Graph Loss: {avg_graph_loss:.6f}, "
                  f"Duration: {epoch_duration:.2f} sec")

    total_training_time = time.time() - start_time_total
    print(f"\nTraining finished. Total duration: {total_training_time:.2f} sec")
    writer.add_scalar('Timing/Total_Training_Duration_sec', total_training_time)

    # Close TensorBoard writer
    writer.close()

    # --- Saving --- #
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Save the trained joint model state dictionary
        model_save_path = os.path.join(args.output_dir, f'joint_ae_model_{args.cancer_type}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Joint model state dict saved to {model_save_path}")

        # Save the final latent embeddings (run inference)
        model.eval()
        all_patient_embeddings = []
        final_gene_embeddings = None
        with torch.no_grad():
            # Get final gene embeddings (only needs to be done once)
            final_gene_embeddings = model.graph_autoencoder.encode(graph_node_features, graph_edge_index).cpu().numpy()

            # Get patient embeddings (process data in batches)
            inference_dataloader = DataLoader(joint_omics_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            for omics_batch_structured in inference_dataloader:
                omics_batch_structured = omics_batch_structured.to(device)
                # Need z_gene on the correct device for the encode step
                z_gene_device = model.graph_autoencoder.encode(graph_node_features, graph_edge_index)
                patient_embeddings = model.omics_processor.encode(omics_batch_structured, z_gene_device)
                all_patient_embeddings.append(patient_embeddings.cpu().numpy())

        final_patient_embeddings = np.concatenate(all_patient_embeddings, axis=0)

        # Save gene embeddings
        gene_emb_save_path = os.path.join(args.output_dir, f'joint_ae_gene_embeddings_{args.cancer_type}.npy')
        np.save(gene_emb_save_path, final_gene_embeddings)
        print(f"Gene embeddings (Z_gene) saved to {gene_emb_save_path}")

        # Save patient embeddings
        patient_emb_save_path = os.path.join(args.output_dir, f'joint_ae_patient_embeddings_{args.cancer_type}.npy')
        np.save(patient_emb_save_path, final_patient_embeddings)
        print(f"Patient embeddings (z_p) saved to {patient_emb_save_path}")

        # Save corresponding patient IDs
        patient_ids_path = os.path.join(args.output_dir, f'joint_ae_patient_ids_{args.cancer_type}.npy')
        np.save(patient_ids_path, np.array(joint_omics_dataset.patient_ids))
        print(f"Patient IDs saved to {patient_ids_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Joint Graph-Omics Autoencoder')

    # Data and Paths
    parser.add_argument('--data_path', type=str, default='data/prepared_data_both.joblib',
                        help='Path to the prepared data joblib file relative to workspace root')
    parser.add_argument('--cancer_type', type=str, default='colorec', choices=['colorec', 'panc'],
                        help='Cancer type to train on')
    parser.add_argument('--modalities', type=str, default='rnaseq,methylation,scnv,miRNA',
                        help='Comma-separated list of omics modalities to use')
    parser.add_argument('--output_dir', type=str, default='../trained_models',
                        help='Directory to save trained models and embeddings')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                        help='Directory to save TensorBoard logs')

    # Model Hyperparameters
    parser.add_argument('--gene_embedding_dim', type=int, default=64,
                        help='Dimension of the latent gene embeddings (Z_gene)')
    parser.add_argument('--patient_embedding_dim', type=int, default=128,
                        help='Dimension of the latent patient embeddings (z_p)')
    parser.add_argument('--graph_dropout', type=float, default=0.5,
                        help='Dropout rate for GCN layers in graph AE')

    # Training Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--omics_loss_weight', type=float, default=1.0,
                        help='Weight for the omics reconstruction loss')
    parser.add_argument('--graph_loss_weight', type=float, default=0.5,
                        help='Weight for the graph reconstruction loss')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='Log training status every n epochs')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of workers for DataLoader')

    args = parser.parse_args()

    # Create output and log directories if they don't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    train_joint_autoencoder(args) 