import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import math

class RiskStratificationLoss(nn.Module):
    """
    Loss function that encourages patient embeddings to form distinct clusters
    
    1. Clustering patient embeddings into two groups (high/low risk)
    2. Maximizing the distance between cluster centers
    3. Minimizing the variance within each cluster
    4. Encouraging balanced cluster sizes
    Args:
        balance_weight (float): Weight for the cluster balance term
        variance_weight (float): Weight for the within-cluster variance term
        separation_weight (float): Weight for the between-cluster separation term
        temperature (float): Temperature parameter for soft clustering
    """
    def __init__(self, balance_weight=0.2, variance_weight=0.2, separation_weight=0.2,
                 logrank_weight=0.4, temperature=0.1, survival_data=None):
        super(RiskStratificationLoss, self).__init__()
        self.balance_weight = balance_weight
        self.variance_weight = variance_weight
        self.separation_weight = separation_weight
        self.logrank_weight = logrank_weight
        self.temperature = temperature
        self.survival_data = survival_data

        # Initialize cluster centers (will be updated during forward pass)
        self.cluster_centers = None




def risk_stratification_loss(embeddings, balance_weight=0.2, variance_weight=0.2,
                            separation_weight=0.2, logrank_weight=0.7, temperature=0.1,
                            survival_times=None, survival_events=None):
    """
    Functional interface for the risk stratification loss.

    Args:
        embeddings (torch.Tensor): Patient embeddings of shape [batch_size, embedding_dim]
        balance_weight (float): Weight for the cluster balance term
        variance_weight (float): Weight for the within-cluster variance term
        separation_weight (float): Weight for the between-cluster separation term
        temperature (float): Temperature parameter for soft clustering

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Check if we have enough samples for clustering (at least 2)
    if embeddings.shape[0] < 2:
        # Return zero loss if not enough samples
        return torch.tensor(0.0, device=embeddings.device)

    # Prepare survival data if provided
    survival_data = None
    if survival_times is not None and survival_events is not None:
        # Ensure they're on the same device as embeddings
        survival_times = survival_times.to(embeddings.device)
        survival_events = survival_events.to(embeddings.device)
        survival_data = (survival_times, survival_events)

    # Create a temporary loss module with a unique name to avoid sharing state between calls
    # This prevents in-place operations from affecting the backward pass
    loss_fn = RiskStratificationLoss(
        balance_weight=balance_weight,
        variance_weight=variance_weight,
        separation_weight=separation_weight,
        logrank_weight=logrank_weight,
        temperature=temperature,
        survival_data=survival_data
    )
    # Move to the same device as the embeddings
    loss_fn = loss_fn.to(embeddings.device)

    # Calculate loss with no_grad for the cluster center updates
    with torch.no_grad():
        # Initialize cluster centers (this won't be part of the computation graph)
        if loss_fn.cluster_centers is None:
            random_centers = torch.randn(2, embeddings.shape[1], device=embeddings.device)
            loss_fn.cluster_centers = torch.nn.functional.normalize(random_centers, p=2, dim=1).detach()

    # Now compute the actual loss (this will be part of the computation graph)
    loss = loss_fn(embeddings)

    # Ensure the loss is a scalar tensor with shape []
    if loss.dim() > 0:
        loss = loss.mean()

    return loss
