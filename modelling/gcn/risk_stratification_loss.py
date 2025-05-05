"""
Risk stratification loss function for training models to separate patients into high and low risk groups.
This loss function directly optimizes for log-rank scores by incorporating a differentiable approximation
of the log-rank test statistic, along with encouraging the model to learn embeddings
that can be effectively separated into distinct risk groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import math

class RiskStratificationLoss(nn.Module):
    """
    Loss function that encourages patient embeddings to form distinct clusters
    that would yield good separation in survival analysis.

    This loss directly optimizes for the log-rank test statistic by:
    1. Clustering patient embeddings into two groups (high/low risk)
    2. Maximizing the distance between cluster centers
    3. Minimizing the variance within each cluster
    4. Encouraging balanced cluster sizes
    5. Directly optimizing a differentiable approximation of the log-rank test statistic

    Args:
        balance_weight (float): Weight for the cluster balance term
        variance_weight (float): Weight for the within-cluster variance term
        separation_weight (float): Weight for the between-cluster separation term
        logrank_weight (float): Weight for the log-rank approximation term
        temperature (float): Temperature parameter for soft clustering
        survival_data (tuple, optional): Tuple of (times, events) for direct log-rank optimization
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

    def _differentiable_logrank_with_survival_data(self, cluster_probs, times, events):
        """
        Compute a more accurate differentiable approximation of the log-rank test statistic
        when survival times and events are available.

        This implementation is closer to the actual log-rank test used in survival analysis.

        Args:
            cluster_probs (torch.Tensor): Soft cluster assignment probabilities [batch_size, 2]
            times (torch.Tensor): Survival times [batch_size]
            events (torch.Tensor): Event indicators (1=event, 0=censored) [batch_size]

        Returns:
            torch.Tensor: Negative approximation of log-rank test statistic (to minimize)
        """
        device = cluster_probs.device

        # Sort by time (ascending)
        sorted_indices = torch.argsort(times)
        sorted_times = times[sorted_indices]
        sorted_events = events[sorted_indices]
        sorted_probs = cluster_probs[sorted_indices]

        # Get unique time points where events occurred
        unique_times = torch.unique(sorted_times[sorted_events == 1])

        # Initialize variables for log-rank calculation
        observed_minus_expected = torch.zeros(1, device=device)
        variance_sum = torch.zeros(1, device=device)

        # For each time point, calculate the contribution to the log-rank statistic
        for t in unique_times:
            # Patients at risk at time t (patients with time >= t)
            at_risk_mask = sorted_times >= t
            n_at_risk = at_risk_mask.sum()

            if n_at_risk < 2:
                continue  # Skip if fewer than 2 patients at risk

            # Events at time t
            events_mask = (sorted_times == t) & (sorted_events == 1)
            n_events = events_mask.sum()

            if n_events == 0:
                continue  # Skip if no events at this time

            # Calculate weighted number of patients in group 0 at risk
            n1_at_risk = torch.sum(sorted_probs[at_risk_mask, 0])

            # Expected number of events in group 0
            e1 = n_events * (n1_at_risk / n_at_risk)

            # Observed number of events in group 0 (weighted by probability)
            o1 = torch.sum(sorted_probs[events_mask, 0])

            # Contribution to the log-rank statistic
            diff = o1 - e1

            # Variance calculation
            n2_at_risk = n_at_risk - n1_at_risk  # Weighted number in group 1
            variance = n_events * (n1_at_risk / n_at_risk) * (n2_at_risk / n_at_risk) * (n_at_risk - n_events) / (n_at_risk - 1 + 1e-8)

            # Accumulate the difference and variance
            observed_minus_expected += diff
            variance_sum += variance

        # Calculate the log-rank test statistic
        # Higher values indicate better separation (lower p-values)
        if variance_sum > 1e-8:
            logrank_stat = (observed_minus_expected ** 2) / variance_sum
        else:
            # Fallback to a simpler approximation if variance is too small
            logrank_stat = torch.abs(observed_minus_expected) + 1e-8

        # Invert since we want to maximize the statistic but minimize the loss
        return -logrank_stat

    def _differentiable_logrank_loss(self, cluster_probs):
        """
        Compute a differentiable approximation of the log-rank test statistic.

        The log-rank test measures the difference between observed and expected events
        across groups. A higher test statistic (and lower p-value) indicates better separation.

        Since we don't have actual survival times during training, we approximate the
        log-rank test statistic based on the cluster probabilities.

        Args:
            cluster_probs (torch.Tensor): Soft cluster assignment probabilities [batch_size, 2]

        Returns:
            torch.Tensor: Negative approximation of log-rank test statistic (to minimize)
        """

        # We want to maximize the separation between clusters, which correlates with
        # a higher log-rank test statistic (lower p-value)

        # Calculate entropy of cluster assignments - lower entropy means more confident assignments
        # which typically leads to better separation in survival curves
        entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1).mean()

        # Calculate the "sharpness" of the cluster assignments
        # Higher values indicate more distinct clusters
        sharpness = torch.mean(torch.max(cluster_probs, dim=1)[0])

        # Calculate the variance of cluster probabilities
        # Higher variance between cluster probabilities indicates better separation
        cluster_mean = torch.mean(cluster_probs, dim=0)
        cluster_var = torch.mean(torch.pow(cluster_probs - cluster_mean.unsqueeze(0), 2))

        # More sophisticated approximation of log-rank test statistic
        # The log-rank test statistic is approximately (O1-E1)^2/V1 where:
        # O1 = observed events in group 1
        # E1 = expected events in group 1
        # V1 = variance of the difference between observed and expected

        # Since we don't have actual survival times, we use the cluster probabilities
        # to approximate the log-rank test statistic

        # Approximate the difference between observed and expected events
        # by the difference in cluster assignment probabilities
        diff = cluster_probs[:, 0] - cluster_probs[:, 1]

        # Square the difference and normalize by the variance
        # This is similar to the form of the log-rank test statistic
        diff_squared = torch.pow(diff, 2)
        diff_var = torch.var(diff) + 1e-8  # Add epsilon to avoid division by zero

        # Approximate log-rank test statistic
        # Higher values indicate better separation (lower p-values)
        batch_size = cluster_probs.shape[0]
        logrank_stat_approx = torch.sum(diff_squared) / (diff_var * batch_size)

        # Combine the metrics to create a comprehensive loss
        # We want to minimize entropy and maximize sharpness, variance, and logrank_stat_approx
        basic_approx = entropy - sharpness - cluster_var

        # Invert logrank_stat_approx since we want to maximize it but minimize the loss
        logrank_approx = basic_approx - logrank_stat_approx

        return logrank_approx

    def forward(self, embeddings):
        """
        Calculate the risk stratification loss.

        Args:
            embeddings (torch.Tensor): Patient embeddings of shape [batch_size, embedding_dim]

        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size, embedding_dim = embeddings.shape
        device = embeddings.device

        # Ensure we have enough samples for clustering
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings for cosine similarity
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)

        # Step 1: Perform soft clustering into 2 groups
        # If we don't have cluster centers or they're on a different device, initialize them
        if self.cluster_centers is None or self.cluster_centers.device != device:
            # Initialize cluster centers randomly and detach them from the computation graph
            with torch.no_grad():
                random_centers = torch.randn(2, embedding_dim, device=device)
                self.cluster_centers = F.normalize(random_centers, p=2, dim=1).detach().clone()

        # Calculate distances to cluster centers
        # Make a copy of the centers to use in the forward pass to avoid modifying the original
        centers_for_forward = self.cluster_centers.detach().clone()
        similarities = torch.mm(embeddings_normalized, centers_for_forward.t())  # [batch_size, 2]

        # Convert similarities to probabilities with temperature
        cluster_probs = F.softmax(similarities / self.temperature, dim=1)  # [batch_size, 2]

        # Step 2: Update cluster centers with soft assignments (outside of computation graph)
        with torch.no_grad():
            # Create a new tensor for updated centers
            updated_centers = torch.zeros_like(self.cluster_centers)

            for k in range(2):
                # Weighted sum of embeddings based on probabilities
                weighted_sum = torch.sum(cluster_probs[:, k].unsqueeze(1) * embeddings_normalized, dim=0)
                # Normalize to unit length
                if torch.sum(cluster_probs[:, k]) > 1e-6:  # Avoid division by zero
                    new_center = weighted_sum / torch.sum(cluster_probs[:, k])
                    new_center = F.normalize(new_center, p=2, dim=0)
                    # Update with momentum to stabilize training
                    updated_centers[k] = 0.9 * self.cluster_centers[k] + 0.1 * new_center
                else:
                    # Keep the original center if no points are assigned to this cluster
                    updated_centers[k] = self.cluster_centers[k]

            # After the loop, update the centers
            self.cluster_centers = updated_centers.detach().clone()

        # Step 3: Calculate loss components

        # 3.1: Cluster balance loss - encourages equal-sized clusters
        cluster_sizes = torch.sum(cluster_probs, dim=0)  # [2]
        balance_loss = torch.var(cluster_sizes) / (torch.mean(cluster_sizes) + 1e-8)

        # 3.2: Within-cluster variance loss - encourages tight clusters
        variance_losses = []
        for k in range(2):
            # Calculate weighted distance from each point to its cluster center
            # Use centers_for_forward to avoid modifying the original centers
            cluster_dists = 1 - torch.mm(embeddings_normalized, centers_for_forward[k].unsqueeze(1))
            weighted_dists = cluster_probs[:, k] * cluster_dists.squeeze()
            if torch.sum(cluster_probs[:, k]) > 1e-6:
                cluster_variance = torch.sum(weighted_dists) / torch.sum(cluster_probs[:, k])
                variance_losses.append(cluster_variance)

        # Sum the variance losses (if any)
        if variance_losses:
            variance_loss = torch.stack(variance_losses).sum()
        else:
            variance_loss = torch.tensor(0.0, device=device)

        # 3.3: Between-cluster separation loss - encourages distant clusters
        # Use centers_for_forward to avoid modifying the original centers
        center_similarity = torch.mm(centers_for_forward[0].unsqueeze(0),
                                    centers_for_forward[1].unsqueeze(1))
        separation_loss = torch.exp(-torch.clamp(1 - center_similarity, min=0))

        # 3.4: Log-rank approximation loss - directly optimizes for log-rank test statistic
        if self.survival_data is not None:
            # If survival data is provided, use the more accurate approximation
            times, events = self.survival_data
            logrank_loss = self._differentiable_logrank_with_survival_data(cluster_probs, times, events)
        else:
            # Otherwise, use the basic approximation
            logrank_loss = self._differentiable_logrank_loss(cluster_probs)

        # Ensure all loss components are scalar tensors
        if balance_loss.dim() > 0:
            balance_loss = balance_loss.mean()
        if variance_loss.dim() > 0:
            variance_loss = variance_loss.mean()
        if separation_loss.dim() > 0:
            separation_loss = separation_loss.mean()
        if logrank_loss.dim() > 0:
            logrank_loss = logrank_loss.mean()

        # Combine loss components without in-place operations
        weighted_balance = self.balance_weight * balance_loss
        weighted_variance = self.variance_weight * variance_loss
        weighted_separation = self.separation_weight * separation_loss
        weighted_logrank = self.logrank_weight * logrank_loss

        # Sum the weighted components
        total_loss = weighted_balance + weighted_variance + weighted_separation + weighted_logrank

        # Ensure the final loss is a scalar
        if total_loss.dim() > 0:
            total_loss = total_loss.mean()

        return total_loss

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
        logrank_weight (float): Weight for the log-rank approximation term
        temperature (float): Temperature parameter for soft clustering
        survival_times (torch.Tensor, optional): Survival times for each patient
        survival_events (torch.Tensor, optional): Event indicators (1=event, 0=censored) for each patient

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
