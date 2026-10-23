"""
Evaluation metrics for trajectory prediction: ADE (Average Displacement Error) and FDE (Final Displacement Error).
These are the standard academic benchmark metrics used in robotics and autonomous systems research.
"""

import torch
import numpy as np


def compute_ade(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Computes Average Displacement Error (ADE).
    ADE is the mean Euclidean distance over all predicted timesteps between predicted and true coordinates.

    Args:
        predicted: Tensor of shape (Batch_Size, Pred_Len, 2)
        ground_truth: Tensor of shape (Batch_Size, Pred_Len, 2)

    Returns:
        float: ADE in coordinate units (meters)
    """
    if isinstance(predicted, np.ndarray):
        diff = predicted - ground_truth
        dist = np.linalg.norm(diff, axis=-1)  # (Batch_Size, Pred_Len)
        return float(np.mean(dist))
    
    # PyTorch Tensor
    diff = predicted - ground_truth
    dist = torch.norm(diff, p=2, dim=-1)  # (Batch_Size, Pred_Len)
    return float(torch.mean(dist).item())


def compute_fde(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Computes Final Displacement Error (FDE).
    FDE is the Euclidean distance at the final predicted timestep (T_pred) between predicted and true coordinates.

    Args:
        predicted: Tensor of shape (Batch_Size, Pred_Len, 2)
        ground_truth: Tensor of shape (Batch_Size, Pred_Len, 2)

    Returns:
        float: FDE in coordinate units (meters)
    """
    if isinstance(predicted, np.ndarray):
        final_diff = predicted[:, -1, :] - ground_truth[:, -1, :]
        dist = np.linalg.norm(final_diff, axis=-1)  # (Batch_Size,)
        return float(np.mean(dist))
    
    # PyTorch Tensor
    final_diff = predicted[:, -1, :] - ground_truth[:, -1, :]
    dist = torch.norm(final_diff, p=2, dim=-1)  # (Batch_Size,)
    return float(torch.mean(dist).item())


def compute_rmse(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Computes Root Mean Square Error (RMSE) across all trajectory coordinates.
    """
    if isinstance(predicted, np.ndarray):
        return float(np.sqrt(np.mean((predicted - ground_truth) ** 2)))
    return float(torch.sqrt(torch.mean((predicted - ground_truth) ** 2)).item())
