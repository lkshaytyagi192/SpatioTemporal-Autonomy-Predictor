"""
Kinematic Baseline: Constant Velocity (CV) / Linear Extrapolation Model.
Provides an empirical, non-deep-learning baseline against which the Deep Learning architecture is benchmarked.
Standard requirement in academic ML research to demonstrate genuine non-linear representation learning.
"""

import torch
import numpy as np


class ConstantVelocityBaseline:
    """
    Extrapolates future trajectory assuming constant velocity derived from the last observed movement vector.
    """
    def __init__(self, pred_len: int = 12):
        self.pred_len = pred_len

    def predict(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_traj: Tensor of shape (Batch_Size, Obs_Len, 2)
        Returns:
            pred_traj: Tensor of shape (Batch_Size, Pred_Len, 2)
        """
        # Velocity vector derived from the last step transition: obs[-1] - obs[-2]
        # shape: (Batch_Size, 2)
        velocity = obs_traj[:, -1, :] - obs_traj[:, -2, :]
        
        batch_size = obs_traj.shape[0]
        device = obs_traj.device
        
        # Steps 1 to pred_len: shape (1, Pred_Len, 1)
        steps = torch.arange(1, self.pred_len + 1, device=device).view(1, -1, 1).float()
        
        # velocity.unsqueeze(1): shape (Batch_Size, 1, 2)
        # displacement: steps * velocity -> (Batch_Size, Pred_Len, 2)
        displacement = steps * velocity.unsqueeze(1)
        
        # Future positions starting from last observed position
        pred_traj = obs_traj[:, -1:, :] + displacement
        return pred_traj
