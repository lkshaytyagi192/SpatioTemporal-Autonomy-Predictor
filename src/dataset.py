"""
PyTorch Trajectory Dataset and Sliding-Window Sequence Loader.
Processes trajectory coordinate logs into observed sequences (obs_len=8) and future ground truth targets (pred_len=12).
Implements translation normalization (relative displacement offsets) to ensure spatial invariance.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        obs_len: int = 8,
        pred_len: int = 12,
        normalize_relative: bool = True
    ):
        """
        Args:
            csv_path: Path to trajectory CSV with columns [frame_id, agent_id, pos_x, pos_y].
            obs_len: Number of timesteps to observe (past). Default 8 (3.2 seconds).
            pred_len: Number of timesteps to predict (future). Default 12 (4.8 seconds).
            normalize_relative: If True, centers observed and target positions relative to the last observed coordinate.
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.normalize_relative = normalize_relative

        df = pd.read_csv(csv_path)
        
        # Group by agent_id and extract valid sequences
        self.obs_list = []
        self.pred_list = []
        self.last_obs_list = []  # Reference origins for reconstruction

        agents = df["agent_id"].unique()
        for agent in agents:
            agent_data = df[df["agent_id"] == agent].sort_values(by="frame_id")
            coords = agent_data[["pos_x", "pos_y"]].values  # (T, 2)
            
            if len(coords) >= self.seq_len:
                obs_coords = coords[:self.obs_len]
                pred_coords = coords[self.obs_len:self.seq_len]
                last_obs = obs_coords[-1]  # Anchor point (origin)
                
                self.last_obs_list.append(last_obs)
                
                if self.normalize_relative:
                    # Normalized relative to anchor
                    self.obs_list.append(obs_coords - last_obs)
                    self.pred_list.append(pred_coords - last_obs)
                else:
                    self.obs_list.append(obs_coords)
                    self.pred_list.append(pred_coords)

        self.obs = torch.tensor(np.array(self.obs_list), dtype=torch.float32)
        self.pred = torch.tensor(np.array(self.pred_list), dtype=torch.float32)
        self.last_obs = torch.tensor(np.array(self.last_obs_list), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            obs_traj: (obs_len, 2)
            pred_traj: (pred_len, 2)
            last_obs: (2,) reference coordinate
        """
        return self.obs[idx], self.pred[idx], self.last_obs[idx]


def get_dataloader(
    csv_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    obs_len: int = 8,
    pred_len: int = 12
) -> DataLoader:
    """Helper to instantiate a PyTorch DataLoader."""
    dataset = TrajectoryDataset(csv_path, obs_len=obs_len, pred_len=pred_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
