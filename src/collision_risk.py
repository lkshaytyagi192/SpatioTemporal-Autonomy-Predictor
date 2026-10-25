"""
Spatio-Temporal Dynamic Collision Risk Assessment Engine.
Performs pairwise proximity analysis across predicted trajectories of multi-agent autonomous systems.
Calculates minimum predicted separation, Time-to-Collision (TTC), and Collision Risk Index (CRI).
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple


class CollisionRiskEngine:
    def __init__(
        self,
        safety_radius: float = 1.5,     # meters - Caution buffer
        critical_radius: float = 0.8,   # meters - Immediate collision threshold
        dt: float = 0.4                 # seconds per prediction step
    ):
        self.safety_radius = safety_radius
        self.critical_radius = critical_radius
        self.dt = dt

    def assess_risk(
        self,
        predicted_trajectories: np.ndarray,
        agent_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Assesses pairwise collision risk for all agent combinations over predicted horizons.

        Args:
            predicted_trajectories: (Num_Agents, Pred_Len, 2) in absolute metric coordinates
            agent_ids: List of agent identifiers corresponding to row indices

        Returns:
            List of risk events with metadata [Agent A, Agent B, Min Dist, TTC, Risk Level, CRI]
        """
        num_agents, pred_len, _ = predicted_trajectories.shape
        risk_reports = []

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                traj_i = predicted_trajectories[i]  # (Pred_Len, 2)
                traj_j = predicted_trajectories[j]  # (Pred_Len, 2)

                # Pairwise Euclidean distances across all future timesteps
                distances = np.linalg.norm(traj_i - traj_j, axis=1)  # (Pred_Len,)
                min_dist = float(np.min(distances))
                min_timestep = int(np.argmin(distances))

                # Collision Risk Index (CRI) normalized between 0.0 and 1.0
                if min_dist < self.critical_radius:
                    risk_level = "CRITICAL_COLLISION"
                    cri = 1.0
                elif min_dist < self.safety_radius:
                    risk_level = "CAUTION_PROXIMITY"
                    cri = round(1.0 - (min_dist - self.critical_radius) / (self.safety_radius - self.critical_radius), 3)
                else:
                    risk_level = "SAFE"
                    cri = 0.0

                ttc = round((min_timestep + 1) * self.dt, 2)  # In seconds

                if risk_level != "SAFE":
                    risk_reports.append({
                        "agent_a": agent_ids[i],
                        "agent_b": agent_ids[j],
                        "min_distance_m": round(min_dist, 3),
                        "time_to_collision_s": ttc,
                        "risk_level": risk_level,
                        "collision_risk_index": cri,
                        "critical_step": min_timestep
                    })

        # Sort descending by risk index
        risk_reports.sort(key=lambda x: x["collision_risk_index"], reverse=True)
        return risk_reports
