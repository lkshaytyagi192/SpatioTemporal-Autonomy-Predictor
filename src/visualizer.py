"""
Trajectory and Collision Risk Visualizer.
Generates publication-quality 2D multi-agent trajectory maps, attention heatmaps,
and dynamic GIF animations showing predicted paths and dynamic collision alert bubbles.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, List, Dict, Any

from src.dataset import TrajectoryDataset
from src.seq2seq_attention import Seq2SeqAttentionTrajectory
from src.collision_risk import CollisionRiskEngine


def render_evaluation_dashboard(
    model_checkpoint: str = "checkpoints/best_model.pt",
    test_csv: str = "data/test_trajectories.csv",
    output_image: str = "results/trajectory_collision_demo.png",
    num_agents_to_plot: int = 4
):
    """
    Renders a comprehensive 3-panel technical evaluation figure:
    - Panel 1: Spatial 2D Trajectories (Observed, Ground Truth, Predicted, Risk Buffer)
    - Panel 2: Temporal Pairwise Proximity Profile vs Safety Thresholds
    - Panel 3: Encoder-Decoder Attention Alignment Heatmap
    """
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset & model
    dataset = TrajectoryDataset(test_csv, normalize_relative=True)
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
    
    model = Seq2SeqAttentionTrajectory().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Select agents for clear visual demonstration
    sample_indices = list(range(min(num_agents_to_plot, len(dataset))))
    obs_batch = dataset.obs[sample_indices].to(device)
    pred_gt_batch = dataset.pred[sample_indices]
    last_obs_batch = dataset.last_obs[sample_indices]

    with torch.no_grad():
        pred_norm, attn_history = model(obs_batch)
        pred_norm = pred_norm.cpu()
        attn_history = attn_history.cpu()

    # Reconstruct absolute metric coordinates
    # Absolute = Relative_coords + last_obs_anchor
    obs_abs = (obs_batch.cpu() + last_obs_batch.unsqueeze(1)).numpy()
    gt_abs = (pred_gt_batch + last_obs_batch.unsqueeze(1)).numpy()
    pred_abs = (pred_norm + last_obs_batch.unsqueeze(1)).numpy()

    # Run collision risk engine
    risk_engine = CollisionRiskEngine(safety_radius=2.0, critical_radius=1.0)
    risk_reports = risk_engine.assess_risk(pred_abs, sample_indices)

    # ------------------ Plotting Dashboard ------------------ #
    fig = plt.figure(figsize=(16, 5), dpi=180)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])

    # --- Panel 1: 2D Spatial Trajectory Map --- #
    ax1 = fig.add_subplot(gs[0])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, agent_id in enumerate(sample_indices):
        c = colors[idx % len(colors)]
        # Observed path (past 8 steps)
        ax1.plot(obs_abs[idx, :, 0], obs_abs[idx, :, 1], color=c, linestyle="-", linewidth=2.0, label=f"Agent {agent_id} (Obs)")
        ax1.scatter(obs_abs[idx, :, 0], obs_abs[idx, :, 1], color=c, s=18, alpha=0.7)
        # Anchor point
        ax1.scatter(obs_abs[idx, -1, 0], obs_abs[idx, -1, 1], color=c, s=60, marker="o", edgecolors="black", zorder=5)

        # Ground truth future path (12 steps)
        ax1.plot(gt_abs[idx, :, 0], gt_abs[idx, :, 1], color=c, linestyle=":", linewidth=1.8, alpha=0.6)
        
        # Predicted future path (12 steps)
        ax1.plot(pred_abs[idx, :, 0], pred_abs[idx, :, 1], color=c, linestyle="--", linewidth=2.4)
        ax1.scatter(pred_abs[idx, -1, 0], pred_abs[idx, -1, 1], color=c, s=60, marker="X", edgecolors="black", zorder=5)

    # Highlight active collision alerts
    for risk in risk_reports:
        a_idx = risk["agent_a"]
        step = risk["critical_step"]
        crit_pos = pred_abs[a_idx, step]
        alert_circle = patches.Circle(crit_pos, radius=risk_engine.critical_radius, color="red", fill=True, alpha=0.25, linestyle="--", linewidth=1.5)
        ax1.add_patch(alert_circle)
        ax1.text(crit_pos[0] + 0.3, crit_pos[1] + 0.3, f"ALERT: A{risk['agent_a']}-A{risk['agent_b']}\nTTC={risk['time_to_collision_s']}s",
                 color="darkred", fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8))

    ax1.set_title("Autonomous Trajectory Forecast & Collision Zones", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Metric X Position (m)", fontsize=10)
    ax1.set_ylabel("Metric Y Position (m)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --- Panel 2: Temporal Proximity Profile (TTC Analysis) --- #
    ax2 = fig.add_subplot(gs[1])
    time_steps = np.arange(1, 13) * 0.4  # in seconds

    for i in range(len(sample_indices)):
        for j in range(i + 1, len(sample_indices)):
            pair_dist = np.linalg.norm(pred_abs[i] - pred_abs[j], axis=1)
            ax2.plot(time_steps, pair_dist, linewidth=2, label=f"Pair A{i}-A{j}")

    ax2.axhline(risk_engine.safety_radius, color="orange", linestyle="--", linewidth=1.5, label="Caution Limit (1.5m)")
    ax2.axhline(risk_engine.critical_radius, color="red", linestyle="-.", linewidth=1.5, label="Critical Collision (0.8m)")
    ax2.set_title("Pairwise Separation Distance over Horizon", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Prediction Horizon (seconds)", fontsize=10)
    ax2.set_ylabel("Inter-Agent Distance (m)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # --- Panel 3: Temporal Attention Alignment Heatmap --- #
    ax3 = fig.add_subplot(gs[2])
    # Average attention across predicted batch: (Pred_Len=12, Obs_Len=8)
    avg_attention = attn_history.mean(dim=0).numpy()
    cax = ax3.imshow(avg_attention, cmap="magma", aspect="auto")
    fig.colorbar(cax, ax=ax3, fraction=0.046, pad=0.04)

    ax3.set_title("Temporal Self-Attention Alignment Matrix", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Observed Past Timestep (t=1..8)", fontsize=10)
    ax3.set_ylabel("Forecasted Future Timestep (t=9..20)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    plt.close()
    print(f"[OK] Evaluation dashboard rendered: {output_image}")


def render_trajectory_animation(
    model_checkpoint: str = "checkpoints/best_model.pt",
    test_csv: str = "data/test_trajectories.csv",
    output_gif: str = "results/trajectory_animation.gif",
    num_agents: int = 5
):
    """
    Renders an animated dynamic GIF showing live trajectory movement and dynamic collision alerts.
    """
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(test_csv, normalize_relative=True)
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
    
    model = Seq2SeqAttentionTrajectory().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_indices = list(range(min(num_agents, len(dataset))))
    obs_batch = dataset.obs[sample_indices].to(device)
    pred_gt_batch = dataset.pred[sample_indices]
    last_obs_batch = dataset.last_obs[sample_indices]

    with torch.no_grad():
        pred_norm, _ = model(obs_batch)
        pred_norm = pred_norm.cpu()

    # Full sequences: (Num_Agents, 20, 2)
    obs_abs = (obs_batch.cpu() + last_obs_batch.unsqueeze(1)).numpy()
    pred_abs = (pred_norm + last_obs_batch.unsqueeze(1)).numpy()
    full_gt = np.concatenate([obs_abs, (pred_gt_batch + last_obs_batch.unsqueeze(1)).numpy()], axis=1)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def update(frame):
        ax.clear()
        ax.set_xlim(np.min(full_gt[:, :, 0]) - 2, np.max(full_gt[:, :, 0]) + 2)
        ax.set_ylim(np.min(full_gt[:, :, 1]) - 2, np.max(full_gt[:, :, 1]) + 2)
        ax.set_title(f"Multi-Agent Autonomy Simulation (Step {frame}/20)", fontsize=11, fontweight="bold")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.grid(True, linestyle=":", alpha=0.5)

        for i in range(num_agents):
            c = colors[i % len(colors)]
            # Path up to current frame
            ax.plot(full_gt[i, :frame + 1, 0], full_gt[i, :frame + 1, 1], color=c, alpha=0.6, linestyle="-")
            # Current agent position
            curr_pos = full_gt[i, frame]
            ax.scatter(curr_pos[0], curr_pos[1], color=c, s=70, edgecolors="black", label=f"Agent {i}")

            # If past observation phase (frame >= 8), display AI predicted trajectory
            if frame >= 7:
                ax.plot(pred_abs[i, :, 0], pred_abs[i, :, 1], color="red", linestyle="--", alpha=0.8, linewidth=1.5)

        # Proximity check at current frame
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(full_gt[i, frame] - full_gt[j, frame])
                if dist < 1.5:
                    mid_pt = (full_gt[i, frame] + full_gt[j, frame]) / 2.0
                    circle = patches.Circle(mid_pt, radius=1.0, color="red", fill=True, alpha=0.25)
                    ax.add_patch(circle)
                    ax.text(mid_pt[0], mid_pt[1], f"PROXIMITY ALERT ({dist:.2f}m)", color="red", fontsize=8, fontweight="bold")

    anim = FuncAnimation(fig, update, frames=20, interval=250)
    writer = PillowWriter(fps=4)
    anim.save(output_gif, writer=writer)
    plt.close()
    print(f"[OK] Trajectory simulation GIF generated: {output_gif}")


if __name__ == "__main__":
    render_evaluation_dashboard()
    render_trajectory_animation()
