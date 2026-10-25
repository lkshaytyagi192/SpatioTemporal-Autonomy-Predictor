"""
Interactive Live Multi-Agent Trajectory & Collision Risk Visualizer.
Runs live simulation on screen with real-time trajectory predictions,
attention-based future path projections, and dynamic safety alerts.

Modes:
1. Normal Safe Tracking:
   python demo_live.py
   (Shows 6 multi-agents moving with safe separation >2.0m - ZERO collisions)

2. Critical Collision Hazard:
   python demo_live.py --scenario collision
   (Shows 2 agents crossing intersection on direct collision course with red danger alert & TTC)
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from src.dataset import TrajectoryDataset
from src.seq2seq_attention import Seq2SeqAttentionTrajectory
from src.collision_risk import CollisionRiskEngine


def generate_crossing_scenario():
    """Generates a dynamic 2-agent intersection crossing scenario where collision occurs."""
    dt = 0.4
    seq_len = 20
    
    # Agent 0: Moving horizontally from left to right along Y=25.0
    x0, y0 = 10.0, 25.0
    vx0, vy0 = 1.6, 0.0
    
    # Agent 1: Moving vertically from bottom to top along X=22.8
    x1, y1 = 22.8, 12.0
    vx1, vy1 = 0.0, 1.6

    full_coords = np.zeros((2, seq_len, 2))
    for t in range(seq_len):
        full_coords[0, t] = [x0 + vx0 * t * dt + np.random.normal(0, 0.02), y0 + vy0 * t * dt]
        full_coords[1, t] = [x1 + vx1 * t * dt, y1 + vy1 * t * dt + np.random.normal(0, 0.02)]
    
    obs_abs = full_coords[:, :8, :]
    gt_abs = full_coords[:, 8:, :]
    return obs_abs, gt_abs, full_coords


def launch_live_simulation(scenario: str = "default"):
    is_collision_mode = (scenario.lower() == "collision")
    mode_name = "CRITICAL COLLISION HAZARD" if is_collision_mode else "NORMAL SAFE TRACKING"
    
    print("\n" + "=" * 65)
    print(f"   SPATIO-TEMPORAL AUTONOMY SIMULATOR: [{mode_name}]")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    checkpoint_path = "checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Seq2SeqAttentionTrajectory().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if is_collision_mode:
        obs_abs, gt_abs, full_gt = generate_crossing_scenario()
        num_agents = 2
        last_obs = torch.tensor(obs_abs[:, -1, :], dtype=torch.float32)
        obs_rel = torch.tensor(obs_abs, dtype=torch.float32) - last_obs.unsqueeze(1)
        with torch.no_grad():
            pred_norm, _ = model(obs_rel.to(device))
            pred_abs = (pred_norm.cpu() + last_obs.unsqueeze(1)).numpy()
    else:
        dataset = TrajectoryDataset("data/test_trajectories.csv", normalize_relative=True)
        num_agents = 6
        sample_indices = list(range(num_agents))
        obs_batch = dataset.obs[sample_indices].to(device)
        pred_gt_batch = dataset.pred[sample_indices]
        last_obs_batch = dataset.last_obs[sample_indices]

        with torch.no_grad():
            pred_norm, _ = model(obs_batch)
            pred_abs = (pred_norm.cpu() + last_obs_batch.unsqueeze(1)).numpy()

        obs_abs = (obs_batch.cpu() + last_obs_batch.unsqueeze(1)).numpy()
        gt_abs = (pred_gt_batch + last_obs_batch.unsqueeze(1)).numpy()
        full_gt = np.concatenate([obs_abs, gt_abs], axis=1)

    # Risk Engine Assessment
    risk_engine = CollisionRiskEngine(safety_radius=2.0, critical_radius=1.0)
    risk_events = risk_engine.assess_risk(pred_abs, list(range(num_agents)))

    if is_collision_mode:
        print(f"[!] HAZARD STATUS: {len(risk_events)} Collision Risk Event(s) Detected!")
        for r in risk_events:
            print(f"    --> [{r['risk_level']}] Agent {r['agent_a']} & Agent {r['agent_b']} will collide at step {r['critical_step']}!")
            print(f"        Minimum Separation: {r['min_distance_m']}m | Time-to-Collision (TTC): {r['time_to_collision_s']} seconds")
    else:
        print("[OK] SAFETY STATUS: 0 Collision Risk Events. All agents maintaining safe clearance (>2.0m).")

    # Plot Window Setup
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.canvas.manager.set_window_title(f"SpatioTemporal-Autonomy-Predictor [{mode_name}]")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    margin = 3.5
    x_min, x_max = np.min(full_gt[:, :, 0]) - margin, np.max(full_gt[:, :, 0]) + margin
    y_min, y_max = np.min(full_gt[:, :, 1]) - margin, np.max(full_gt[:, :, 1]) + margin

    def animate(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        is_pred_phase = (frame >= 7)
        phase_str = "PREDICTION PHASE (AI Trajectory Forecasting)" if is_pred_phase else "OBSERVATION PHASE (Tracking Past 3.2s)"
        phase_col = "darkred" if is_pred_phase else "darkblue"

        if is_collision_mode:
            sub_title = "WARNING: Direct Crossing Trajectory Detected!" if is_pred_phase else "Tracking Approaching Agents..."
            sub_col = "red" if is_pred_phase else "gray"
        else:
            sub_title = "STATUS: All Agents Safe (Clearance > 2.0m)"
            sub_col = "green"

        ax.set_title(
            f"Mode: {mode_name} | Step {frame + 1}/20\n[{phase_str}]\n{sub_title}",
            fontsize=11, fontweight="bold", color=phase_col
        )
        ax.set_xlabel("X Metric Position (Meters)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y Metric Position (Meters)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Plot predicted paths in Prediction Phase
        if is_pred_phase:
            for i in range(num_agents):
                # Red dashed AI predicted trajectory
                ax.plot(pred_abs[i, :, 0], pred_abs[i, :, 1], color="red", linestyle="--", linewidth=2.2, alpha=0.85)
                ax.scatter(pred_abs[i, -1, 0], pred_abs[i, -1, 1], color="red", marker="x", s=80, zorder=6)

            # If risk exists, draw forward collision alert
            for r in risk_events:
                step = r["critical_step"]
                crit_pos = (pred_abs[r["agent_a"], step] + pred_abs[r["agent_b"], step]) / 2.0
                danger_circle = patches.Circle(crit_pos, radius=1.2, color="red", fill=True, alpha=0.25, linestyle="--")
                ax.add_patch(danger_circle)
                ax.text(
                    crit_pos[0] + 0.4, crit_pos[1] + 0.4,
                    f"COLLISION PREDICTED!\nTTC: {r['time_to_collision_s']}s (Min Dist: {r['min_distance_m']}m)",
                    color="darkred", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", edgecolor="red")
                )

        # Draw historical trajectories & current positions
        for i in range(num_agents):
            c = colors[i % len(colors)]
            ax.plot(full_gt[i, :frame + 1, 0], full_gt[i, :frame + 1, 1], color=c, linewidth=2.0, alpha=0.7)
            curr = full_gt[i, frame]
            ax.scatter(curr[0], curr[1], color=c, s=130, edgecolors="black", linewidth=1.5, zorder=5)
            ax.text(curr[0] + 0.3, curr[1] + 0.3, f"Agent {i}", fontsize=9, fontweight="bold", color=c)

        # Proximity check at current frame
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                curr_dist = np.linalg.norm(full_gt[i, frame] - full_gt[j, frame])
                if curr_dist < 2.0:
                    mid = (full_gt[i, frame] + full_gt[j, frame]) / 2.0
                    circle = patches.Circle(mid, radius=1.0, color="red", fill=True, alpha=0.3)
                    ax.add_patch(circle)
                    ax.text(
                        mid[0], mid[1] - 1.2, f"IMMINENT COLLISION ({curr_dist:.2f}m)",
                        color="red", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.95)
                    )

        # HUD Box
        hud_lines = [
            "Legend:",
            "- Solid Line: Observed Path",
            "- Red Dashed: AI Prediction",
            "- Red Circle: Collision Hazard"
        ]
        ax.text(
            0.02, 0.98, "\n".join(hud_lines), transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
        )

    anim = FuncAnimation(fig, animate, frames=20, interval=300, repeat=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="default", choices=["default", "collision"])
    args = parser.parse_args()
    launch_live_simulation(scenario=args.scenario)
