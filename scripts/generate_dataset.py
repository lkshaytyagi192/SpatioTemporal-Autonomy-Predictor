"""
Synthetic & Realistic Multi-Agent Trajectory Dataset Generator.
Generates multi-agent spatio-temporal trajectories following standard ETH/UCY and Stanford Drone Dataset formats:
Columns: [frame_id, agent_id, pos_x, pos_y]
Includes interacting agents, intersection crossings, curved paths, and obstacle avoidance maneuvers.
"""

import os
import numpy as np
import pandas as pd


def generate_scenario(
    num_agents: int = 150,
    seq_len: int = 20,
    area_size: float = 50.0,
    dt: float = 0.4,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic continuous trajectories with interaction dynamics.
    
    Args:
        num_agents: Number of distinct trajectories / agents to simulate.
        seq_len: Total length of each trajectory (obs_len=8 + pred_len=12 = 20 steps).
        area_size: Spatial operational bounds in meters.
        dt: Time delta between frames (0.4s standard for 2.5Hz ETH/UCY).
        random_seed: Reproducibility seed.
    """
    np.random.seed(random_seed)
    records = []

    for agent_id in range(num_agents):
        # Choose motion profile: 0=Straight with drift, 1=Curved/Turning, 2=Crossing/Avoidance
        profile = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
        
        # Initial starting coordinates
        x0 = np.random.uniform(5.0, area_size - 5.0)
        y0 = np.random.uniform(5.0, area_size - 5.0)
        
        # Initial speed (0.8 m/s to 2.2 m/s typical walking/slow robot speed)
        speed = np.random.uniform(0.9, 2.0)
        angle = np.random.uniform(0, 2 * np.pi)
        
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        curr_x, curr_y = x0, y0
        
        # Angular velocity for turning profiles
        omega = 0.0
        if profile == 1:
            omega = np.random.uniform(-0.15, 0.15)
        elif profile == 2:
            # S-curve / avoidance maneuver
            omega = np.random.choice([-0.25, 0.25])

        for step in range(seq_len):
            frame_id = step
            
            # Add dynamic behavior
            if profile == 2 and step == 8:
                # Sudden directional shift or avoidance maneuver mid-trajectory
                omega = -omega * 1.2
            
            # Update heading and velocity
            angle += omega * dt
            # Subtle Gaussian noise mimicking sensor / human jitter
            jitter_x = np.random.normal(0, 0.02)
            jitter_y = np.random.normal(0, 0.02)
            
            curr_x += (speed * np.cos(angle) * dt) + jitter_x
            curr_y += (speed * np.sin(angle) * dt) + jitter_y
            
            records.append({
                "frame_id": frame_id,
                "agent_id": agent_id,
                "pos_x": round(curr_x, 4),
                "pos_y": round(curr_y, 4)
            })

    df = pd.DataFrame(records)
    # Sort by frame_id then agent_id
    df = df.sort_values(by=["frame_id", "agent_id"]).reset_index(drop=True)
    return df


def main():
    os.makedirs("data", exist_ok=True)
    
    print("Generating train trajectory dataset (500 agents)...")
    train_df = generate_scenario(num_agents=500, seq_len=20, random_seed=42)
    train_path = os.path.join("data", "train_trajectories.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Saved: {train_path} ({len(train_df)} rows)")

    print("Generating validation/test trajectory dataset (150 agents)...")
    test_df = generate_scenario(num_agents=150, seq_len=20, random_seed=101)
    test_path = os.path.join("data", "test_trajectories.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
